import os
import re
import json
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from pymongo import MongoClient, UpdateOne
from volcenginesdkarkruntime import Ark

# =========================
# 0) 配置区
# =========================
EMBED_MODEL = os.getenv("EMBED_MODEL", "doubao-embedding-vision-250615")
MRL_DIM = int(os.getenv("MRL_DIM", "1024"))

# 检索相关
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "10"))
FINAL_N = int(os.getenv("FINAL_N", "4"))
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vec_index")
VECTOR_PATH = os.getenv("VECTOR_PATH", "embedding")
VECTOR_CANDIDATES = int(os.getenv("VECTOR_CANDIDATES", "200"))
PY_RERANK_LIMIT = int(os.getenv("PY_RERANK_LIMIT", "500"))

# Agent 回合
MAX_POLICY_ROUNDS = int(os.getenv("MAX_POLICY_ROUNDS", "4"))  # Lesson10 policy loop rounds
TOPK_INIT = int(os.getenv("TOPK_INIT", "8"))
TOPK_CAP = int(os.getenv("TOPK_CAP", "20"))

# 模型
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "doubao-pro-32k")  # 用于 QA / Critic / JSON 解析
POLICY_MODEL = os.getenv("POLICY_MODEL", "doubao-pro-32k")  # 用于 policy 决策
DEFAULT_VISIBILITY = os.getenv("DEFAULT_VISIBILITY", "internal")


# =========================
# 1) 数据结构
# =========================
@dataclass
class Evidence:
    id: str
    source: str  # 展示用：publish_at | title | section_path
    text: str
    score: float
    source_id: str  # 去重用：建议=article_id（或 article_id#section）
    article_id: str
    title: str
    publish_at: str
    section_path: str


@dataclass
class AgentState:
    # policy stats
    retrieve_rounds: int = 0
    rewrite_times: int = 0
    top_k: int = TOPK_INIT
    decision_trace: List[str] = field(default_factory=list)

    # evidence de-dup
    seen_source_ids: Set[str] = field(default_factory=set)

    # notes
    notes: List[str] = field(default_factory=list)


# =========================
# 2) 客户端初始化
# =========================
def get_mongo_collection():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "rag")
    col_name = os.getenv("MONGO_COL", "article_chunks")
    client = MongoClient(uri)
    return client[db_name][col_name]


def get_ark_client() -> Ark:
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing env ARK_API_KEY")
    return Ark(api_key=api_key)


col = get_mongo_collection()
ark = get_ark_client()


# =========================
# 3) Embedding：豆包向量
# =========================
def embed_text(texts: List[str]) -> List[List[float]]:
    """
    统一把 multimodal_embeddings 的返回解析成 List[List[float]]

    调用Ark embeddings 也算外部服务
    """
    mm_input = [{"type": "text", "text": t} for t in texts]
    resp = ark.multimodal_embeddings.create(model=EMBED_MODEL, input=mm_input)

    data = getattr(resp, "data", None)
    if data is None:
        return []

    emb = getattr(data, "embedding", None)
    if emb is not None:
        # 单条向量
        if isinstance(emb, list) and (len(emb) == 0 or isinstance(emb[0], (int, float))):
            return [emb]
        # 多条向量
        if isinstance(emb, list) and (len(emb) == 0 or isinstance(emb[0], list)):
            return emb

    # 某些 SDK 返回 list[data_item]
    if isinstance(data, list):
        out = []
        for item in data:
            e = getattr(item, "embedding", None) or (item.get("embedding") if isinstance(item, dict) else None)
            if e is not None:
                out.append(e)
        return out

    return []


def encode_texts(texts: List[str], is_query: bool, mrl_dim: int = MRL_DIM) -> np.ndarray:
    if not texts:
        return np.zeros((0, mrl_dim), dtype=np.float32)

    if is_query:
        texts = [
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            f"Query: {t}"
            for t in texts
        ]

    try:
        emb_list = embed_text(texts)
        if len(emb_list) != len(texts):
            if len(emb_list) == 0:
                emb_list = [[0.0] * mrl_dim for _ in texts]
            elif len(emb_list) < len(texts):
                dim = len(emb_list[0])
                emb_list = emb_list + [[0.0] * dim for _ in range(len(texts) - len(emb_list))]
            else:
                emb_list = emb_list[: len(texts)]

        emb = torch.tensor(emb_list, dtype=torch.float32)

        if mrl_dim is not None:
            assert mrl_dim in [2048, 1024, 512, 256]
            emb = emb[:, :mrl_dim]

        emb = torch.nn.functional.normalize(emb, dim=1, p=2).cpu().numpy().astype(np.float32)
        return emb
    except Exception:
        return np.zeros((len(texts), mrl_dim), dtype=np.float32)


def encode_query(query: str) -> np.ndarray:
    """
    调用 Ark embedding
    """
    return encode_texts([query], is_query=True, mrl_dim=MRL_DIM)[0]


# =========================
# 4) 索引建议
# =========================
def ensure_basic_indexes():
    col.create_index([("visibility", 1)])
    col.create_index([("publish_at", 1)])
    col.create_index([("section_path", 1)])
    col.create_index([("source_id", 1)])
    col.create_index([("article_id", 1)])


def create_index_text():
    col.create_index([
        ("title", "text"),
    ])
    col.create_index([
        ("text", "text"),
    ])
    col.create_index([
        ("section_path", "text"),
    ])


# =========================
# 5) 写入：稿件 chunk -> embedding -> upsert Mongo
# =========================
def upsert_chunks(chunks: List[Dict[str, Any]]):
    """
    期望每个 chunk 至少包含：
      _id, article_id, title, publish_at, section_path, text, source_id, visibility
    """
    if not chunks:
        return

    texts = []
    for c in chunks:
        title = str(c.get("title", "") or "")
        section_path = str(c.get("section_path", "") or "")
        text = str(c.get("text", "") or "")
        texts.append(f"{title}\n{section_path}\n{text}")

    vecs = encode_texts(texts, is_query=False, mrl_dim=MRL_DIM)

    ops = []
    for c, v in zip(chunks, vecs):
        doc = dict(c)
        doc.setdefault("visibility", DEFAULT_VISIBILITY)
        doc.setdefault("source_id", str(doc.get("article_id", doc.get("_id"))))
        doc["embedding_model"] = EMBED_MODEL
        doc["embedding_dim"] = int(MRL_DIM)
        doc["embedding"] = v.tolist()
        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))

    col.bulk_write(ops, ordered=False)


# =========================
# 6) 检索：优先 $vectorSearch；否则 Python 精排
# =========================
def _build_filter(
        project_keyword: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        exclude_source_ids: Set[str],
        article_id: Optional[str] = None,
) -> Dict[str, Any]:
    f: Dict[str, Any] = {"visibility": DEFAULT_VISIBILITY}

    # project_keyword 用作栏目/分类过滤（section_path 模糊匹配）
    if project_keyword:
        f["section_path"] = {"$regex": project_keyword}

    # 只在某篇稿件内问答
    if article_id:
        f["article_id"] = article_id

    if date_from or date_to:
        f["publish_at"] = {}
        if date_from:
            f["publish_at"]["$gte"] = date_from
        if date_to:
            f["publish_at"]["$lte"] = date_to

    if exclude_source_ids:
        f["source_id"] = {"$nin": list(exclude_source_ids)}

    return f


def rerank_evidences_llm(
        query: str,
        evidences: List[Dict[str, Any]],
        top_k: int = 5,
        max_text_len: int = 300,
) -> List[Dict[str, Any]]:
    """
    使用 LLM 对 merge 后的 evidences 做 rerank
    输入 evidence 至少包含:
    {
        "source_id":str,
        "text":str,
        "scores":List[float],
        "queries_hit":List[str]
    }
    返回按 rerank 后顺序排序的 top_k evidences
    """
    query = query.strip()
    if not query:
        return []

    if not evidences:
        return []

    if top_k <= 0:
        return []

    if len(evidences) <= top_k:
        return evidences

    candidates = []
    # 预处理 evidence, 生成给 LLM 的候选文本
    for ev in evidences:
        if not isinstance(ev, dict):
            continue

        source_id = ev.get("source_id")
        text = ev.get("text", "")[:max_text_len]
        scores = ev.get("scores", [])
        queries_hit = ev.get("queries_hit", [])

        if not source_id or not isinstance(text, str):
            continue

        text = text.strip()
        if not text:
            continue

        if not isinstance(scores, list):
            scores = []
        if not isinstance(queries_hit, list):
            queries_hit = []

        max_score = max(scores) if scores else 0
        hit_count = len(set(queries_hit))
        short_text = text[:max_text_len]

        candidates.append({
            "source_id": source_id,
            "text": short_text,
            "max_score": max_score,
            "hit_count": hit_count
        })

    if not candidates:
        return []

    if len(candidates) <= top_k:
        source_id_set = {c["source_id"] for c in candidates}
        return [
                   ev
                   for ev in evidences
                   if isinstance(ev, dict) and ev.get("source_id") in source_id_set
               ][:top_k]

    # 2.构造给 LLM 的输入文本
    candidates_lines = []
    for idx, c in enumerate(candidates, start=1):
        block = (
            f"[{idx}]\n"
            f"source_id: {c['source_id']}\n"
            f"max_score: {c['max_score']}\n"
            f"hit_count: {c['hit_count']}\n"
            f"text: {c['text']}\n"
        )
        candidates_lines.append(block)

    candidate_text = "\n".join(candidates_lines)

    system = (
        "你是一个检索证据重排器。"
        "任务：根据原问题，从候选证据中选出最相关、最能直接支持回答的证据。"
        "规则："
        "- 优先选择与原问题语义最相关、最能直接支持回答的证据"
        "- 可以参考 max_score 和 hit_count，但最终以语义相关性和回答支持度为准"
        "- 只返回最相关的 top_k 个 source_id"
        "- 只输出 JSON，不包含任何解释或额外的字段"
        '输出格式：{"top_source_ids":["s1","s2"]}'
    )

    user = (
        f"原问题：\n{query}\n\n"
        f"top_k：{top_k}\n\n"
        f"候选证据：\n{candidate_text}"
    )
    try:
        out = call_llm_json(system=system, user=user)
    except Exception:
        return evidences[:top_k]

    top_source_ids = out.get("top_source_ids", [])
    if not isinstance(top_source_ids, list):
        return evidences[:top_k]

    # 容错后处理: 过滤非法id+去重+补齐未返回的
    valid_source_ids = [c["source_id"] for c in candidates]
    valid_source_ids_set = set(valid_source_ids)

    reranked_ids: List[str] = []
    seen = set()

    # 先保留 LLM 返回的合法 source_id 顺序
    for sid in top_source_ids:
        if not isinstance(sid, str):
            continue
        sid = sid.strip()
        if not sid:
            continue
        if sid not in valid_source_ids_set:
            continue
        if sid in seen:
            continue

        reranked_ids.append(sid)
        seen.add(sid)

    for sid in valid_source_ids:
        if sid not in seen:
            reranked_ids.append(sid)
            seen.add(sid)

    # 再把未返回的候选原顺序补齐
    reranked_ids = reranked_ids[:top_k]

    # 按reranked_ids回填原始 evidence
    evidence_map: Dict[str, Dict[str, Any]] = {}
    for ev in evidences:
        if not isinstance(ev, dict):
            continue
        sid = ev.get("source_id")
        if not isinstance(sid,str) or not sid.strip():
            continue
        evidence_map[sid.strip()] = ev

    reranked_evidences = [
        evidence_map[sid]
        for sid in reranked_ids
        if sid in evidence_map
    ]

    return reranked_evidences


def merged_dict_to_evidence_List(
        items: List[Dict[str, Any]]
) -> List[Evidence]:
    results: List[Evidence] = []

    for i, item in enumerate(items, start=1):
        source_id = str(item.get("source_id", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        scores = item.get("scores", []) or []

        if not source_id or not text:
            continue

        score = max(scores) if scores else 0.0

        results.append(
            Evidence(
                id=f"M{i}",
                source=source_id,
                text=text,
                score=float(score),
                source_id=source_id,
                article_id="",
                title="",
                publish_at="",
                section_path=""
            )
        )
    return results


def multi_query_hybrid_retrieve(
        queries: List[str],
        top_k_query: int = 5,
        final_top_k: int = 8,
        project_keyword: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        exclude_source_ids: Optional[Set[str]] = None,
        article_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    对多个 query 分别做 hybrid retrieve
    再按 source_id 合并,最后排序并截断
    """
    if not queries:
        return []

    merged: Dict[str, Dict[str, Any]] = {}

    for query in queries:
        if not isinstance(query, str):
            continue

        query = query.strip()
        if not query:
            continue

        try:
            evidences = hybrid_retrieve(
                query=query,
                top_k=top_k_query,
                project_keyword=project_keyword,
                date_from=date_from,
                date_to=date_to,
                exclude_source_ids=exclude_source_ids,
                article_id=article_id
            )
        except Exception:
            continue

        if not evidences:
            continue

        for ev in evidences:
            if not isinstance(ev, Evidence):
                continue

            source_id = ev.source_id
            text = ev.text
            score = ev.score

            if source_id not in merged:
                merged[source_id] = {
                    "source_id": source_id,
                    "scores": [score],
                    "text": text,
                    "queries_hit": [query]
                }
            else:
                # 确保每一个 query 只命中一次
                # queries_hit去重是为了保证"命中次数"这个信号是真的,而不是重复计算
                if query not in merged[source_id]["queries_hit"]:
                    merged[source_id]["queries_hit"].append(query)
                merged[source_id]["scores"].append(score)

    merged_list = list(merged.values())

    merged_list.sort(
        key=lambda x: (
            max(x["scores"]) if x["scores"] else 0,
            len(x["queries_hit"])
        ),
        reverse=True
    )
    return merged_list[:final_top_k]


def to_bool(x: Any):
    if x is True:
        return True

    if isinstance(x, str):
        v = x.strip().lower()

        if v in ("true", "1"):
            return True

    if isinstance(x, (int, float)):
        return x != 0

    return False


def filter_expanded_queries_llm(
        current_query: str,
        queries: List[str]
) -> List[str]:
    """
    过滤扩展query
    只保留与原问题意图一致的 query
    """
    if not isinstance(current_query, str):
        return []

    current_query = current_query.strip()
    if not current_query:
        return []

    if not queries:
        return []

    system = (
        "你是一个检索query过滤器。"
        "任务："
        "判断候选query是否保持与原问题相同的问题意图。"
        "规则："
        "- 保留与原问题意图一致的query"
        "- 如果query改变了问题方向，则判为false"
        "- 如果query只是同义表达、关键词补充或更具体的检索表达，则判为true"
        "- 只输出JSON，不包含任何解释或额外字段"
        '输出格式：{"keep":[true,false]}'
    )

    user = json.dumps({
        "current_query": current_query,
        "queries": queries
    }, ensure_ascii=False)

    try:
        out = call_llm_json(system=system, user=user)
    except Exception:
        return []

    keep = out.get("keep", [])
    if not isinstance(keep, list):
        return []

    valid_queries = [
        q for q, k in zip(queries, keep) if to_bool(k)
    ]
    return valid_queries


def expand_queries_llm(
        question: str,
        max_queries: int = 2
) -> List[str]:
    """
    query expansion
    """
    system = (
        "你是一个检索查询扩展器。"
        f"根据 current_query，生成 max_queries 个语义一致、更适合文档检索的query。"
        "要求："
        "- 不改变原问题含义"
        "- 不改变原问题意图"
        "- 只做同义表达、关键词补充或更具体的检索表达"
        "- 不要发散到无关主题"
        "- 只输出 JSON"
        "输出格式："
        "{"
        '"queries":[]'
        "}"
    )
    user = json.dumps({
        "current_query": question,
        "max_queries": max_queries
    })
    out = call_llm_json(system=system, user=user)
    queries = out.get("queries", [])
    if not isinstance(queries, list):
        return []
    return [query.strip() for query in queries if query.strip()][:max_queries]


def deduplicate_queries(
        queries: List[str]
) -> List[str]:
    """
    按轻量归一化的结果去重,但保留原始 query 文本

    使用正则去除标点符号 并且小写
    """
    seen = set()
    results: List[str] = []

    for q in queries:
        if not isinstance(q, str):
            continue

        raw = q.strip()
        if not raw:
            continue

        norm = normalize_query_for_dedup(raw)
        if not norm:
            continue

        if norm not in seen:
            seen.add(norm)
            results.append(raw)

    return results


def normalize_query_for_dedup(
        query: str
) -> str:
    if not isinstance(query, str):
        return ""

    q = query.strip().lower()
    q = re.sub(r"[，。！？、；：‘’“”（）《》【】〈〉,.!?;:()\"'`\[\]{}<>/\\\-_=+@#$%^&*~|]+", "", q)
    return q


def build_multi_queries(
        current_query: str,
        max_expand_queries: int = 2
) -> List[str]:
    if not isinstance(current_query, str):
        return []

    current_query = current_query.strip()
    if not current_query:
        return []

    expand_queries = expand_queries_llm(
        question=current_query,
        max_queries=max_expand_queries
    )

    valid_queries = filter_expanded_queries_llm(
        current_query=current_query,
        queries=expand_queries
    )

    all_queries = [current_query] + valid_queries
    all_queries = deduplicate_queries(all_queries)

    return all_queries


def retrieve_with_expanded_queries(
        current_query: str,
        *,
        project_keyword: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        article_id: Optional[str] = None,
        max_expanded_queries: int = 2,
        top_k_per_query: int = 5,
        final_top_k: int = 8,
        rerank_top_k: int = 5
) -> List[Evidence]:
    """
    current_query
    -> expand query
    -> filter queries / dedup
    -> hybrid retrieve
    -> rerank
    -> 转成 Evidence 列表
    """
    all_queries = build_multi_queries(
        current_query=current_query,
        max_expand_queries=max_expanded_queries
    )

    if not all_queries:
        return []

    merged_items = multi_query_hybrid_retrieve(
        queries=all_queries,
        top_k_query=top_k_per_query,
        final_top_k=final_top_k,
        project_keyword=project_keyword,
        date_from=date_from,
        date_to=date_to,
        article_id=article_id,
    )

    if not merged_items:
        return []

    reranked_items = rerank_evidences_llm(
        query=current_query,
        evidences=merged_items,
        top_k=rerank_top_k
    )

    return merged_dict_to_evidence_List(reranked_items)


def hybrid_retrieve(
        query: str,
        top_k: int = CANDIDATE_K,
        project_keyword: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        exclude_source_ids: Set[str] = set(),
        article_id: Optional[str] = None,
) -> List[Evidence]:
    vector_results = retrieve_topk(
        query=query,
        top_k=top_k,
        project_keyword=project_keyword,
        date_from=date_from,
        date_to=date_to,
        exclude_source_ids=exclude_source_ids,
        article_id=article_id,
    )

    if len(vector_results) >= top_k:
        return vector_results[:top_k]

    keyword_results = keyword_retrieve(
        query=query,
        top_k=top_k,
        project_keyword=project_keyword,
        date_from=date_from,
        date_to=date_to,
        exclude_source_ids=exclude_source_ids,
        article_id=article_id,
    )

    return rrf_fuse(vector_results, keyword_results, top_k=top_k)


def keyword_retrieve(
        query: str,
        top_k: int = CANDIDATE_K,
        project_keyword: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        exclude_source_ids: Set[str] = set(),
        article_id: Optional[str] = None,
) -> List[Evidence]:
    base_filter = _build_filter(
        project_keyword=project_keyword,
        date_from=date_from,
        date_to=date_to,
        exclude_source_ids=exclude_source_ids,
        article_id=article_id,
    )

    query_filter = {
        "$and": [
            {"$text": {"$search": query}},
            base_filter
        ]
    }

    cursor = col.find(
        query_filter,
        {
            "_id": 1,
            "source_id": 1,
            "article_id": 1,
            "title": 1,
            "publish_at": 1,
            "section_path": 1,
            "text": 1,
            "score": {"$meta": "textScore"}
        }
    ).sort([("score", {"$meta": "textScore"})]).limit(top_k)

    docs = list(cursor)
    evs = []

    for i, d in enumerate(docs, start=1):
        source_id = str(d.get("source_id", d.get("article_id", d.get("_id"))))
        article_id_val = str(d.get("article_id", "") or "")
        title = str(d.get("title", "") or "")
        publish_at = str(d.get("publish_at", "") or "")
        section_path = str(d.get("section_path", "") or "")
        text = str(d.get("text", "") or "")

        source = "|".join(x for x in [publish_at, title, section_path])

        evs.append(
            Evidence(
                id=f"K{i}",
                source=source,
                text=text,
                source_id=source_id,
                article_id=article_id_val,
                title=title,
                publish_at=publish_at,
                section_path=section_path,
                score=float(d.get("score", 0.0))
            )
        )

    return evs


def rrf_fuse(
        vector_results: List[Evidence] = None,
        keyword_results: List[Evidence] = None,
        top_k: int = FINAL_N,
        rrf_k: int = 60
) -> List[Evidence]:
    """
    使用 RRF 融合 vector + keyword 检索结果

    1.遍历 vector results 按enumerate(...,start=1)算rank
    2.遍历 keyword results 按enumerate(...,start=1)算rank
    3.对每个source_id加上1/(rrf_rank+rank)
    4.最后按融合后的rank排序
    5.返回List[Evidence]
    """
    fusion: Dict[str, Dict[str, Any]] = {}

    # vector_results
    for rank, ev in enumerate(vector_results, start=1):
        sid = ev.source_id
        score = 1 / (rrf_k + rank)

        if sid not in fusion:
            fusion[sid] = {
                "rrf_score": score,
                "evidence": ev
            }
        else:
            fusion[sid]["rrf_score"] += score

            # 保留原始 score 更高的 evidence
            if ev.score > fusion[sid]["evidence"].score:
                fusion[sid]["evidence"] = ev

    # keyword results
    for rank, ev in enumerate(keyword_results, start=1):
        sid = ev.source_id
        score = 1 / (rrf_k + rank)

        if sid not in fusion:
            fusion[sid] = {
                "rrf_score": score,
                "evidence": ev
            }
        else:
            fusion[sid]["rrf_score"] += score

            if ev.score > fusion[sid]["evidence"].score:
                fusion[sid]["evidence"] = ev

    # 排序
    ranked = sorted(
        fusion.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    # 返回的结果
    results: List[Evidence] = []

    for i, item in enumerate(ranked[:top_k], start=1):
        ev = item["evidence"]
        score = item["rrf_score"]

        results.append(
            Evidence(
                id=f"H{i}",
                source=ev.source,
                text=ev.text,
                score=score,
                source_id=ev.source_id,
                article_id=ev.article_id,
                title=ev.title,
                publish_at=ev.publish_at,
                section_path=ev.section_path
            )
        )

    return results


def retrieve_topk(
        query: str,
        top_k: int = CANDIDATE_K,
        project_keyword: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        exclude_source_ids: Set[str] = set(),
        article_id: Optional[str] = None,
) -> List[Evidence]:
    """
    访问Mongo/$vectorSearch
    """
    qv = encode_query(query)
    base_filter = _build_filter(project_keyword, date_from, date_to, exclude_source_ids, article_id=article_id)

    # ---- A) $vectorSearch ----
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": VECTOR_PATH,
                    "queryVector": qv.tolist(),
                    "numCandidates": VECTOR_CANDIDATES,
                    "limit": top_k,
                    "filter": base_filter,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "source_id": 1,
                    "article_id": 1,
                    "title": 1,
                    "publish_at": 1,
                    "section_path": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        docs = list(col.aggregate(pipeline))
        evs: List[Evidence] = []
        for i, d in enumerate(docs, start=1):
            source_id = str(d.get("source_id", d.get("article_id", d.get("_id"))))
            article_id_val = str(d.get("article_id", "") or "")
            title = str(d.get("title", "") or "")
            publish_at = str(d.get("publish_at", "") or "")
            section_path = str(d.get("section_path", "") or "")
            text = str(d.get("text", "") or "")

            source = " | ".join([x for x in [publish_at, title, section_path] if x]).strip()
            evs.append(
                Evidence(
                    id=f"E{i}",
                    source=source,
                    text=text,
                    score=float(d.get("score", 0.0)),
                    source_id=source_id,
                    article_id=article_id_val,
                    title=title,
                    publish_at=publish_at,
                    section_path=section_path,
                )
            )
        return evs
    except Exception:
        pass

    # ---- B) Python rerank ----
    cursor = col.find(
        base_filter,
        {"embedding": 1, "source_id": 1, "article_id": 1, "title": 1, "publish_at": 1,
         "section_path": 1, "text": 1},
    ).limit(PY_RERANK_LIMIT)

    docs = list(cursor)
    if not docs:
        return []

    doc_vecs = np.array([d.get("embedding", [0.0] * MRL_DIM) for d in docs], dtype=np.float32)
    scores = doc_vecs @ qv

    idx = np.argsort(-scores)[:top_k]
    evs: List[Evidence] = []
    for rank, j in enumerate(idx, start=1):
        d = docs[int(j)]
        source_id = str(d.get("source_id", d.get("article_id", d.get("_id"))))
        article_id_val = str(d.get("article_id", "") or "")
        title = str(d.get("title", "") or "")
        publish_at = str(d.get("publish_at", "") or "")
        section_path = str(d.get("section_path", "") or "")
        text = str(d.get("text", "") or "")
        source = " | ".join([x for x in [publish_at, title, section_path] if x]).strip()

        evs.append(
            Evidence(
                id=f"E{rank}",
                source=source,
                text=text,
                score=float(scores[int(j)]),
                source_id=source_id,
                article_id=article_id_val,
                title=title,
                publish_at=publish_at,
                section_path=section_path,
            )
        )
    return evs


# ==================
# Planner Step
# ==================
TOOLS = {
    "hybrid_retrieve": hybrid_retrieve
}


def plan_tool_call_llm(question: str, top_k: int) -> Dict[str, Any]:
    """
    让模型输出 tool call JSON (最小版: hybrid_retrieve / no_tool)
    规则: 模型只负责 query/top_k: 过滤条件由代码注入 (更安全)
    """
    system = (
        "你是一个工具规划器(Tool Planner)。\n"
        "可用工具：\n"
        "1) hybrid_retrieve:从内部MongoDB数据库检索相关证据。\n"
        "任务: 根据用户问题判断是否需要检索。\n"
        "规则: \n"
        "- 若问题需要基于稿件证据回答(事实/要点/风险/结论/数据等)，选择 hybrid_retrieve。\n"
        "- 若不需要检索即可回答(寒暄/纯写作/纯格式化/与稿件无关等)，选择 no_tool。\n"
        "- 必须严格输出 JSON，不要任何额外文字。\n\n"
        "输出 schema: \n"
        "{\n"
        '  \"tool\": \"hybrid_retrieve\" | \"no_tool\",\n'
        '  \"args\": {\n'
        '    \"query\": \"string\",\n'
        '    \"top_k\": 0\n'
        "  }\n"
        "}\n"
        "约束: \n"
        f"- top_k 固定使用{top_k}(不要改成其他值)。\n"
    )
    user = json.dumps({"question": question}, ensure_ascii=False)

    out = call_llm_json(system=system, user=user, max_tokens=200, temperature=0.2)
    if not isinstance(out, dict):
        return {"tool": "hybrid_retrieve", "args": {"query": question, "top_k": top_k}}
    tool = out.get("tool")
    args = out.get("args") if isinstance(out.get("args"), dict) else {}
    if tool not in ["hybrid_retrieve", "no_tool"]:
        tool = "hybrid_retrieve"
    q = str(args.get("query", "") or "").strip() or question
    return {"tool": tool, "args": {"query": q, "top_k": top_k}}


def executor_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in TOOLS:
        return {"ok": False, "error": "tool_not_found", "result": None}

    try:
        tool_fn = TOOLS[tool_name]
        result = tool_fn(**args)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": f"tool_exception_{type(e).__name__}:{str(e)}"}


# =========================
# 7) LLM JSON 调用 + 解析兜底
# =========================
def _extract_text_from_response(resp) -> str:
    texts: List[str] = []
    for item in getattr(resp, "output", []) or []:
        item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
        if item_type != "message":
            continue
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None) or []
        for c in content:
            c_type = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else None)
            if c_type == "output_text":
                t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                if t:
                    texts.append(t)
    return "\n".join(texts).strip()


def call_llm_json(system: str, user: str, max_tokens: int = 800, temperature: float = 0.2) -> Dict[str, Any]:
    resp = ark.responses.create(
        model=JUDGE_MODEL,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    content = _extract_text_from_response(resp)

    try:
        return json.loads(content)
    except Exception:
        pass

    l = content.find("{")
    r = content.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return json.loads(content[l: r + 1])
        except Exception:
            pass

    return {"_raw": content[:600], "parse_error": True}


def diagnose_case(item: Dict[str, Any]) -> Dict[str, Any]:
    case = item.get("case", {}) or {}
    res = item.get("res", {}) or {}
    score = item.get("score", {}) or {}

    ans = res.get("answer_json", {}) or {}
    answer_text = str(ans.get("answer", "") or "").strip()
    citations = ans.get("citations", []) if isinstance(ans.get("citations"), list) else []
    evidence = res.get("evidence", []) or []
    trace = res.get("decision_trace", []) or []
    notes = res.get("notes", []) or []

    tags = []

    if not evidence:
        tags.append("no_evidence")
    if not answer_text:
        tags.append("no_answer")
    if answer_text and not citations:
        tags.append("citations_missing")
    if any("MAX_ROUNDS_REACHED" in str(t) for t in trace):
        tags.append("policy_max_rounds")
    if any("REWRITE_QUERY" in str(t) for t in trace):
        tags.append("has_rewrite")
    if any(str(n).startswith("critic: pass=False") for n in notes):
        tags.append("critic_failed")

    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "article_id": case.get("article_id"),
        "final_score": case.get("final_score"),
        "rounds": case.get("rounds"),
        "rewrites": case.get("rewrites"),
        "citation_ok": case.get("citation_ok"),
        "must_ok": case.get("must_ok"),
        "tags": tags
    }


# =========================
# 8) QA 输出 Schema（稿件问答）
# =========================
def _normalize_qa_schema(ans: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "answer": str(ans.get("answer", "") or "").strip(),
        "highlights": ans.get("highlights", []) if isinstance(ans.get("highlights"), list) else [],
        "citations": ans.get("citations", []) if isinstance(ans.get("citations"), list) else [],
        "unanswered": ans.get("unanswered", []) if isinstance(ans.get("unanswered"), list) else [],
        "confidence": float(ans.get("confidence", 0.5)) if str(ans.get("confidence", "")).strip() != "" else 0.5,
    }
    out["highlights"] = [str(x).strip() for x in out["highlights"] if str(x).strip()]
    out["unanswered"] = [str(x).strip() for x in out["unanswered"] if str(x).strip()]
    out["citations"] = [str(x).strip() for x in out["citations"] if str(x).strip()]
    out["confidence"] = max(0.0, min(1.0, float(out["confidence"])))
    return out


def _fix_qa_citations(ans: Dict[str, Any], valid_ids: Set[str]) -> Dict[str, Any]:
    cits = ans.get("citations", [])
    if not isinstance(cits, list):
        cits = []
    ans["citations"] = [c for c in cits if c in valid_ids]
    return ans


def _qa_citation_coverage(ans: Dict[str, Any]) -> float:
    if ans.get("answer", "").strip() and isinstance(ans.get("citations"), list) and len(ans["citations"]) > 0:
        return 1.0
    return 0.0


def generate_qa_answer_llm(question: str, evidence: List[Evidence]) -> Dict[str, Any]:
    ev_pack = [{"id": e.id, "source": e.source, "text": e.text[:900]} for e in evidence]

    system = (
        "你是稿件问答助手（RAG）。"
        "必须仅基于 evidence 回答问题，严禁编造。"
        "如果 evidence 不足以回答，请在 unanswered 中写清楚缺少的信息点。"
        "必须严格输出 JSON，不要任何额外文字。\n"
        "schema:\n"
        "{\n"
        '  \"answer\": \"string\",\n'
        '  \"highlights\": [\"string\"],\n'
        '  \"citations\": [\"E1\"],\n'
        '  \"unanswered\": [\"string\"],\n'
        '  \"confidence\": 0.0\n'
        "}\n"
        "规则：\n"
        "- answer 简洁准确。\n"
        "- citations 只能引用 evidence 的 id（E1/E2...）。\n"
        "- 没证据支持的内容不要写进 answer。\n"
    )

    user = json.dumps({"question": question, "evidence": ev_pack}, ensure_ascii=False)
    out = call_llm_json(system=system, user=user, max_tokens=900, temperature=0.2)
    return _normalize_qa_schema(out if isinstance(out, dict) else {})


def render_qa_markdown(question: str, ans: Dict[str, Any]) -> str:
    cits = ans.get("citations", [])
    md = []
    md.append(f"## Q: {question}\n")
    md.append(f"**A:** {ans.get('answer', '') or '（暂无答案）'}")
    if cits:
        md.append(f"\n**Citations:** {', '.join(cits)}")
    hl = ans.get("highlights", [])
    if hl:
        md.append("\n### 要点")
        for x in hl:
            md.append(f"- {x}")
    ua = ans.get("unanswered", [])
    if ua:
        md.append("\n### 未回答/缺证据点")
        for x in ua:
            md.append(f"- {x}")
    md.append(f"\n**可信度：{int(float(ans.get('confidence', 0.5)) * 100)}%**")
    return "\n".join(md)


# =========================
# 9) Lesson10 Policy（内置版）
# =========================
DECISIONS = ("SUFFICIENT", "RETRIEVE_MORE", "REWRITE_QUERY")


@dataclass
class PolicyDecision:
    decision: str
    rationale: str = ""
    rewrite_query: Optional[str] = None


class RetrievalPolicy:
    def __init__(self, ark_client: Ark, model_name: str, max_evidence_chars: int = 3000):
        self.ark = ark_client
        self.model_name = model_name
        self.max_evidence_chars = max_evidence_chars

    def _trim(self, s: str) -> str:
        if not s:
            return ""
        if len(s) <= self.max_evidence_chars:
            return s
        head = s[: int(self.max_evidence_chars * 0.65)]
        tail = s[-int(self.max_evidence_chars * 0.30):]
        return head + "\n...\n" + tail

    def decide(self, user_question: str, current_query: str, evidence_text: str) -> PolicyDecision:
        """
        LLM 做决策
        """
        evidence_text = self._trim(evidence_text)

        # "- 若缺少关键要素（时间、主体、范围、结论、影响）→ 不足\n"
        # 太过于严格 使用宽松规则
        # "- 若 evidence 已包含问答问题的核心要点（即使细节不完整），可判定为 SUFFICIENT\n"
        # "- 不需要覆盖所有细节，只要能够回答主要问题即可\n"
        system = (
            "你是一个 RAG 检索控制器（Retrieval Controller）。\n"
            "你的任务是判断：当前检索到的 evidence 是否足以回答用户问题。\n\n"
            "如果不足，请判断：\n"
            "1）是否继续使用当前 query 扩大检索（RETRIEVE_MORE）\n"
            "2）是否需要改写 query 再检索（REWRITE_QUERY）\n\n"
            "严格输出格式：\n"
            "DECISION: <SUFFICIENT|RETRIEVE_MORE|REWRITE_QUERY>\n"
            "RATIONALE: <一句简短原因>\n"
            "REWRITE_QUERY: <仅当 decision 为 REWRITE_QUERY 时给出>\n\n"
            "判断规则：\n"
            # "- 若缺少关键要素（时间、主体、范围、结论、影响）→ 不足\n"
            "- 若 evidence 已包含问答问题的核心要点（即使细节不完整），可判定为 SUFFICIENT\n"
            "- 不需要覆盖所有细节，只要能够回答主要问题即可\n"
            "- 若 evidence 部分相关但信息不完整 → RETRIEVE_MORE\n"
            "- 若 query 与 evidence 明显不匹配或过于宽泛 → REWRITE_QUERY\n"
            "- 若 evidence 已足以支持回答 → SUFFICIENT\n"
        )

        user = f"""用户问题:
{user_question}

当前查询:
{current_query}

已检索证据(摘要):
{evidence_text}
"""
        try:
            resp = self.ark.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=260,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            return PolicyDecision("RETRIEVE_MORE", f"policy_exception_{type(e).__name__}", None)

        # parse
        decision = None
        rationale = ""
        rewrite_query = None

        m = re.search(r"DECISION:\s*([A-Z_]+)", text)
        if m:
            decision = m.group(1).strip()
        m = re.search(r"RATIONALE:\s*(.*)", text)
        if m:
            rationale = m.group(1).strip()
        m = re.search(r"REWRITE_QUERY:\s*(.*)", text)
        if m:
            rewrite_query = m.group(1).strip()

        if decision not in DECISIONS:
            decision = "RETRIEVE_MORE"
            rationale = rationale or "unclear_policy_output_default_retrieve"
            rewrite_query = None

        if decision != "REWRITE_QUERY":
            rewrite_query = None

        return PolicyDecision(decision=decision, rationale=rationale[:220], rewrite_query=rewrite_query)


policy = RetrievalPolicy(ark_client=ark, model_name=POLICY_MODEL)


def format_evidence_for_policy(evidence: List[Evidence], max_chunks: int = 6) -> str:
    lines = []
    for i, e in enumerate(evidence[:max_chunks], 1):
        txt = (e.text or "").replace("\n", " ").strip()
        lines.append(f"[{i}] source_id={e.source_id} source={e.source} text={txt[:320]}")
    return "\n".join(lines)


def rewrite_query_llm_by_evidence(question: str, current_query: str, evidence: List[Evidence], turn: int) -> str:
    """
    使用 LLM 改写 query
    """
    system = (
        "你是检索查询改写器。根据 question 与已检索 evidence，生成更具体、更可检索的一行 query。"
        "只输出 query，不要解释。"
        "要求：包含可检索关键词（主体/事件/时间/地点/指标/结论/争议点等），避免空泛词。"
    )
    ev_brief = [{"source": e.source, "snippet": e.text[:180]} for e in evidence[:5]]
    user = json.dumps(
        {"question": question, "current_query": current_query, "turn": turn, "evidence": ev_brief},
        ensure_ascii=False
    )
    resp = ark.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()


# =========================
# 10) Lesson11 Critic（QA 版）+ 修复策略
# =========================
def critic_qa_llm(question: str, evidence: List[Evidence], answer_json: Dict[str, Any]) -> Dict[str, Any]:
    ev_pack = [{"id": e.id, "source": e.source, "text": e.text[:450]} for e in evidence]
    system = (
        "你是稿件问答系统的审稿人(Critic)。只做审查，不要重写答案。"
        "必须严格输出 JSON，不要任何额外文本。\n"
        "schema:\n"
        "{\n"
        '  \"pass\": boolean,\n'
        '  \"issues\": string[],\n'
        '  \"suggested_fix\": \"REANSWER\" | \"RETRIEVE_MORE\" | \"EVIDENCE_ONLY\",\n'
        '  \"confidence_adjust\": float\n'
        "}\n"
        "审查规则：\n"
        "- answer 若明显超出 evidence 内容范围（新增事实/时间/主体/结论），判 fail。\n"
        "- answer 非空但 citations 为空，倾向 fail。\n"
        "- citations 引用不存在的证据 id，判 fail。\n"
        "- confidence_adjust 建议在 [-0.5, 0.0]，fail 时更低。\n"
    )
    user = json.dumps({"question": question, "evidence": ev_pack, "answer": answer_json}, ensure_ascii=False)
    out = call_llm_json(system=system, user=user, max_tokens=450, temperature=0.2)

    if not isinstance(out, dict):
        return {"pass": False, "issues": ["critic_output_not_json"], "suggested_fix": "EVIDENCE_ONLY",
                "confidence_adjust": -0.3}

    if out.get("suggested_fix") not in ["REANSWER", "RETRIEVE_MORE", "EVIDENCE_ONLY"]:
        out["suggested_fix"] = "EVIDENCE_ONLY" if not bool(out.get("pass")) else "REANSWER"
    if not isinstance(out.get("issues"), list):
        out["issues"] = ["critic_issues_not_list"]
    try:
        out["confidence_adjust"] = float(out.get("confidence_adjust", -0.1))
    except Exception:
        out["confidence_adjust"] = -0.1
    return out


def evidence_only_qa_fallback(question: str, evidence: List[Evidence]) -> Dict[str, Any]:
    if not evidence:
        return _normalize_qa_schema({
            "answer": "",
            "highlights": [],
            "citations": [],
            "unanswered": ["未检索到相关稿件证据，无法回答。"],
            "confidence": 0.2
        })

    hl = []
    for e in evidence[: min(3, len(evidence))]:
        hl.append(e.text[:160])

    return _normalize_qa_schema({
        "answer": "根据已检索到的稿件片段，目前只能确认以下要点（其余细节缺少证据支撑）。",
        "highlights": hl,
        "citations": [evidence[0].id],
        "unanswered": ["需要更多与问题直接相关的稿件段落/上下文来完整回答。"],
        "confidence": 0.35
    })


def apply_critic_and_fix_qa(
        question: str,
        state: AgentState,
        project_keyword: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        final_evidence: List[Evidence],
        answer_json: Dict[str, Any],
        article_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Evidence]]:
    critic = critic_qa_llm(question, final_evidence, answer_json)
    state.notes.append(f"critic: pass={critic.get('pass')} fix={critic.get('suggested_fix')}")
    if critic.get("issues"):
        state.notes.append(f"critic: issues={critic.get('issues')[:6]}")

    # adjust confidence
    adj = float(critic.get("confidence_adjust", 0.0))
    answer_json["confidence"] = max(0.0, min(1.0, float(answer_json.get("confidence", 0.5)) + adj))

    if bool(critic.get("pass")):
        return answer_json, final_evidence

    fix = critic.get("suggested_fix")
    if fix == "REANSWER":
        state.notes.append("critic_fix: reanswer")
        answer_json = generate_qa_answer_llm(question, final_evidence)
        answer_json = _normalize_qa_schema(answer_json)
        return answer_json, final_evidence

    if fix == "RETRIEVE_MORE":
        state.notes.append("critic_fix: retrieve_more_then_answer")
        more = retrieve_topk(
            query=question,
            top_k=min(CANDIDATE_K + 6, TOPK_CAP),
            project_keyword=project_keyword,
            date_from=date_from,
            date_to=date_to,
            exclude_source_ids=state.seen_source_ids,
            article_id=article_id,
        )
        if more:
            merged = final_evidence + [e for e in more if e.source_id not in {x.source_id for x in final_evidence}]
            final_evidence = merged[:FINAL_N]
        answer_json = generate_qa_answer_llm(question, final_evidence)
        answer_json = _normalize_qa_schema(answer_json)
        return answer_json, final_evidence

    state.notes.append("critic_fix: evidence_only")
    return evidence_only_qa_fallback(question, final_evidence), final_evidence


class ToolExecutor:

    def __init__(self, tools):
        self.tools = tools

    def execute(self, tool_name, args):
        if tool_name not in self.tools:
            return {"ok": False, "error": "tool_not_found"}

        tool_fn = self.tools[tool_name]
        result = tool_fn(**args)
        return {"ok": True, "result": result}


# ==================
# Observation handle
# ==================
def handle_no_evidence(
        question: str,
        current_query: str,
        state: AgentState,
        round_id: int,
) -> Dict[str, Any]:
    """
    空证据处理
    - rewrite次数太多 -> FALLBACK
    - REWRITE
    {
        "action":"REWRITE" | "FALLBACK" | "CONTINUE", # 告诉主循环下一步干什么
        "query":"...", # 新的 query(如果rewrite)
        "response":None # 如果需要直接返回答案
    }
    """
    if state.rewrite_times >= 2:
        state.decision_trace.append(
            f"Round{round_id}: NO_EVIDENCE_AFTER_REWRITE_LIMIT -> EVIDENCE_ONLY"
        )
        ans = evidence_only_qa_fallback(question, [])
        return {
            "action": "FALLBACK",
            "query": current_query,
            "response": {
                "question": question,
                "mode": "policy",
                "turns": state.retrieve_rounds,
                "notes": state.notes + ["fallback: rewrite_limit_no_evidence"],
                "decision_trace": state.decision_trace,
                "evidence": [],
                "answer_json": ans,
                "answer_markdown": render_qa_markdown(question, ans),
            },
        }

    state.rewrite_times += 1
    state.decision_trace.append(
        f"Round{round_id}: NO_EVIDENCE -> FORCE_REWRITE | top_k={state.top_k} | q='{current_query[:60]}'"
    )

    try:
        new_q = rewrite_query_llm_by_evidence(question, current_query, [], round_id).strip()
        if new_q and len(new_q) >= 3:
            state.decision_trace.append(f"Round{round_id}: rewrite_source=no_evidence_llm")
            return {"action": "REWRITE", "query": new_q, "response": None}

        state.decision_trace.append(f"Round{round_id}: rewrite_source=no_evidence_fallback_question")
        return {"action": "REWRITE", "query": question, "response": None}

    except Exception as e:
        state.decision_trace.append(
            f"Round{round_id}: rewrite_source=no_evidence_exception({type(e).__name__})"
        )
        return {"action": "REWRITE", "query": question, "response": None}


# ==================
# Planner Step
# ==================
def plan_tool_step(
        current_query: str,
        state: AgentState,
) -> Dict[str, Any]:
    """
    Planner 只负责:
    - 是否调用工具
    - 调用哪个工具
    - query / top_k
    """
    return plan_tool_call_llm(question=current_query, top_k=state.top_k)


def handle_observation(
        question: str,
        current_query: str,
        evidences: List[Evidence],
        state: AgentState,
        round_id: int
) -> Dict[str, Any]:
    """
    Observation 处理层:
    - no evidence
    - policy decide
    - sufficient | rewrite_query | retrieve_more
    """
    if not evidences:
        return handle_no_evidence(question, current_query, state, round_id)

    evidence_text = format_evidence_for_policy(evidences)
    dec = policy.decide(
        user_question=question,
        current_query=current_query,
        evidence_text=evidence_text
    )

    state.decision_trace.append(
        f"Round{round_id}: {dec.decision} | top_k={state.top_k} | q='{current_query[:60]}' | why='{dec.rationale}'"
    )

    if dec.decision == "SUFFICIENT":
        return {
            "action": "SUFFICIENT",
            "query": current_query,
            "evidences": evidences,
            "response": None
        }

    if dec.decision == "REWRITE_QUERY":
        state.rewrite_times += 1

        # policy 直接给了 rewrite_query 就用；否则 evidence 驱动改写
        if dec.rewrite_query and len(dec.rewrite_query.strip()) >= 3:
            state.decision_trace.append(f"Round{round_id}: rewrite_source=policy")
            return {
                "action": "REWRITE",
                "query": dec.rewrite_query.strip(),
                "evidences": evidences,
                "response": None
            }

        try:
            new_q = rewrite_query_llm_by_evidence(question, current_query, evidences, round_id).strip()
            if new_q and len(new_q) >= 3:
                state.decision_trace.append(f"Round{round_id}: rewrite_source=evidence_llm")
                return {
                    "action": "REWRITE",
                    "query": new_q,
                    "evidences": evidences,
                    "response": None
                }
            else:
                state.decision_trace.append(f"Round{round_id}: rewrite_source=fallback_question")
                return {
                    "action": "REWRITE",
                    "query": question,
                    "evidences": evidences,
                    "response": None
                }
        except Exception as e:
            state.decision_trace.append(f"Round{round_id}: rewrite_source=exception_fallback({type(e).__name__})")
            return {
                "action": "REWRITE",
                "query": question,
                "evidences": evidences,
                "response": None
            }
    return {
        "action": "RETRIEVE_MORE",
        "query": current_query,
        "evidences": evidences,
        "response": None
    }


# def compress_evidence(
#         question: str,
#         evidences: List[Evidence],
#         pre_evidence_max_chars: int = 100
# ) -> List[Evidence]:
#     """
#     把证据压缩成更短的关键事实版本
#     返回新的 Evidence 列表(保留 id/source,替换 text)
#     :param question:
#     :param evidences:
#     :param pre_evidence_max_chars:
#     :return:
#     """
#     if not evidences:
#         return []
#
#     compressed: List[Evidence] = []
#
#     system = (
#         "你是一个证据压缩器 (Evidence Compression)。\n"
#         "你的任务是根据问题，从 evidence 中提炼出最关键、最有助于回答问题的事实。\n"
#         "要求：\n"
#         "- 只保留与回答问题直接相关的信息\n"
#         "- 不要补充原文没有的信息\n"
#         "- 尽量压缩，保留事实、时间、数字、主体、结论\n"
#         "- 如果 evidence 与问题关系很弱，可以输出原文的一句短摘录\n"
#         "- 只输出 JSON，不要额外文字\n"
#         '输出格式：{"compressed_text":"..."}\n'
#     )
#
#     for e in evidences:
#         user = json.dumps(
#             {
#                 "question": question,
#                 "evidence": {
#                     "id": e.id,
#                     "text": e.text[:1200]
#                 }
#             }
#         )
#
#         out = call_llm_json(
#             system=system,
#             user=user,
#             max_tokens=200,
#             temperature=0.0,
#         )
#
#         compressed_text = ""
#         if isinstance(out, dict):
#             compressed_text = str(out.get("compressed_text", "")).strip()
#
#         if not compressed_text:
#             compressed_text = e.text[:pre_evidence_max_chars]
#
#         compressed_text = compressed_text[:pre_evidence_max_chars]
#
#         compressed.append(
#             Evidence(
#                 id=e.id,
#                 source=e.source,
#                 text=compressed_text,
#                 score=e.score,
#                 source_id=e.source_id,
#                 article_id=e.article_id,
#                 title=getattr(e, "title", None),
#                 publish_at=e.publish_at,
#                 section_path=e.section_path
#             )
#         )
#     return compressed

def llm_rerank_evidence(
        question: str,
        evidences: List[Evidence],
        final_n: int = FINAL_N
) -> List[Evidence]:
    """
    排序器
    把最相关的顶上来
    :return: 排序后的FINAL_N条数据
    """

    if not evidences:
        return []

    RERANK_MAX_INPUT = 15

    evidences = evidences[:RERANK_MAX_INPUT]

    # 证据压缩(evidence compression)
    ev_pack = [
        {"id": e.id, "text": e.text[:400]}
        for e in evidences
    ]

    system = (
        "你是一个检索结果排序器（Reranker）。\n"
        "根据问题，判断哪些 evidence 最有助于回答问题。\n"
        "优先选择包含答案关键事实的 evidence。\n"
        "只返回按相关性从高到低排序后的 evidence id 列表。\n"
        "必须输出 JSON，不要任何额外文字。\n"
        "输出格式：\n"
        "{\n"
        '  "ranked_ids": ["H3", "H1", "H2"]\n'
        "}\n"
        "规则：\n"
        "- ranked_ids 必须包含输入 evidence 的所有 id\n"
        "- 不要编造新的 id\n"
        "- 按最有助于回答问题的顺序排序\n"
    )

    user = json.dumps({
        "question": question,
        "evidence": ev_pack
    }, ensure_ascii=False)

    out = call_llm_json(system=system, user=user, max_tokens=300, temperature=0.0)

    ranked_ids = out.get("ranked_ids", []) if isinstance(out, dict) else []

    if not isinstance(ranked_ids, list):
        ranked_ids = []

    if not ranked_ids:
        return evidences[:final_n]

    id_map = {e.id: e for e in evidences}
    ranked: List[Evidence] = []
    for rid in ranked_ids:
        rid = str(rid).strip()
        if rid in id_map and id_map[rid] not in ranked:
            ranked.append(id_map[rid])

    # 兜底
    for e in evidences:
        if e not in ranked:
            ranked.append(e)

    return ranked[:final_n]


def finalize_answer(
        question: str,
        final_evidence: List[Evidence],
        state: AgentState,
        *,
        project_keyword: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        article_id: Optional[str],
        enable_critic: bool = True,
) -> Dict[str, Any]:
    if not final_evidence:
        ans = _normalize_qa_schema({
            "answer": "",
            "highlights": [],
            "citations": [],
            "unanswered": ["未检索到相关稿件证据，无法回答。"],
            "confidence": 0.2,
        })
        return {
            "question": question,
            "mode": "policy",
            "turns": state.retrieve_rounds,
            "notes": state.notes + ["fallback: no_evidence"],
            "decision_trace": state.decision_trace,
            "evidence": [],
            "answer_json": ans,
            "answer_markdown": render_qa_markdown(question, ans),
        }

    final = final_evidence[:FINAL_N]
    ans = generate_qa_answer_llm(question=question, evidence=final)
    valid_ids = {e.id for e in final}
    ans = _fix_qa_citations(ans, valid_ids)

    if _qa_citation_coverage(ans) < 1.0 and final:
        state.notes.append("fallback: low_citation_coverage -> evidence_only")
        ans = evidence_only_qa_fallback(question, final)
        ans = _fix_qa_citations(ans, valid_ids)

    if enable_critic:
        ans, final2 = apply_critic_and_fix_qa(
            question=question,
            state=state,
            project_keyword=project_keyword,
            date_from=date_from,
            date_to=date_to,
            final_evidence=final,
            answer_json=ans,
            article_id=article_id,
        )
        valid_ids2 = {e.id for e in final2}
        ans = _fix_qa_citations(ans, valid_ids2)
        final = final2

    return {
        "question": question,
        "mode": "policy",
        "turns": state.retrieve_rounds,
        "notes": state.notes,
        "decision_trace": state.decision_trace,
        "evidence": [
            {
                "id": e.id,
                "source": e.source,
                "score": e.score,
                "text": e.text,
                "source_id": e.source_id,
                "article_id": e.article_id,
            }
            for e in final
        ],
        "answer_json": ans,
        "answer_markdown": render_qa_markdown(question, ans),
    }


# ==================
# Execute Step
# ==================
def execute_tool_step(
        executor: ToolExecutor,
        tool_call: Dict[str, Any],
        *,
        project_keyword: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        exclude_source_ids: Set[str],
        article_id: Optional[str],
        state: AgentState,
) -> List[Evidence]:
    """
    Executor 只负责
    - 进入系统参数
    - 调用工具
    - 返回 evidences
    """
    if tool_call["tool"] == "no_tool":
        return []

    args = dict(tool_call["args"])
    args.update({
        "project_keyword": args.get("project_keyword", project_keyword),
        "date_from": args.get("date_from", date_from),
        "date_to": args.get("date_to", date_to),
        "exclude_source_ids": args.get("exclude_source_ids", []),
        "article_id": args.get("article_id", article_id),
    })
    exe = executor_tool(tool_call["tool"], args)
    if not exe["ok"]:
        state.notes.append(f"tool_error:{exe['error']}")
        return []
    else:
        return exe["result"] or []


# =========================
# 11) Policy 主循环（问答智能体核心）
# =========================
def qa_agent_with_policy(
        question: str,
        *,
        project_keyword: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        article_id: Optional[str] = None,
        max_rounds: int = MAX_POLICY_ROUNDS,
        enable_critic: bool = True,
) -> Dict[str, Any]:
    state = AgentState(top_k=TOPK_INIT)
    current_query = question
    exclude_source_ids: Set[str] = set()
    last_evidence: List[Evidence] = []

    for r in range(1, max_rounds + 1):
        state.retrieve_rounds += 1

        # 1) planner
        tool_call = plan_tool_step(current_query, state)

        if tool_call["tool"] == "no_tool":
            evidences = []
        else:
            planned_query = str(tool_call["args"].get("query", "") or current_query).strip() or current_query

            evidences = retrieve_with_expanded_queries(
                current_query=planned_query,
                project_keyword=project_keyword,
                date_from=date_from,
                date_to=date_to,
                max_expanded_queries=2,
                top_k_per_query=state.top_k,
                final_top_k=max(state.top_k, FINAL_N),
                rerank_top_k=FINAL_N,
                article_id=article_id,
            )

        last_evidence = evidences

        # 更新去重状态
        for ev in evidences:
            exclude_source_ids.add(ev.source_id)
            state.seen_source_ids.add(ev.source_id)

        # 3) Observation
        obs = handle_observation(
            question=question,
            current_query=current_query,
            evidences=evidences,
            state=state,
            round_id=r,
        )
        if obs["action"] == "FALLBACK":
            return obs["response"]

        if obs["action"] == "REWRITE":
            current_query = obs["query"]
            continue

        if obs["action"] == "RETRIEVE_MORE":
            state.top_k = min(state.top_k + 3, TOPK_CAP)
            continue

        if obs["action"] == "SUFFICIENT":
            return finalize_answer(
                question=question,
                final_evidence=obs["evidences"],
                state=state,
                project_keyword=project_keyword,
                date_from=date_from,
                date_to=date_to,
                article_id=article_id,
                enable_critic=enable_critic,
            )

    state.decision_trace.append("Final: MAX_ROUNDS_REACHED -> answer with last evidence.")
    return finalize_answer(
        question=question,
        final_evidence=last_evidence,
        state=state,
        project_keyword=project_keyword,
        date_from=date_from,
        date_to=date_to,
        article_id=article_id,
        enable_critic=enable_critic,
    )


# =========================
# 12) Demo 数据：写入几条稿件 chunks
# =========================
def seed_demo_data(project: str = "财经"):
    """
    写入 demo 稿件 chunks（用于你快速验证链路）
    """
    ensure_basic_indexes()
    chunks = [
        {
            "_id": "A20260215_005#c1",
            "article_id": "A20260215_005",
            "title": "AI 基础设施投资风险升温",
            "publish_at": "2026-02-15",
            "section_path": project,
            "visibility": DEFAULT_VISIBILITY,
            "source_id": "A20260215_005",
            "text": "稿件指出，随着算力中心建设加快，行业开始重新评估投资节奏与项目回报，但正文没有展开具体风险细项。",
        },
        {
            "_id": "A20260216_006#c1",
            "article_id": "A20260216_006",
            "title": "数据中心扩建进入新阶段",
            "publish_at": "2026-02-16",
            "section_path": project,
            "visibility": DEFAULT_VISIBILITY,
            "source_id": "A20260216_006",
            "text": "文中提到，AI 数据中心建设面临电力容量约束、GPU 供应链波动，以及资本开支回收周期拉长等风险。",
        },
        {
            "_id": "A20260218_007#c1",
            "article_id": "A20260218_007",
            "title": "消费电子市场回暖",
            "publish_at": "2026-02-18",
            "section_path": project,
            "visibility": DEFAULT_VISIBILITY,
            "source_id": "A20260218_007",
            "text": "报告显示，消费电子需求回升带动产业链出货改善，但与 AI 基础设施投资关系不大。",
        }
    ]
    upsert_chunks(chunks)


# --------------------
# 运行测试集 path evalset(JSONL)
# --------------------
def load_evalset(jsonl_path: str) -> List[Dict[str, Any]]:
    cases = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line.strip()))
    return cases


# --------------------
# 运行单条测试数据
# --------------------
def run_one_case(
        case: Dict[str, Any],
        *,
        enable_critic: bool = True
) -> Dict[str, Any]:
    q = case.get("question", "")
    article_id = case.get("article_id")
    project = case.get("project_keyword")
    date_from = case.get("date_from")
    date_to = case.get("date_to")

    res = qa_agent_with_policy(
        question=q,
        project_keyword=project,
        date_from=date_from,
        date_to=date_to,
        article_id=article_id,
        enable_critic=enable_critic,
    )
    res["case_id"] = case.get("id")
    res["gold_keywords"] = case.get("gold_keywords", [])
    res["must_answer"] = case.get("must_answer", True)
    return res


# =========================
# 评分 (弱监督: 关键命中 + 引用覆盖 + must_answer)
# =========================
def _contains_keyword(
        text: str,
        kw: str
) -> bool:
    if not text or not kw:
        return False
    return kw in text


def _to_bool(v, default=True):
    """
    获取bool数据 如果没有 则使用默认值
    """
    if v is None:
        return default
    elif isinstance(v, bool):
        return v
    elif isinstance(v, str):
        return bool(v)
    elif isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ["true", "1", "yes", "y"]:
        return True
    if s in ["false", "0", "no", "n"]:
        return False
    return default


def score_one_case(
        res: Dict[str, Any],
        case: Dict[str, Any]
) -> Dict[str, Any]:
    ans = (res.get("answer_json") or {})
    answer_text = str(ans.get("answer", "") or "")
    citations = ans.get("citations", []) if isinstance(ans.get("citations"), list) else []
    decision_trace = res.get("decision_trace", []) or []
    notes = res.get("notes", []) or []

    gold_keywords = case.get("gold_keywords", []) or []
    must_answer = _to_bool(case.get("must_answer", True), default=True)

    # 1) 关键词命中率
    hit = 0
    for kw in gold_keywords:
        if _contains_keyword(answer_text, kw):
            hit += 1
    kw_score = (hit / len(gold_keywords)) if gold_keywords else 0.0

    # 2) citation coverage 引用是否存在 (answer 非空 且 citations 非空 -> 1，否则 0)
    # 系统有没有做到“有证据说话”。
    citation_ok = 1.0 if (answer_text.strip() and len(citations) > 0) else 0.0

    # 3) must_answer(必须回答的 case 是否真的回答了)
    # 系统有没有“该答不答”的情况（检索失败/过度拒答）。
    must_ok = 1.0 if (not must_answer or answer_text.strip()) else 0.0

    # 4) basic stats
    rounds = int(res.get("turns", 0) or 0)  # 跑了几轮检索
    rewrites = 0  # 改写次数
    for t in decision_trace:
        if "REWRITE_QUERY" in str(t) and "FORCE_REWRITE" not in str(t):
            rewrites += 1

    # 5) critic pass signal (因为 notes 写了 "critic: pass=")
    critic_pass = None
    for n in notes:
        if str(n).startswith("critic: pass="):
            critic_pass = bool("pass=True" in str(n))

    # 综合得分
    final = 0.45 * kw_score + 0.35 * citation_ok + 0.2 * must_ok

    return {
        "kw_score": round(kw_score, 3),
        "citation_ok": float(citation_ok),
        "must_ok": float(must_ok),
        "final_score": round(final, 3),
        "rewrites": rewrites,
        "rounds": rounds,
        "critic_pass": critic_pass,
    }


def run_eval(
        jsonl_path: str,
        *,
        enable_critic: bool = True,
        limit: Optional[int] = None
) -> Dict[str, Any]:
    cases = load_evalset(jsonl_path)
    if limit:
        cases = cases[:limit]

    results = []
    for c in cases:
        res = run_one_case(c, enable_critic=enable_critic)
        sc = score_one_case(res, c)
        results.append({"case": c, "res": res, "score": sc})

    # 平均数 聚合
    n = len(results)
    avg_final = sum(x["score"]["final_score"] for x in results) / n if n else 0.0
    avg_rounds = sum(x["score"]["rounds"] for x in results) / n if n else 0.0
    avg_rewrites = sum(x["score"]["rewrites"] for x in results) / n if n else 0.0
    citation_rate = sum(x["score"]["citation_ok"] for x in results) / n if n else 0.0
    must_rate = sum(x["score"]["must_ok"] for x in results) / n if n else 0.0

    # policy 决策分布 (从 decision_trace 每轮的 "RoundX:<DECISION>" 提取)
    dec_cnt = {"SUFFICIENT": 0, "RETRIEVE_MORE": 0, "REWRITE_QUERY": 0}
    for x in results:
        trace = x["res"].get("decision_trace", []) or []
        for t in trace:
            s = str(t)
            if " SUFFICIENT " in s or ": SUFFICIENT" in s:
                dec_cnt["SUFFICIENT"] += 1
            elif " RETRIEVE_MORE " in s or ": RETRIEVE_MORE" in s:
                dec_cnt["RETRIEVE_MORE"] += 1
            elif " REWRITE_QUERY " in s or ": REWRITE_QUERY" in s:
                dec_cnt["REWRITE_QUERY"] += 1

    return {
        "n": n,
        "avg_final": round(avg_final, 3),
        "avg_rounds": round(avg_rounds, 3),
        "avg_rewrites": round(avg_rewrites, 3),
        "citation_rate": round(citation_rate, 3),
        "must_rate": round(must_rate, 3),
        "dec_count": dec_cnt,
        "results": results,  # 详细结果（想轻量可后面再裁剪）
    }


def print_eval_report(
        report: Dict[str, Any]
):
    print("\n==== EVAL REPORT ====")
    print("N:", report.get("n", 0))
    print("avg_final:", report.get("avg_final", 0.0))
    print("avg_rounds:", report.get("avg_rounds", 0.0))
    print("avg_rewrites:", report.get("avg_rewrites", 0.0))
    print("citation_rate:", report.get("citation_rate", 0.0))
    print("must_answer_rate:", report.get("must_rate", 0.0))
    print("decision_counts:", report.get("dec_count", {"SUFFICIENT": 0, "RETRIEVE_MORE": 0, "REWRITE_QUERY": 0}))
    print("=======================\n")


# =========================
# 13) CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_demo", action="store_true", help="写入 demo 稿件数据到 Mongo")
    ap.add_argument("--ask", type=str, default="", help="提问（针对稿件库）")
    ap.add_argument("--project", type=str, default=None, help="栏目/分类过滤（section_path regex）")
    ap.add_argument("--article_id", type=str, default=None, help="仅在某篇稿件内检索问答")
    ap.add_argument("--date_from", type=str, default=None, help="publish_at >= date_from")
    ap.add_argument("--date_to", type=str, default=None, help="publish_at <= date_to")
    ap.add_argument("--policy", action="store_true", help="启用 policy 智能体（推荐）")
    ap.add_argument("--critic", action="store_true", help="启用 critic 审稿修复（推荐）")
    ap.add_argument("--eval", type=str, default=None, help="评测集 JSONL 路径")
    ap.add_argument("--eval_limit", type=int, default=None)
    args = ap.parse_args()

    # 评测模型优先
    if args.eval:
        rep = run_eval(args.eval, enable_critic=bool(args.critic), limit=args.eval_limit)
        print_eval_report(rep)
        return

    # 写 demo 数据
    if args.seed_demo:
        seed_demo_data(project=args.project or "财经")
        print("seed_demo done.")
        return

    # 正式问答模式
    if not args.ask:
        print('No question. Use --ask "..."')
        return

    res = qa_agent_with_policy(
        question=args.ask,
        project_keyword=args.project,
        date_from=args.date_from,
        date_to=args.date_to,
        article_id=args.article_id,
        enable_critic=True if args.critic else False,
    ) if (args.policy or True) else {}

    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n=== answer_markdown ===")
    print(res.get("answer_markdown", ""))


if __name__ == "__main__":
    main()
