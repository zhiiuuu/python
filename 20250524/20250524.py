"""
用于实现基于大模型(LLM)的循环式智能代理对话系统
参数llm 是LLM API
把 LLM 放进一个带有工具调用能力的循环里 你就拥有了一个不断自我改进 能动式的助手 它能自动执行脚本,处理代码 甚至能安装依赖 只要你给他权限
以下是核心的9行代码
"""
# def loop(llm):
#   msg = user_input()
#   while True:
#     output, tool_calls = llm(msg)
#     print("Agent:", output)
#     if tool_calls:
#       msg = [handle_tool_call(ts) for tc in tool_calls]
#     else:
#       msg = user_input()
"""
使用 Claude 3.7 Sonnet 与 bash 工具接入做了很多实践，比如：
  1让模型直接跑 git 命令（代替你去查 Stack Overflow）
  2自动合并代码
  3自动根据类型报错修改类型
  4安装缺失的工具
  5根据环境自适应命令（如不同版本的 grep）
"""


#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "anthropic>=0.45.0",
# ]
# ///

# 导入标准库：操作系统接口
import os

# 用于执行外部 shell 命令
import subprocess

# 类型提示相关模块（不是必须，但写了代码更清晰）
from typing import Dict, List, Any, Optional, Tuple, Union

# 引入 Anthropic 的官方 SDK，用于调用 Claude 模型
import anthropic

# 主函数，程序从这里开始运行
def main():
    try:
        print("\n=== LLM Agent Loop with Claude and Bash Tool ===\n")
        print("Type 'exit' to end the conversation.\n")  # 提示用户如何退出

        # 创建一个 Claude LLM 实例，并启动循环对话
        loop(LLM("claude-3-7-sonnet-latest"))

    # 如果按 Ctrl+C，则退出程序
    except KeyboardInterrupt:
        print("\n\nExiting. Goodbye!")

    # 捕获其他所有运行错误并打印出来
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")


# 对话主循环
def loop(llm):
    # 首次输入
    msg = user_input()
    while True:
        # 调用 LLM 处理输入，返回文本结果和工具调用（如 bash）
        output, tool_calls = llm(msg)
        print("Agent: ", output)

        # 如果 LLM 想用工具（比如 bash），就调用处理器处理
        if tool_calls:
            # 处理每一个工具调用，结果变成下一轮对话输入
            msg = [handle_tool_call(tc) for tc in tool_calls]
        else:
            # 如果没有调用工具，就等待用户输入下一句
            msg = user_input()


# bash 工具的描述和输入格式（提供给 LLM 理解用）
bash_tool = {
    "name": "bash",  # 工具名称
    "description": "Execute bash commands and return the output",  # 描述
    "input_schema": {
        "type": "object",  # 输入是一个 JSON 对象
        "properties": {
            "command": {
                "type": "string",  # 只需要一个字符串参数：command
                "description": "The bash command to execute"
            }
        },
        "required": ["command"]  # command 是必须的字段
    }
}


# 执行 bash 命令并返回结果（包括标准输出、错误、退出码）
def execute_bash(command):
    try:
        # 通过 subprocess 执行 bash 命令
        result = subprocess.run(
            ["bash", "-c", command],  # 用 bash 执行命令
            capture_output=True,      # 捕获 stdout / stderr
            text=True,                # 返回字符串格式，而不是字节
            timeout=10                # 最多执行 10 秒
        )
        # 格式化输出结果
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT CODE: {result.returncode}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

# 获取用户输入
def user_input():
    x = input("You: ")  # 提示输入
    if x.lower() in ["exit", "quit"]:
        print("\nExiting agent loop. Goodbye!")
        raise SystemExit(0)  # 主动退出程序
    # 返回格式为 Claude 接收的消息格式（JSON）
    return [{"type": "text", "text": x}]


class LLM:
    def __init__(self, model):
        # 检查 API Key 是否存在
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("ANTHROPIC_API_KEY environment variable not found.")

        # 创建 Claude API 客户端
        self.client = anthropic.Anthropic()
        self.model = model
        self.messages = []  # 保存整个对话历史

        # 系统提示，告诉模型你是什么身份，有哪些工具
        self.system_prompt = """You are a helpful AI assistant with access to bash commands.
        You can help the user by executing commands and interpreting the results.
        Be careful with destructive commands and always explain what you're doing.
        You have access to the bash tool which allows you to run shell commands."""

        # 可用工具列表，提供给 Claude 用（当前只有 bash）
        self.tools = [bash_tool]

    def __call__(self, content):
        # 把用户输入加入对话历史
        self.messages.append({"role": "user", "content": content})

        # 加入 ephemeral（临时缓存标记），防止污染对话
        self.messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        # 向 Claude 发送对话请求
        response = self.client.messages.create(
            model=self.model,
            max_tokens=20_000,  # 最多返回 token 数（影响价格）
            system=self.system_prompt,
            messages=self.messages,
            tools=self.tools
        )

        # 移除临时标记
        del self.messages[-1]["content"][-1]["cache_control"]

        # 初始化 assistant 回复内容和工具调用列表
        assistant_response = {"role": "assistant", "content": []}
        tool_calls = []
        output_text = ""

        # 处理 Claude 返回的内容（可能是文本或工具调用）
        for content in response.content:
            if content.type == "text":
                text_content = content.text
                output_text += text_content
                assistant_response["content"].append({"type": "text", "text": text_content})
            elif content.type == "tool_use":
                assistant_response["content"].append(content)
                tool_calls.append({
                    "id": content.id,
                    "name": content.name,
                    "input": content.input
                })

        # 把 AI 的响应也保存到对话历史
        self.messages.append(assistant_response)
        return output_text, tool_calls

# 执行工具调用（当前只支持 bash 工具）
def handle_tool_call(tool_call):
    if tool_call["name"] != "bash":
        raise Exception(f"Unsupported tool: {tool_call['name']}")

    # 获取命令字符串
    command = tool_call["input"]["command"]
    print(f"Executing bash command: {command}")

    # 执行命令并输出
    output_text = execute_bash(command)
    print(f"Bash output:\n{output_text}")

    # 把执行结果返回给 LLM（tool_result 格式）
    return dict(
        type="tool_result",
        tool_use_id=tool_call["id"],
        content=[dict(
            type="text",
            text=output_text
        )]
    )

if __name__ == "__main__":
    main()