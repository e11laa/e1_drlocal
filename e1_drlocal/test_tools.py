from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool

@tool
def fake_tool():
    """fake tool"""
    return "ok"

try:
    llm = ChatOllama(model="nemotron-3-nano:4b").bind_tools([fake_tool])
    print("nemotron binding ok")
except Exception as e:
    print(f"nemotron err: {e}")

try:
    llm = ChatOllama(model="qwen3.5:0.8b").bind_tools([fake_tool])
    print("qwen binding ok")
except Exception as e:
    print(f"qwen err: {e}")

try:
    llm = ChatOllama(model="gpt-oss:20b").bind_tools([fake_tool])
    print("gpt-oss binding ok")
except Exception as e:
    print(f"gpt-oss err: {e}")
