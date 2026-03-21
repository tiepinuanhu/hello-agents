from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

def _prompt_hyde(query: str) -> Optional[str]:
    """生成假设性文档用于改善检索"""
    try:
        from hello_agents import HelloAgentsLLM
        llm = HelloAgentsLLM()
        prompt = [
            {"role": "system", "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
            {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"}
        ]
        return llm.invoke(prompt)
    except Exception:
        return None



if __name__ == "__main__":
    query = "什么是机器学习？"
    hyde_doc = _prompt_hyde(query)
    print("生成的假设性文档：")
    print(hyde_doc)