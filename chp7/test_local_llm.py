from hello_agents import HelloAgentsLLM



from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# llm_client = HelloAgentsLLM(temperature=0)


llm_client = HelloAgentsLLM(
    provider="ollama",
    model="llama3", # 需与 `ollama run` 指定的模型一致
    base_url="http://localhost:11434/v1",
    api_key="ollama" # 本地服务同样不需要真实 Key
)

messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]

# 发起调用，think等方法都已从父类继承，无需重写
response_stream = llm_client.think(messages)

# 打印响应
# print("ModelScope Response:")
for chunk in response_stream:
#     # chunk在my_llm库中已经打印过一遍，这里只需要pass即可
    # print(chunk, end="", flush=True)
    pass