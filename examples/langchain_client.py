from langchain.llms import ChatGLM # qwen is not registered now

llm = ChatGLM(endpoint_url="http://127.0.0.1:8000", max_token=4096, top_p=0.7, temperature=0.95, with_history=False)
print(llm.predict("你是谁"))