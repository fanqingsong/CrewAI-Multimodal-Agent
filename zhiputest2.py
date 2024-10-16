from langchain_zhipu import ChatZhipuAI
from dotenv import load_dotenv

load_dotenv()



llm = ChatZhipuAI()

# invoke
print(llm.invoke("hi").content)

# stream
# for s in llm.stream("hi"):
#   print(s)
#
# # astream
# async for s in llm.astream("hi"):
#   print(s)