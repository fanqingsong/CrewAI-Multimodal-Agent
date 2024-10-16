from langchain_zhipu import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm4v = ChatZhipuAI(model="glm-4v")

prompt = ChatPromptTemplate.from_messages([
    ("human", [
          {
            "type": "text",
            "text": "图里有什么"
          },
          {
            "type": "image_url",
            "image_url": {
                "url" : "https://img1.baidu.com/it/u=1369931113,3388870256&fm=253&app=138&size=w931&n=0&f=JPEG&fmt=auto?sec=1703696400&t=f3028c7a1dca43a080aeb8239f09cc2f"
            }
          }
        ]),
])

ret = (prompt|llm4v).invoke({})
print(ret.content)

