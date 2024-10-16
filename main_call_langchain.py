
import os
from crewai import Crew,Process
from crewai import Task
from crewai_tools import tool
from crewai import LLM
import base64
import httpx
from mimetypes import guess_type
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models.openai import ChatOpenAI
# from langchain.chat_models.openai import ChatOpenAI
# from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from crewai import Agent
from dotenv import load_dotenv
from markdown_it.rules_inline import image

from custom_llm import CustomLLM
from langchain_zhipu import ChatZhipuAI


load_dotenv()

#
# print("-------------------os.environ.get OPENAI_API_BASE_URL -------------------")
# print(os.environ.get("OPENAI_API_BASE_URL"))


# defalut_llm = ChatOpenAI(openai_api_base=os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
#                         openai_api_key=os.environ.get("OPENAI_API_KEY"),
#                         temperature=0.1,
#                         # model_name=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
#                         top_p=0.3)


defalut_llmv = ChatOpenAI(base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        temperature=0.1,
                        model=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
                        top_p=0.3)

# defalut_llm = CustomLLM()

# defalut_llmv = ChatZhipuAI(
#     model="glm-4v",
#     temperature=0.5,
# )

# defalut_llm = ChatZhipuAI(
#     model="glm-4",
#     temperature=0.5,
# )

# defalut_llm = LLM(
#     model="openai/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
# )

defalut_llm = LLM(
    model="openai/glm-4",
)


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    # Default to png
    if mime_type is None:
        mime_type = 'image/png'

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

## text to image
def image2text(image_path:str,prompt:str) -> str:
  """This tool is useful when we want to generate textual descriptions from images."""
  print("aaa------------------------------")
  # Function
  # image_data = base64.b64encode(httpx.get(image_path).content).decode("utf-8")
  image_data = local_image_to_data_url(image_path)
  message = HumanMessage(
      content=[
          {"type": "text", "text": prompt},
          {
              "type": "image_url",
              "image_url": {"url": image_data},
          },
      ],
  )
  response = defalut_llmv.invoke([message])
  print(response.content)

  return response.content


## Router Tool
@tool("router tool")
def router_tool(question:str) -> str:
  """Router Function"""
  prompt = f"""Based on the Question provide below determine the following:
1. Is the question directed at describing the image ?
Question: {question}

RESPONSE INSTRUCTIONS:
- Answer either 1.
- Answer should strictly be a string.
- Do not provide any preamble or explanations except for 1.

OUTPUT FORMAT:
1
"""
  response = defalut_llm.invoke(prompt).content
  if response == "1":
    return 'image2text'
  else:
    return None  


@tool("retriver tool")
def retriver_tool(router_response:str,question:str,image_path:str) -> str:
  """Retriver Function"""
  if router_response == 'image2text':
    return image2text(image_path,question)
  else:
    return None

Router_Agent = Agent(
  role='Router',
  goal='Route user question to a image to text',
  backstory=(
    "You are an expert at routing a user question to a image to text."
    "Use the image to text to generate text describing the image based on the textual description."
    "You do not need to be stringent with the keywords in the question related to these topics. "
  ),
  verbose=True,
  allow_delegation=False,
  llm=defalut_llm,
  tools=[router_tool],
)

router_task = Task(
    description=("Analyse the keywords in the question {question}"
    "If the question {question} instructs to describe a image then use the image path {image_path} to generate a detailed and high quality images covering all the nuances secribed in the textual descriptions provided in the question {question}."
    "Based on the keywords decide whether it is eligible for a text to image or text to speech or web search."
    "Return a single word 'image2text' if it is eligible for describing the image based on the question {question} and iamge url{image_path}."
    "Do not provide any other premable or explaination."
    ),
    expected_output=("Give a choice 'image2text' based on the question {question} and image url {image_path}"
    "Do not provide any preamble or explanations except for 'image2text'."),
    agent=Router_Agent,
)


##Retriever Agent
Retriever_Agent = Agent(
    role="Retriever",
    goal="Use the information retrieved from the Router to answer the question and image path provided.",
    backstory=(
        "You are an assistant for directing tasks to respective agents based on the response from the Router."
        "Use the information from the Router to perform the respective task."
        "Do not provide any other explanation"
    ),
    verbose=True,
    allow_delegation=False,
    llm=defalut_llm,
    tools=[retriver_tool],
)

retriever_task = Task(
    description=("Based on the response from the 'router_task' generate response for the question {question} with the help of the respective tool."
    "Use the image2text tool to describe the image provide in the image url in case the router task output is 'image2text'."
    ),
    expected_output=("You should analyse the output of the 'router_task'"
    "If the response is 'image2text' then use the 'image2text' tool to describe the image based on the question {question} and {image_path}."
    ),
    agent=Retriever_Agent,
    context=[router_task],
)


crew = Crew(
    agents=[Router_Agent,Retriever_Agent],
    tasks=[router_task,retriever_task],
    verbose=True,
)


# print("-------- helmet -----------")
#
# inputs ={"question":"are they all wearing helmets in this image?","image_path":"./helmet.jpg"}
# result = crew.kickoff(inputs=inputs)
# print(result.raw)

#
# print("-------- fire -----------")
#
# inputs ={"question":"Is there any danger in this image?","image_path":"./fire.png"}
# result = crew.kickoff(inputs=inputs)
# print(result.raw)


# print("-------- smoke -----------")
#
# inputs ={"question":"Is there any danger in this image?","image_path":"./smoke.png"}
# result = crew.kickoff(inputs=inputs)
# print(result.raw)


image2text("./smoke.png", "Is there any danger in this image?")

# print("-------- security -----------")
#
# inputs ={"question":"Is there any danger in this image?","image_path":"./security.jpg"}
# result = crew.kickoff(inputs=inputs)
# print(result.raw)

















