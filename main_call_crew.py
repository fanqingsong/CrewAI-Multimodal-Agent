
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

@tool("image analysis tool")
def image_analysis_tool(image_path:str, prompt:str) -> str:
    """IMAGE ENCODER Function"""
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

Security_Agent = Agent(
    role="Security Master",
    goal="Identify all danger points based on image path provided.",
    backstory=(
        "You are a security assistant for directing danger points in the image, report to user."
    ),
    verbose=True,
    allow_delegation=False,
    llm=defalut_llm,
    tools=[image_analysis_tool],
)

Security_task = Task(
    description=("generate response for the question {question} for image with path of {image_path} with the help of the respective tool."
    "Use the image analysis tool to describe the image provide in the image path."
    ),
    expected_output=("You should analyse the output of the tools' output"
    "If the tool is 'image_analysis_tool' then describe the image based on the question {question} and {image_path}."
    ),
    agent=Security_Agent,
)


crew = Crew(
    agents=[Security_Agent],
    tasks=[Security_task],
    verbose=True,
)

print("-------- security -----------")

inputs ={"question":"Is there any danger in this image?","image_path":"./security.jpg"}
result = crew.kickoff(inputs=inputs)
print(result.raw)


