import requests
from typing import Any, Dict, List, Optional
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import  BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult



class CustomLLM(BaseChatModel):
    api_url: str = "http://172.20.160.1:1234/v1/chat/completions"
    model_name: str = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    streams: bool = False

    def __init__(self, **data: Any):
        super().__init__(**data)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 将 langchain messages 转换为 api 需要的格式
        messages_mapped = [
            {
                "role": "user",
                "content": message.content
            }
            for message in messages
        ]
        payload = {
            "model": self.model_name,
            "messages": messages_mapped,
            "stream": self.streams
        }
        response = requests.post(self.api_url, json=payload)
        print(response.json())
        tokens = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        print(tokens)
        message = AIMessage(
            content=tokens,
            additional_kwargs={},
            response_metadata={
                "time_in_seconds": 3,
            },
        )
        print(message)
        generation = ChatGeneration(message=message)
        print(generation)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """获取此聊天模型使用的语言模型类型。"""
        return "echoing-chat-model-advanced"

#使用这个模型
if __name__ == '__main__':
    model = CustomLLM()
    res = model.invoke("你好")
    print(res.content)

