import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

API_KEY = "Dummy API Key"


def get_client(base_url: str | None = None) -> openai.Client:
    return openai.Client(api_key=API_KEY, base_url=base_url)


def chat_completion(
    messages: list[ChatCompletionMessageParam],
    model: str,
    base_url: str | None = None,
    temperature: float = 0.8,
    **kwargs,
) -> ChatCompletion:
    client = get_client(base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response
