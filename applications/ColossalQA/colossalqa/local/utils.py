"""
Generation utilities
"""
import json
from typing import List

import requests


def post_http_request(
    prompt: str, api_url: str, n: int = 1, max_tokens: int = 100, temperature: float = 0.0, stream: bool = False
) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": 1,
        "use_beam_search": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True, timeout=3)
    return response


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output
