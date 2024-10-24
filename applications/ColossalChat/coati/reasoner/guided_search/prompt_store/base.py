from pydantic import BaseModel


class PromptCFG(BaseModel):
    model: str
    base_url: str
    max_tokens: int = 4096
    critic_system_prompt: str
    refine_system_prompt: str
    evaluate_system_prompt: str
