"""
Prompts for Qwen Series.
"""

from coati.reasoner.guided_search.prompt_store.base import PromptCFG

Qwen32B_prompt_CFG = PromptCFG(
    base_url="http://0.0.0.0:8008/v1",
    model="Qwen2.5-32B-Instruct",
    critic_system_prompt="Provide a detailed and constructive critique to improve the answer. "
    "Highlight specific areas that need refinement or correction.",
    refine_system_prompt="""# Instruction
                            Refine the answer based on the critique. The response should begin with [reasoning process]...[Verification]... and end with [Final Answer].
                         """,
    evaluate_system_prompt=(
        "Analyze this answer strictly and critic, provide a reward score between -100 and 100 for the answer quality, using very strict standards. "
        "Do not give a full score above 95. Make sure the reward score is an integer. "
        "Return *ONLY* the score."
    ),
)
