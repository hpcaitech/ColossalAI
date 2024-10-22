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
                            Refine the answer based on the critique. Your refined answer should be a direct and concise solution to the problem.

                            ## Additional guidelines
                            - Your response should not refer to or discuss the criticisms.
                            - Do not repeat the problem statement.
                            - Respond with only the answer.
                         """,
    evaluate_system_prompt=(
        "Provide a reward score between -100 and 100 for the answer quality, using very strict standards. "
        "Do not give a full score above 95. Make sure the reward score is an integer. "
        "Return *ONLY* the score."
    ),
)
