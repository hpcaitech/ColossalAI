"""
Prompts for Qwen Series.
"""

from coati.reasoner.guided_search.prompt_store.base import PromptCFG

Qwen32B_prompt_CFG = PromptCFG(
    base_url="http://0.0.0.0:8008/v1",
    model="Qwen2.5-32B-Instruct",
    base_system_prompt="The user will present a problem. Analyze and solve the problem in the following structure:\n"
    "Begin with [Reasoning Process] to explain the approach. \n Proceed with [Verification] to confirm the solution. \n Conclude with [Final Answer] in the format: 'Answer: [answer]'",
    critic_system_prompt="Provide a detailed and constructive critique of the answer, focusing on ways to improve its clarity, accuracy, and relevance."
    "Highlight specific areas that need refinement or correction, and offer concrete suggestions for enhancing the overall quality and effectiveness of the response.",
    refine_system_prompt="""# Instruction
                            Refine the answer based on the critique. The response should begin with [reasoning process]...[Verification]... and end with [Final Answer].
                         """,
    evaluate_system_prompt=(
        "Critically analyze this answer and provide a reward score between -100 and 100 based on strict standards."
        "The score should clearly reflect the quality of the answer."
        "Make sure the reward score is an integer. You should only return the score. If the score is greater than 95, return 95."
    ),
)
