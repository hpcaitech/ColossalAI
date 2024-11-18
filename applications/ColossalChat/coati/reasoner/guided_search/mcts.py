"""
Implementation of MCTS + Self-refine algorithm.

Reference:
1. "Accessing GPT-4 level Mathematical Olympiad Solutions via Monte
Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report"
2. https://github.com/BrendanGraham14/mcts-llm/
3. https://github.com/trotsky1997/MathBlackBox/
4. https://github.com/openreasoner/openr/blob/main/reason/guided_search/tree.py
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import tqdm
from coati.reasoner.guided_search.llm import chat_completion
from coati.reasoner.guided_search.prompt_store.base import PromptCFG
from pydantic import BaseModel


class MCTSNode(BaseModel):
    """
    Node for MCTS.
    """

    answer: str
    parent: MCTSNode = None
    children: list[MCTSNode] = []
    num_visits: int = 0
    Q: int = 0
    rewards: list[int] = []

    def expand_node(self, node) -> None:
        self.children.append(node)

    def add_reward(self, reward: int) -> None:
        self.rewards.append(reward)
        self.Q = (np.min(self.rewards) + np.mean(self.rewards)) / 2


class MCTS(BaseModel):
    """
    Simulation of MCTS process.
    """

    problem: str
    max_simulations: int
    cfg: PromptCFG
    C: float = 1.4
    max_children: int = 2
    epsilon: float = 1e-5
    root: MCTSNode = None

    def initialization(self):
        """
        Root Initiation.
        """
        # Simple answer as root. You can also use negative response such as "I do not know" as a response.
        base_answer = self.sample_base_answer()
        self.root = MCTSNode(answer=base_answer)
        self.self_evaluate(self.root)

    def is_fully_expanded(self, node: MCTSNode):
        return len(node.children) >= self.max_children or any(child.Q > node.Q for child in node.children)

    def select_node(self) -> MCTSNode:
        """
        Select next node to explore.
        """
        candidates: list[MCTSNode] = []
        to_explore = deque([self.root])

        while to_explore:
            current_node = to_explore.popleft()
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_explore.extend(current_node.children)

        if not candidates:
            return self.root

        return max(candidates, key=self.compute_uct)

    def self_evaluate(self, node: MCTSNode):
        """
        Sample reward of the answer.
        """
        reward = self.sample_reward(node)
        node.add_reward(reward)

    def back_propagation(self, node: MCTSNode):
        """
        Back propagate the value of the refined answer.
        """
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (parent.Q + best_child_Q) / 2
            parent.num_visits += 1
            parent = parent.parent

    def compute_uct(self, node: MCTSNode):
        """
        Compute UCT.
        """
        if node.parent is None:
            return -100
        return node.Q + self.C * math.sqrt(math.log(node.parent.num_visits + 1) / (node.num_visits + self.epsilon))

    def simulate(self):
        self.initialization()
        for _ in tqdm.tqdm(range(self.max_simulations)):
            node = self.select_node()
            child = self.self_refine(node)
            node.expand_node(child)
            self.self_evaluate(child)
            self.back_propagation(child)

        return self.get_best_answer()

    def get_best_answer(self):
        to_visit = deque([self.root])
        best_node = self.root

        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
            to_visit.extend(current_node.children)

        return best_node.answer

    def self_refine(self, node: MCTSNode):
        """
        Refine node.
        """
        critique_response = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.cfg.critic_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                        ]
                    ),
                },
            ],
            model=self.cfg.model,
            base_url=self.cfg.base_url,
            max_tokens=self.cfg.max_tokens,
        )
        critique = critique_response.choices[0].message.content
        assert critique is not None
        refined_answer_response = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.cfg.refine_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                            f"<critique>\n{critique}\n</critique>",
                        ]
                    ),
                },
            ],
            model=self.cfg.model,
            base_url=self.cfg.base_url,
            max_tokens=self.cfg.max_tokens,
        )
        refined_answer = refined_answer_response.choices[0].message.content
        assert refined_answer is not None

        return MCTSNode(answer=refined_answer, parent=node)

    def sample_base_answer(self):
        response = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.cfg.base_system_prompt,
                },
                {
                    "role": "user",
                    "content": f"<problem>\n {self.problem} \n</problem> \nLet's think step by step",
                },
            ],
            model=self.cfg.model,
            base_url=self.cfg.base_url,
            max_tokens=self.cfg.max_tokens,
        )
        assert response.choices[0].message.content is not None
        return response.choices[0].message.content

    def sample_reward(self, node: MCTSNode):
        """
        Calculate reward.
        """
        messages = [
            {
                "role": "system",
                "content": self.cfg.evaluate_system_prompt,
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<answer>\n{node.answer}\n</answer>",
                    ]
                ),
            },
        ]
        for attempt in range(3):
            try:
                response = chat_completion(
                    messages=messages,
                    model=self.cfg.model,
                    base_url=self.cfg.base_url,
                    max_tokens=self.cfg.max_tokens,
                )
                assert response.choices[0].message.content is not None
                return int(response.choices[0].message.content)
            except ValueError:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        },
                        {
                            "role": "user",
                            "content": "Failed to parse reward as an integer.",
                        },
                    ]
                )
                if attempt == 2:
                    raise
