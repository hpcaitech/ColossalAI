import os

from transformers import AutoTokenizer
from utils import ChatPromptProcessor, Dialogue

CONTEXT = 'Below is an instruction that describes a task. Write a response that appropriately completes the request. Do not generate new instructions.'
tokenizer = AutoTokenizer.from_pretrained(os.environ['PRETRAINED_PATH'])

samples = [
    ([
        Dialogue(
            instruction='Who is the best player in the history of NBA?',
            response=
            'The best player in the history of the NBA is widely considered to be Michael Jordan. He is one of the most successful players in the league, having won 6 NBA championships with the Chicago Bulls and 5 more with the Washington Wizards. He is a 5-time MVP, 1'
        ),
        Dialogue(instruction='continue this talk', response=''),
    ], 128,
     'Below is an instruction that describes a task. Write a response that appropriately completes the request. Do not generate new instructions.\n\n### Instruction:\nWho is the best player in the history of NBA?\n\n### Response:\nThe best player in the history of the NBA is widely considered to be Michael Jordan. He is one of the most successful players in the league, having won 6 NBA championships with the Chicago Bulls and 5 more with the Washington Wizards. He is a 5-time MVP, 1\n\n### Instruction:\ncontinue this talk\n\n### Response:\n'
    ),
    ([
        Dialogue(
            instruction='Who is the best player in the history of NBA?',
            response=
            'The best player in the history of the NBA is widely considered to be Michael Jordan. He is one of the most successful players in the league, having won 6 NBA championships with the Chicago Bulls and 5 more with the Washington Wizards. He is a 5-time MVP, 1'
        ),
        Dialogue(instruction='continue this talk', response=''),
    ], 200,
     'Below is an instruction that describes a task. Write a response that appropriately completes the request. Do not generate new instructions.\n\n### Instruction:\ncontinue this talk\n\n### Response:\n'
    ),
    ([
        Dialogue(
            instruction='Who is the best player in the history of NBA?',
            response=
            'The best player in the history of the NBA is widely considered to be Michael Jordan. He is one of the most successful players in the league, having won 6 NBA championships with the Chicago Bulls and 5 more with the Washington Wizards. He is a 5-time MVP, 1'
        ),
        Dialogue(instruction='continue this talk', response=''),
    ], 211,
     'Below is an instruction that describes a task. Write a response that appropriately completes the request. Do not generate new instructions.\n\n### Instruction:\ncontinue this\n\n### Response:\n'
    ),
    ([
        Dialogue(instruction='Who is the best player in the history of NBA?', response=''),
    ], 128,
     'Below is an instruction that describes a task. Write a response that appropriately completes the request. Do not generate new instructions.\n\n### Instruction:\nWho is the best player in the history of NBA?\n\n### Response:\n'
    ),
]


def test_chat_prompt_processor():
    processor = ChatPromptProcessor(tokenizer, CONTEXT, 256)
    for history, max_new_tokens, result in samples:
        prompt = processor.preprocess_prompt(history, max_new_tokens)
        assert prompt == result


if __name__ == '__main__':
    test_chat_prompt_processor()
