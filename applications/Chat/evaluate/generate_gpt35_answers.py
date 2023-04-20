#    Adapted form https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/qa_baseline_gpt35.py
#    Copyright 2023 LM-SYS@FastChat

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
import json
import os
import time
import concurrent.futures

import openai
import tqdm
import shortuuid
import logging

from utils import jload, jdump

MODEL = 'gpt-3.5-turbo'
MAX_API_RETRY = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_answer(question: str, max_tokens: int):
    answer = question
    prompt = question['instruction'] if question['input'] == "" else question['instuction'] + \
            " " + question['input']
    for _ in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                }, {
                    'role': 'user',
                    'content': prompt,
                }],
                max_tokens=max_tokens,
            )
            answer['output'] = response['choices'][0]['message']['content']
            return answer
        except Exception as e:
            logger.error(e)
            time.sleep(1)
    logger.error(f' Answer {question["id"]} failed after {MAX_API_RETRY} retries.')
    return answer

def evaluate_gpt35(args):
    questions=jload(args.dataset)
    
    logger.info(
        f' Total number of answers: {len(questions)}.')
    logger.info(
        f' Waiting for {args.request_time_gap} seconds before sending the next request.')
    
    answers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for question in questions:
            future = executor.submit(get_answer, question, args.max_tokens)
            futures.append(future)

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            answers.append(future.result())

    answers.sort(key=lambda x: x['id'])

    jdump(answers, os.path.join(args.answer_path,
          f'gpt35_answers.json'))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GPT 3.5.')
    parser.add_argument('--dataset', type=str, default="questions.json")
    parser.add_argument('--answer_path', type=str, default="answer")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--openai_key', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=1024)
    
    args = parser.parse_args()
    
    if args.openai_key is not None:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    openai.api_key = os.getenv("OPENAI_API_KEY")
        
    evaluate_gpt35(args)
