#    Adapted form https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/eval_gpt_review.py
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
import re
import concurrent.futures

import openai
import tqdm
import shortuuid
import logging

from utils import jload, jdump, get_json_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_API_RETRY = 3


def get_eval(sys_prompt, user_prompt: str, answer_id: int, max_tokens: int, model: str):
    logging.basicConfig(level=logging.INFO)
    for _ in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{
                    'role': 'system',
                    'content': sys_prompt
                }, {
                    'role': 'user',
                    'content': user_prompt,
                }],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            review = response['choices'][0]['message']['content']
            return {"review": review, 'id': answer_id}
        except Exception as e:
            logger.error(e)
            time.sleep(1)
    logger.error(f' Review {answer_id} failed after {MAX_API_RETRY} retries.')
    return 'error'


def parse_score(review):
    try:
        pattern = re.compile('([0-9]|10) out of 10')
        sp = re.findall(pattern, review)
        if len(re.findall(pattern, review)) == 2:
            return [float(sp[0]), float(sp[1])]

        pattern = re.compile('a score of ([0-9]|10)')
        sp = re.findall(pattern, review)
        if len(re.findall(pattern, review)) == 2:
            return [float(sp[0]), float(sp[1])]

        pattern = re.compile('([0-9]|10)/10')
        sp = re.findall(pattern, review)
        if len(re.findall(pattern, review)) == 2:
            return [float(sp[0]), float(sp[1])]

        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception('Invalid score pair.')
    except Exception as e:
        return [-1, -1]


def gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2):
    reviewer_idx = 0
    for idx, reviewer in enumerate(reviewer_jsons):
        if reviewer['category'] == cat:
            reviewer_idx = idx
            break
    prompt_id = reviewer_jsons[reviewer_idx]['prompt_id']
    prompt_json = prompt_jsons[prompt_id-1]
    assert prompt_json['prompt_id'] == prompt_id

    sys_prompt = prompt_json['system_prompt']
    prompt_template = prompt_json['prompt_template']
    defaults = prompt_json['defaults']
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, **defaults)

    return sys_prompt, prompt, reviewer_idx+1


def evaluate(args):
    answer1_jsons = jload(args.answer_file_list[0])
    answer2_jsons = jload(args.answer_file_list[1])
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)

    assert len(answer1_jsons) == len(answer2_jsons)

    handles = []
    review_jsons = []

    total_len = len(answer1_jsons)
    question_idx_list = list(range(total_len))

    logger.info(
        f' Total number of answers: {len(answer2_jsons)}.')

    reviews = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for i in question_idx_list:
            assert answer1_jsons[i]['id'] == answer2_jsons[i]['id']
            answer_id = answer1_jsons[i]['id']

            ques = answer1_jsons[i]['instruction'] if answer1_jsons[i]['input'] == "" else answer1_jsons[i]['instuction'] + \
                " " + answer1_jsons[i]['input']
            cat = answer1_jsons[i]['category']
            ans1 = answer1_jsons[i]['output']
            ans2 = answer2_jsons[i]['output']

            sys_prompt, prompt, reviewer_id = gen_prompt(
                reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2)

            review_id = shortuuid.uuid()
            review_jsons.append({
                'review_id': review_id,
                'id': answer_id,
                'reviewer_id': reviewer_id,
                'metadata': {}
            })

            future = executor.submit(
                get_eval, sys_prompt, prompt, answer_id, args.max_tokens, args.model)
            futures.append(future)

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            reviews.append(future.result())

    reviews.sort(key=lambda x: x['id'])
    review_jsons.sort(key=lambda x: x['id'])

    ans1_score = 0
    ans2_score = 0
    better_count = 0
    worse_count = 0
    tie_count = 0
    invalid_count = 0

    better_file = []
    worse_file = []
    tie_file = []
    invalid_file = []
    output_review_file = []

    for idx, review in enumerate(reviews):
        scores = parse_score(review['review'])
        review_jsons[idx]['review'] = review['review']
        review_jsons[idx]['score'] = scores

        if scores[0] == -1 and scores[1] == -1:
            invalid_count += 1
            invalid_file.append(review_jsons[idx])
            logger.info(f' Invalid score pair: {review_jsons[idx]["id"]}.')
        else:
            if scores[0] > scores[1]:
                worse_count += 1
                worse_file.append(review_jsons[idx])
            elif scores[0] < scores[1]:
                better_count += 1
                better_file.append(review_jsons[idx])
            else:
                tie_count += 1
                tie_file.append(review_jsons[idx])
            ans1_score += scores[0]
            ans2_score += scores[1]

        output_review_file.append(review_jsons[idx])

    better_file.sort(key=lambda x: x['id'])
    worse_file.sort(key=lambda x: x['id'])
    tie_file.sort(key=lambda x: x['id'])
    invalid_file.sort(key=lambda x: x['id'])
    output_review_file.sort(key=lambda x: x['id'])

    name1 = os.path.basename(args.answer_file_list[0]).split("_answers")[0]
    name2 = os.path.basename(args.answer_file_list[1]).split("_answers")[0]
    prefix = f"{name1}_vs_{name2}"

    jdump(better_file, os.path.join(
        args.output_folder, prefix, f"{prefix}_better.json"))
    jdump(worse_file, os.path.join(
        args.output_folder, prefix, f"{prefix}_worse.json"))
    jdump(tie_file, os.path.join(
        args.output_folder, prefix, f"{prefix}_tie.json"))
    jdump(invalid_file, os.path.join(
        args.output_folder, prefix, f"{prefix}_invalid.json"))
    jdump(output_review_file, os.path.join(
        args.output_folder, prefix, f"{prefix}_review.json"))

    if os.path.exists(os.path.join(args.output_folder, "results.json")):
        results = jload(os.path.join(args.output_folder, "results.json"))
    else:
        results = {}
    results[prefix] = {'model': [name1, name2], 'better': better_count, 'worse': worse_count, 'tie': tie_count, 'win_rate': better_count /
                       (len(reviews)-invalid_count), 'score': [ans1_score/(len(reviews)-invalid_count), ans2_score/(len(reviews)-invalid_count)]}
    jdump(results, os.path.join(args.output_folder, "results.json"))

    logger.info(f' Total {invalid_count} invalid score pair(s).')
    logger.info(f' Model {name2} has {better_count} better answer(s).')
    logger.info(f' Model {name2} has {worse_count} worse answer(s).')
    logger.info(f' {tie_count} answer(s) play(s) to a tie.')
    logger.info(
        f' Win rate of model {name2}: {better_count/(len(reviews)-invalid_count):.2f}')
    logger.info(
        f' Model {name1} average score: {ans1_score/(len(reviews)-invalid_count):.2f}')
    logger.info(
        f' Model {name2} average score: {ans2_score/(len(reviews)-invalid_count):.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Model evaluation.')
    parser.add_argument('--answer_file_list', nargs='+', default=[])
    parser.add_argument('--prompt_file')
    parser.add_argument('--reviewer_file')
    parser.add_argument('--output_folder', type=str, default="./output")
    parser.add_argument('--openai_key', type=str, default=None)
    parser.add_argument('--model', type=str, default="gpt-4")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    if args.openai_key is not None:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    evaluate(args)
