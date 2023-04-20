import argparse
import os
import json
import copy

from utils import jdump, get_json_list


def format_questions(args):
    questions = get_json_list(args.questions_path)
    keys=questions[0].keys()
    
    formatted_questions=copy.deepcopy(questions)
    for i in range(len(formatted_questions)):
        formatted_questions[i]['instruction']=questions[i]['text']
        formatted_questions[i]['input']=""
        formatted_questions[i]['output']=""
        formatted_questions[i]['id']=questions[i]['question_id']
        for key in keys:
            if key=="category":
                continue
            del formatted_questions[i][key]
    
    jdump(formatted_questions, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions_path', type=str, default='table/question.jsonl')
    parser.add_argument('--save_path', type=str, default="table/questions.json")
    args = parser.parse_args()
    format_questions(args)