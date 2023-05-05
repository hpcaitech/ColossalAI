import json
import os
import sys 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=None, help='input file')
    parser.add_argument('--output_file', type=str, default=None, help='output file')
    args = parser.parse_args()
    with open(args.input_file, 'r',encoding='UTF-8') as f:
        with open(args.output_file, 'w',encoding='UTF-8') as f2:
            for line in f:
                line = line.strip()
                data = json.loads(line)
                prompt = data['prompt']
                response = data['chosen']
                instructions = "请根据法律知识回答。提问："+prompt+"\t回答\t"+response+'\n'
                f2.write(instructions)