from typing import List, Tuple
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import glob
import argparse

def get_overall_score(model_name : str, path : str, eval_config_file : str):
    json_path = os.path.join(path, '%s_evaluation_statistics.json'%model_name)
    statistics = json.load(open(json_path, 'r', encoding='utf8'))
    try:
        with open(eval_config_file, 'r', encoding='utf8') as f:
            config = json.load(f)
    except Exception as e:
        print(e)
        print("no config file found. adding now...")
        config = {}
        for key in statistics.keys():
            config[key] = {}
            config[key]['weight']=1.0
            config[key]['metrics'] = {}
            for metrics in statistics[key].keys():
                config[key]['metrics'][metrics] = 1.0/len(statistics[key])
        with open(eval_config_file, 'w', encoding='utf8') as f:
            f.write(json.dumps(config, ensure_ascii=False, indent=4))
    numerator = 0
    denominator = 0

    # calculate weighted average for a model over all tasks
    for key in config:
        if key not in statistics:
            raise(KeyError("%s not found in statistics"%key))
        # calculate weighted average for task over all evaluation metrics
        delta_numerator = 0
        delta_denominator = 0

        for metrics in statistics[key]:
            delta_numerator += statistics[key][metrics]['avg_score']*config[key]['metrics'][metrics]
            delta_denominator += config[key]['metrics'][metrics]
        if delta_denominator==0:
            raise(ValueError('divide by zero: sum of weights should be non-zero'))
        delta_numerator /= delta_denominator
        numerator += delta_numerator*config[key]['weight']
        denominator += config[key]['weight']
    if denominator==0:
        raise(ValueError('divide by zero: sum of weights should be non-zero'))
    return numerator/denominator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_statistics_dir',
                        default='./')
    parser.add_argument('--output_path', default='./evaluation_result')
    parser.add_argument('--config_path', default='./config')
    parser.add_argument('--language', default='en', choices=['en','cn'])
    args = parser.parse_args()
    model_statistics_dir = args.model_statistics_dir

    result = {}
    for model_statistics in glob.glob((model_statistics_dir if model_statistics_dir[-1]=='/' else model_statistics_dir+'/')+'*.json'):
        model_name = model_statistics.split('/')[-1].split('_')[0]
        if model_name=='overall':
            continue
        if not os.path.exists(args.config_path):
            # Create the directory
            os.makedirs(args.config_path)
            print(f"The directory '{args.config_path}' has been created.")
        else:
            print(f"The directory '{args.config_path}' already exists.")
        score = get_overall_score(model_name, model_statistics_dir, 
                                  os.path.join(args.config_path,'overall_performance_weight_%s_config.json'%args.language))
        result[model_name] = score
        print("model: %s --- score: %f"%(model_name, score))
    with open(os.path.join(args.output_path, 'overall_score_%s.json'%args.language),
              'w',encoding='utf8') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
    print(result)
    df = pd.DataFrame(list(result.items()), columns=['Model Name', 'Score'])
    print(df)
    sns.set()
    fig = plt.figure(figsize=(16, 10))
    plt.ylim((0, 4))

    fig = sns.barplot(x="Model Name", y="Score", data=df)
    fig.set_title(f"Comparison between Different Models (Overall Performance)")
    plt.xlabel("Model Name")
    plt.ylabel("Overall Score")
    plt.savefig(os.path.join(args.output_path, 'overall_score_%s.png'%args.language))
    
    



