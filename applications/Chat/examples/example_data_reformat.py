jsonl_file = 'seed_prompts_xx.jsonl'  # seed_prompts_en.jsonl or seed_prompts_ch.json from InstructionWild
reformat_file = 'prompts_xx.jsonl'  # reformat jsonl file used as Prompt dataset in Stage3

data = ''
with open(jsonl_file, 'r', encoding="utf-8") as f1:
    for jsonstr in f1.readlines():
        jsonstr = '\t' + jsonstr.strip('\n') + ',\n'
        data = data + jsonstr
    data = '[\n' + data + ']'

with open(reformat_file, 'w') as f2:
    f2.write(data)