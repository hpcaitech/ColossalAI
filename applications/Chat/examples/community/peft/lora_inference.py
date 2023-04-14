from peft import PeftModel
from transformers import AutoTokenizer,AutoModelForCausalLM
import sys
import torch
if __name__ == '__main__':
    model_path = sys.argv[1]
    peft_id = sys.argv[2]
    device = sys.argv[3]
    #init the AutoModelForCausalLM from model_path
    #init the AutoTokenizer from model_path
    #init the PeftModel from model and peft_id
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    if not peft_id == 'None':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = PeftModel.from_pretrained(model,peft_id).to(device)
    print(model)
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": -1,
        "max_new_tokens":256,
    }


    questions = []
    while True:
        question = input("Enter a question: ")
        if question == "launch":
            inputs = tokenizer(questions, return_tensors='pt',padding="longest",truncation=True)
            with torch.no_grad():
                outputs = model.generate(input_ids=inputs["input_ids"].to(device),attention_mask=inputs["attention_mask"] , **generation_kwargs)
            print(outputs)
            responses = [output for output in outputs]
            print(responses)
            print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
            questions = []
        else:
            input_prompts = "提问: " + question + " 回答: "
            questions.append(input_prompts)
        