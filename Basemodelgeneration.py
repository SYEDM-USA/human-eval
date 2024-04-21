# This is an example script that uses CodeParrot to generate programs for a given set of prompts.
# model options: codeparrot/codeparrot, codeparrot/codeparrot-small

from transformers import pipeline
import os
import sys
from datetime import datetime
import json
import jsonlines

def write_to_jsonl(task_id, completion):
    filename = f'data/{model}_HumanEval_samples.jsonl'
    with open(filename, 'a', encoding='utf8') as json_file:
        json_file.write(json.dumps({"task_id": task_id, "completion": completion}))
        json_file.write('\n')

def codeparrot(task_id, prompts, model_name, num_gen_per_prompt, num_prompts_per_gen=1):
    if type(prompts) == str:
        prompts = [prompts]
    pipe = pipeline("text-generation", model=model_name, max_new_tokens=300, pad_token_id=50256, device=0, batch_size=num_prompts_per_gen)
    
    if num_prompts_per_gen > 1 and pipe.model.__class__.__name__.startswith("GPT2"):
        # IMPORTANT: Change the configuration of tokenizer to make batching work for GPT2
        # cf.
        #   https://github.com/huggingface/transformers/issues/21202 
        # Since gpt2 doesn't have a pad_token
        if not pipe.tokenizer.special_tokens_map.get("pad_token"):
            pad_token = {"pad_token":"<|endoftext|>"}
            pipe.tokenizer.add_special_tokens(pad_token)
            pipe.model.resize_token_embeddings(len(pipe.tokenizer))
        # Make sure the padding_side is 'left' (if you open gpt2tokenizer you will find that by default
        # the padding_side is 'right')
        # cf.
        #   https://github.com/huggingface/transformers/issues/18478
        #   https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2
        pipe.tokenizer.padding_side = "left" #For BERT like models use "right"

    gen_start_time = datetime.now()
    outputs = pipe(prompts, num_return_sequences=num_gen_per_prompt)
    # print("Output: ", outputs)
    print(f"Generated {num_prompts_per_gen} prompts * {num_gen_per_prompt} files: {(datetime.now() - gen_start_time).total_seconds()} [sec]")
    for i in range(num_prompts_per_gen):
        for j in range(num_gen_per_prompt):
            if "generated_text" in outputs[i][j]:
                write_to_jsonl(task_id, outputs[i][j])
                # print(outputs[i][j]["generated_text"])
                # print("--------------------------------------------------------------------------")

if __name__=='__main__':

    start_time = datetime.now()

    model_name = sys.argv[1]
    num_gen_per_prompt = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if model_name == "codeparrot/codeparrot":
        model = "CodeParrot"
    elif model_name == "codeparrot/codeparrot-small":
        model = "CodeParrotSmall"
    else:
        print("Usage: python3 Example_Parrot.py codeparrot/codeparrot <number per prompt , pass@k>")
        exit(1)
    prompt_file = "data/HumanEval.jsonl"
    if not os.path.exists(prompt_file):
        print("No HumanEval.jsonl file found in data folder")

    ## READ THE PROMPTS AND STORE IN ARRAY : LOOP OVER THEM AND STORE IN data/<model_name>HumanEval_samples.jsonl

    with jsonlines.open(prompt_file, "r") as f:
        print(f"{prompt_file} undergoing generation...")
        for line in f.iter():
            # print(line['prompt'])
            codeparrot(line['task_id'], line['prompt'], model_name, num_gen_per_prompt, num_prompts_per_gen=1)
    
    end_time = datetime.now()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time in hours, minutes, and seconds
    print(f'Time elapsed: {elapsed_time.days} days, {elapsed_time.seconds//3600} hours, {(elapsed_time.seconds//60)%60} minutes, {elapsed_time.seconds%60} seconds')