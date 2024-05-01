import json
import multiprocessing
import os
import re
from collections import defaultdict
import subprocess
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset, load_metric
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from joblib import Parallel, delayed
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, StoppingCriteria, StoppingCriteriaList
import tempfile

EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(self, tokenizer, dataset, n_tasks=None, n_copies=1):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.n_tasks = len(dataset) if n_tasks is None else n_tasks
        self.n_copies = n_copies

    def __iter__(self):
        prompts = []
        for task in range(self.n_tasks):
            # without strip, the model generate commented codes ...
            prompts.append(self.tokenizer.eos_token + self.dataset[task]["prompt"].strip())
        outputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        for task in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "ids": outputs.input_ids[task],
                    "task_id": task,
                    "input_len": outputs.attention_mask[task].sum(),
                }


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any(stop_string in decoded_generation for stop_string in self.eof_strings))
        return all(done)


def remove_last_block(string):
    """Remove the last block of the code containing EOF_STRINGS"""
    string_list = re.split("(%s)" % "|".join(EOF_STRINGS), string)
    # last string should be ""
    return "".join(string_list[:-2])

def check_file(test, source, language):
    dolos_result = subprocess.run([f"node dolos_match_and_score.js {test} {source}"], shell=True, stdout=subprocess.PIPE)
    dolos_score = 0
    try:
        dolos_score = dolos_result.stdout.decode().split("Similarity: ")[1].replace("\n", "")
    except:
        dolos_score = 0
    return dolos_score

def evaluate_similarity(prompt, generated_text, source, language='python'):
    with tempfile.NamedTemporaryFile(mode="w+", suffix='.py') as f:
        f.write(generated_text)
        f.flush()
        dolos_score = check_file(f.name, source, language)
    return float(dolos_score)


def generate_code_with_filtering(model, tokenizer, prompt, example_solution, similarity_threshold = 0.5, max_new_tokens = 300, chunk_size = 50, device= 'cuda'):
    for _ in range(max_new_tokens // chunk_size):
        encoded = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(
            **encoded,
            max_new_tokens=chunk_size,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_texts = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)

        # similarity_scores = [evaluate_similarity(prompt, gen_text, source_paths[idx//num_sequences]) for idx, gen_text in enumerate(generated_texts)]
        similarity_scores = Parallel(n_jobs=-1)(delayed(evaluate_similarity)(
            prompt,
            gen_text,
            example_solution)
            for idx, gen_text in enumerate(generated_texts))

        if similarity_scores > similarity_threshold:
            # Roll back one token for this sequence
            current_token = outputs['sequences'][-1].item()
            while True:
                # set current token score to -inf
                outputs['scores'][-1][current_token] = -float('inf')

                # Check if all logits are -inf (no valid tokens left)
                if torch.all(outputs['scores'][-1] == -float('inf')).item():
                    outputs['sequences'][-1] = tokenizer.eos_token_id
                    generated_texts = tokenizer.decode(outputs['sequences'][:-1], skip_special_tokens=True)
                    finished = 1
                    break

                # sample new token
                next_token = torch.multinomial(torch.nn.functional.softmax(outputs['scores'][-1], dim=-1), num_samples=1)
                outputs['sequences'][-1] = next_token
                current_token = next_token.item()

                # Check the similarity score for the new token
                new_generated_text = tokenizer.decode(outputs['sequences'], skip_special_tokens=True)
                new_similarity_score = evaluate_similarity(prompt, new_generated_text, example_solution)

                if new_similarity_score <= similarity_threshold:
                    break
        if finished==0:
            generated_texts = tokenizer.decode(outputs['sequences'], skip_special_tokens=True)
    return generated_texts

def complete_code(accelerator, model, tokenizer, dataloader, n_tasks, batch_size=20, **gen_kwargs):
    """Generate multiple codes for each task in the dataset. This function leverage accelerator to distribute
    the processing to multiple GPUs.
    dataloader, a wrapper around a TokenizeDataset objectm is supposed to send all the prompts from
    the evalution dataset to the modelm as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1]
    where nc is the number of copies of the prompt, and nt is the number of tasks.
    nc is such that num_sample = nc * batch_size

    Parameters
    ----------
    accelerator: Accelerator

    model: transformers.PreTrainedModel
        Code generation model. AutoTokenizer.from_pretrained(model_ckpt), ex model_ckpt = "lvwerra/codeparrot"

    tokenizer: transformers.AutoTokenizer
        The tokenizer used to train model

    dataloader: DataLoader
        The dataloader is a wrapper around a TokenizeDataset object. It is designed to be used with multiple GPUs.

    n_tasks: int
        The number of tasks in the dataset. It is used to determine the length of the output.
        Should be aligned with the number of tasks in the TokenizeDataset.

    batch_size: int
        num_return_sequences per copy of the prompt such that num_sample = batch_size * n_copies

    gen_kwargs: dict
        Keyword arguments for the generation function of the model.

    Returns
    -------
    code_gens: list of list of str, of length n_tasks
        List of generated codes for each task.
        Each element is a list of generated codes for each task, with length num_samples
    """
    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            gen_kwargs["stopping_criteria"][0].start_length = batch["ids"].shape[-1]

            # generated_tokens = accelerator.unwrap_model(model).generate(
            #     input_ids=batch["ids"][:, : batch["input_len"]], num_return_sequences=batch_size, **gen_kwargs
            # )

            generated_tokens = generate_code_with_filtering(model, tokenizer)


            print(generated_tokens)
            print("Here")
            # each task is generated batch_size times
            generated_tasks = batch["task_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather((generated_tokens, generated_tasks))
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for task, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[task].append(generated_tokens)

    code_gens = [[] for _ in range(n_tasks)]
    for task, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            code_gens[task].append(remove_last_block(s))
    return code_gens


def main():

    transformers.logging.set_verbosity_error()
    # enables code execution in code_eval metric
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    # make sure tokenizer plays nice with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Use dataset load to feed to accelerate
    accelerator = Accelerator()
    set_seed(0, device_specific=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('codeparrot/codeparrot-small')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('codeparrot/codeparrot-small')

    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = 'left'

    # Generation settings
    gen_kwargs = {
        "do_sample": True,
        "temperature": 0.1,
        "max_new_tokens": 300,
        "top_p": 0.95,
        "top_k": 0,
        "stopping_criteria": StoppingCriteriaList([EndOfFunctionCriteria(0, EOF_STRINGS, tokenizer)]),
    }

    # Load evaluation dataset and metric
    human_eval = load_dataset("openai_humaneval")
    code_eval_metric = load_metric("code_eval")

    n_tasks = len(human_eval["test"])
    n_copies = 200 // 100

    human_eval_tokenized = TokenizedDataset(tokenizer, human_eval["test"], n_copies=n_copies, n_tasks=n_tasks)
    # do not confuse args.batch_size, which is actually the num_return_sequences
    human_eval_loader = DataLoader(human_eval_tokenized, batch_size=1)

    # Run a quick test to see if code evaluation is enabled
    try:
        _ = code_eval_metric.compute(references=[""], predictions=[[""]])
    except ValueError as exception:
        print(
            'Code evaluation not enabled. Read the warning below carefully and then use `--HF_ALLOW_CODE_EVAL="1"`'
            " flag to enable code evaluation."
        )
        raise exception

    model, human_eval_loader = accelerator.prepare(model, human_eval_loader)

    # print(human_eval)

    generations = complete_code(
        accelerator,
        model,
        tokenizer,
        human_eval_loader,
        n_tasks=n_tasks,
        batch_size=40,
        **gen_kwargs,
    )

    if accelerator.is_main_process:
        references = []

        for task in tqdm(range(n_tasks)):
            test_func = human_eval["test"][task]["test"]
            entry_point = f"check({human_eval['test'][task]['entry_point']})"
            references.append("\n" + test_func + "\n" + entry_point)

        with open('References.out', 'w') as ref_file:
            json.dump(references, ref_file)
        with open('Generation.out', 'w') as gen_file:
            json.dump(generations, gen_file)

        exit(1)
        # Evaluate completions with "code_eval" metric
        pass_at_k, _ = code_eval_metric.compute(
            references=references, predictions=generations, num_workers=2
        )
        print(f"Results: {pass_at_k}")

        # Save results to json file
        with open('Exec_results.out', "w") as fp:
            json.dump(pass_at_k, fp)


# For some reason the folliwng seems to be necessary sometimes for code_eval to work nice with multiprocessing
# https://stackoverflow.com/questions/60804599/python-multiprocessing-keeps-spawning-the-whole-script
if __name__ == "__main__":
    main()