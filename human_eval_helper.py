from datasets import load_metric
import json
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


references = []
generations = []

with open('References-small.out', 'r') as ref_file:
    references = json.load(ref_file)
with open('Generation-small.out', 'r') as gen_file:
    generations = json.load(gen_file)

# print(references, generations)
# exit(1)

code_eval_metric = load_metric("code_eval")
# Evaluate completions with "code_eval" metric
pass_at_k, _ = code_eval_metric.compute(
    references=references, predictions=generations, num_workers=2
)
print(f"Results: {pass_at_k}")

# Save results to json file
with open('Exec_results.out', "w") as fp:
    json.dump(pass_at_k, fp)