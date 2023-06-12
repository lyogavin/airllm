from transformers import AutoTokenizer

from datasets import load_dataset, Dataset


model_id = "timdettmers/guanaco-33b-merged"
tokenizer = AutoTokenizer.from_pretrained(model_id)

ds = load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0")


source_template = "Below is an instruction that describes a task, paired with an input that provides further context. " \
        "Write a response that appropriately completes the request.\n\n" \
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "

ds = ds.map(lambda x: {'source_length': len(tokenizer.encode(source_template.format(**x))),
                  'target_length': len(tokenizer.encode(x['output']))})


df = ds["train"].to_pandas()


for qt in [0.8, 0.85, 0.9, 0.95, 0.98]:

    print(f"source len @qt{qt}: {df['source_length'].quantile(qt)}")
    print(f"target len @qt{qt}: {df['target_length'].quantile(qt)}")