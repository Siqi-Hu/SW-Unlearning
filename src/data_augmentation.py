from datasets import concatenate_datasets, load_dataset

# load the dataset from a Hugging Face repo
dataset = load_dataset(
    "Siqi-Hu/Llama2-7B-generic-predictions-starwars-bootstrap-coef-10-once",
    split="train",
)

# duplicate the dataset 10 times and shuffle it
duplicated = concatenate_datasets([dataset] * 10).shuffle(seed=42)

duplicated.push_to_hub(
    "Siqi-Hu/Llama2-7B-generic-predictions-starwars-bootstrap-coef-10-once-augmented"
)
