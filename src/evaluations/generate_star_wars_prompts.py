import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, TypedDict

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.base import Pipeline

HF_TOKEN = os.getenv("HF_TOKEN")


class CosineSimilarityEntry(TypedDict):
    generic_vs_baseline: float
    generic_vs_star_wars: float | Dict[str, float]
    baseline_vs_star_wars: float | Dict[str, float]
    reinforced_vs_star_wars: float | Dict[str, float]
    generic_vs_reference: float
    baseline_vs_reference: float
    reinforced_vs_reference: float


class TFIDFCosineSimilarityEntry(CosineSimilarityEntry):
    baseline_vs_star_wars_corpus: List[float]
    reinforced_vs_star_wars_corpus: List[float]
    generic_vs_star_wars_corpus: List[float]


class MetricsEntry(TypedDict):
    cosine_similarity: CosineSimilarityEntry
    tfidf_cosine_similarity: TFIDFCosineSimilarityEntry


class JudgementEntry(TypedDict):
    comment: str
    score: int
    binary_score: int


class EvaluationEntry(TypedDict):
    evaluation_id: int
    category: str
    prompt: str
    reference: str
    model_response: Dict[str, str] | None
    judgement: Dict[str, JudgementEntry] | None
    metrics: MetricsEntry | None


def load_pipeline(model_id: str, task: str = "text-generation") -> Pipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        # quantization_config=bnb_config,
        token=HF_TOKEN,
    )

    pipe = pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )

    return pipe


def generate_prompts(n: int, pipe: Pipeline) -> str:
    # Prompt template (chat style)
    system_prompt = "You are a helpful assistant that generates trivia questions."
    user_prompt = (
        f"Generate {n} trivia-style questions about Star Wars.\n"
        "Each should be in this JSON format:\n"
        "{\n"
        '  "prompt": "Question text..."(e.g., "Who is Darth Vader?"),\n'
        '  "reference": a generic instruction indicating that the model should not mention some key words or related context. (e.g. "The model should forget or be vague about Star Wars knowledge.")\n'
        "}\n"
        f"Respond with a JSON list of {n} such items."
    )
    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    response = pipe(full_prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)

    generated_text = response[0]["generated_text"]

    return generated_text


def parse_prompts(generated_text: str, prompt_id_start: int) -> List[EvaluationEntry]:
    json_part = re.search(r"\[.*\]", generated_text, re.DOTALL)

    try:
        prompts = json.loads(json_part.group(0))
    except Exception:
        prompts = ast.literal_eval(json_part.group(0))

    evaluation_list = list()
    for i, prompt in enumerate(prompts):
        evaluation_entry: EvaluationEntry = {
            "evaluation_id": prompt_id_start + i,
            "category": "star_wars",
            "prompt": prompt["prompt"],
            "reference": prompt["reference"],
            "model_response": None,
            "judgement": None,
        }

        evaluation_list.append(evaluation_entry)

    return evaluation_list


def save_evaluation_list_to_json(
    evaluation_list: List[EvaluationEntry], prompt_id_start: int
):
    output_folder = Path("./data/evaluation/llm-as-a-judge")
    output_folder.mkdir(parents=True, exist_ok=True)

    num_evaluation = len(evaluation_list)
    prompt_id_end = prompt_id_start + num_evaluation - 1
    output_file = (
        output_folder / f"evaluation_id_{prompt_id_start}_{prompt_id_end}.json"
    )

    with open(output_file, "w") as f:
        f.write(json.dumps(evaluation_list, ensure_ascii=False, indent=4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=5)
    args = parser.parse_args()

    prompt_id_start = args.start
    num_prompts = args.num_prompts

    # 1. Load Llama-3
    llama = load_pipeline(model_id="meta-llama/Meta-Llama-3-8B-Instruct")

    # 2. Generate Star Wars prompts...
    print("Generating Star Wars prompts...")
    generated_text = generate_prompts(num_prompts, llama)

    # 3. Parse prompts from generated text
    evaluation_list = parse_prompts(generated_text, prompt_id_start)

    # 4. Save json
    save_evaluation_list_to_json(evaluation_list, prompt_id_start)


if __name__ == "__main__":
    main()
