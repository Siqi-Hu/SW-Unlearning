import json
from pathlib import Path
from typing import List

from generate_star_wars_prompts import EvaluationEntry


def load_evaluation_json(filename: Path) -> List[EvaluationEntry]:
    with open(filename, "r") as f:
        evaluations = json.load(f)

    return evaluations


def consolidate_evaluation_list(folder: str) -> List[EvaluationEntry]:
    folder = Path(folder)
    evaluation_list = list()
    for file in folder.rglob("evaluation_id_*.json"):
        evaluation_list.extend(load_evaluation_json(file))

    evaluation_list = sorted(evaluation_list, key=lambda x: x["evaluation_id"])
    return evaluation_list


def save_consolidated(
    evaluation_list: List[EvaluationEntry], output_folder: str
) -> None:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "evaluation_dataset.json"

    with open(output_file, "w") as f:
        f.write(json.dumps(evaluation_list, ensure_ascii=False, indent=4))


def main():
    folder = "./data/evaluation/llm-as-a-judge"
    evaluation_list = consolidate_evaluation_list(folder)
    save_consolidated(evaluation_list, folder)


if __name__ == "__main__":
    main()
