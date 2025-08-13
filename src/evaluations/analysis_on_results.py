import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_data(json_file: Path) -> pd.DataFrame:
    judgement = dict()

    with open(json_file, "r") as f:
        raw_data = json.load(f)

    for data in raw_data:
        judge_dict = data["judgement"]

        for model_name, judge in judge_dict.items():
            score_list = judgement.get(model_name, list())

            try:
                score_list.append(judge["score"])
            except Exception as _:
                score_list.append(np.nan)
                print(f"No score found in: {data['evaluation_id']=} {model_name=}")
            finally:
                judgement[model_name] = score_list

    return pd.DataFrame.from_dict(judgement)


def main():
    # json_file = "data/evaluation/llm-as-a-judge/evaluated_mistral_1.json"
    json_file = "data/evaluation/llm-as-a-judge/evaluated_mistral.json"
    judgement_scores = read_data(Path(json_file))

    print(judgement_scores.mean())


if __name__ == "__main__":
    main()
