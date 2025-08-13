import json
import re
from pathlib import Path


def clean_text(file_name):
    file = open(file_name, "r")
    raw = file.read()
    file.close()

    # remove the page number
    cleaned_text = re.sub(r"^\s*\d+\.\s*$", "", raw, flags=re.MULTILINE)

    return cleaned_text


def split_text(content, passage_list):
    # scene pattern to split the text into different passages
    scene_pattern = r"\b(INT|EXT)\.?[\s-]+.*"

    scenes = re.split(scene_pattern, content)
    scenes = [scene.strip() for scene in scenes if scene.strip()]  # remove empty scene

    for scene in scenes:
        if scene not in ["INT.", "EXT.", "INT", "EXT"]:
            passage_list.append(scene)


if __name__ == "__main__":
    root = Path("./data/star_wars_transcripts/")
    passage_list = list()
    for file in root.iterdir():
        if file.stem.startswith("star-wars"):
            cleaned_text = clean_text(file)
            split_text(cleaned_text, passage_list)

    with open(root / "split.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(passage_list, ensure_ascii=False, indent=4))
