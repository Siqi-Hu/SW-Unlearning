import json
from pathlib import Path


def prep_message(file_name, prompt_file):
    file_prompt_template = open(prompt_file, "r")
    prompt_template = file_prompt_template.read()
    file_prompt_template.close()

    with open(file_name, "r") as f:
        split_text = json.load(f)

    messages = list()

    for split in split_text:
        message = dict()

        message["role"] = "user"

        content = "{}[START_OF_PASSAGE]\n{}\n[END_OF_PASSAGE]".format(
            prompt_template, split
        )
        message["content"] = content

        messages.append(message)

    save_json_file = prompt_file.parent / "messages.json"
    with open(save_json_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(messages, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    root = Path("./data/star_wars_transcripts/")
    file_name = root / "split.json"
    prompt_file = Path("./data/dictionary/prompt_temp.txt")
    prep_message(file_name, prompt_file)
