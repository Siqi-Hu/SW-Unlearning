import json
from collections import Counter
from pathlib import Path


def check_missing_files(list_of_files):
    file_ids = [int(int(file.stem.split("_")[1]) / 10) for file in list_of_files]
    start, end = file_ids[0], file_ids[-1]

    num_of_files = len(file_ids)

    missing_id = sorted(set(range(start, end + 1)).difference(set(file_ids)))

    # missing_file_names = ["dictionary_{}_{}.json".format(id * 10, id * 10 + 9) for id in missing_id]

    print("Total # of files: ", num_of_files)
    # print("Missing id: ", missing_id )
    assert len(missing_id) == 0, "Missing file id: %s" % (missing_id)
    # print("Missing file names: ", missing_file_names)


def merge_dict(list_of_files):
    merged_dict = dict()

    for file in list_of_files:
        with open(file, "r") as f:
            dict_messages = json.load(f)

        for dict_message in dict_messages:
            dictionary = dict_message["dictionary"]
            for key, value in dictionary.items():
                merged_dict[key.strip()] = merged_dict.get(key, []) + [value]

    # convert the value of merged_dict to dict
    converted_merged_dict = {
        anchor_term: dict(
            sorted(
                Counter(generic_translations).items(), key=lambda x: x[1], reverse=True
            )
        )
        for (anchor_term, generic_translations) in merged_dict.items()
    }

    # select the generic translation who has appeared the most times, if it is a tie, selected the first one
    consolidated_dict = {
        anchor_term: list(generic_translations.keys())[0]
        for (anchor_term, generic_translations) in converted_merged_dict.items()
    }

    # remove "" in the dict
    del consolidated_dict[""]

    # remove anchor terms that are splittable, and each of the splits are in the dictionary, such as "BLASTER FIRE" where "BLASTER", "FIRE" are both in the dictionary
    # TODO
    consolidated_dict = {
        k: v
        for k, v in consolidated_dict.items()
        if not splittable_key(consolidated_dict, k)
    }

    print(
        "Total number of entries in anchor expressions dictionary: ",
        len(consolidated_dict),
    )
    return consolidated_dict


# source: from Harry Potter's paper
def splittable_key(dict, key):
    # If "Harry's" and "Harry" are both in the dict, we remove the former
    # This may need to be adapted for different tokenizers
    if key[-2:] == "'s" and key[:-2] in dict.keys():
        return True

    words = key.split()
    if len(words) == 1:
        return False

    return all([word in dict.keys() for word in words])


if __name__ == "__main__":
    dict_path = Path("./data/dictionary")
    files = sorted(
        dict_path.glob("dictionary_*.json"), key=lambda x: int(x.stem.split("_")[1])
    )
    check_missing_files(files)
    consolidated_dict = merge_dict(files)
    consolidated_dict_path = dict_path / "consolidated_dictionary.json"
    with open(consolidated_dict_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(consolidated_dict, ensure_ascii=False, indent=4))
    # print(consolidated_dict)
