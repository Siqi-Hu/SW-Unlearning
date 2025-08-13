import re
from pathlib import Path
from typing import List

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


class StarWarsDatasetLoader:
    """A class to load and process Star Wars transcripts into tokenized chunks."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        # pretrained_model: str,
        context_length: int = 512,
        chunk_size: int = 1024,  # only for reading files in chunks
    ):
        # self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.chunk_size = chunk_size

    def _preprocess_text(self, text: str) -> str:
        # remove the page number
        text = re.sub(r"^\s*\d+\.\s*$", "", text, flags=re.MULTILINE)

        # remove single new line symbol
        text = re.sub(r"(?<!\n)\n(?!\n)", "", text)

        # remove consecutives tabs
        text = re.sub(r"\t+", "", text)

        # replace multiple spaces with a single space: put it after removing single new line, as there could be \s\n\s
        text = re.sub(r" +", " ", text)

        # # replace double new lines symbol with only one new line symbol
        # text = re.sub(r"\n{2,}", "\n", text)
        # remove all new line symbols
        text = re.sub(r"\n", "", text)

        # add <|endoftext|> after scene break
        # text = re.sub(r"(^|\n)(\s*)?(\d+)?(\.?\s*)?(INT|EXT)(\.?\s*)?", f"\\1{self.tokenizer.eos_token}\\n\\3\\4\\5\\6", text)
        text = re.sub(
            r"(^|\s*)(\d+)?(\.?\s*)?(INT|EXT)(\.?\s*)?",
            f"\\1{self.tokenizer.eos_token} \\2\\3\\4\\5",
            text,
        )

        return text

    def read_text_files_into_chunks_of_tokens(
        self, input_file_dir: str | Path, overlap_ratio: float = 0.1
    ) -> List[List[torch.tensor]]:
        """
        Reads text files from the given directory, preprocesses and tokenizes the content,
        and splits it into chunks of tokens with overlapping.

        Args:
            input_file_dir (str | Path): Directory containing text files.
            overlap_ratio (float): Proportion of overlap between chunks (0.0 to 0.5).
                                Default is 0.1 (10% overlap).

        Returns:
            List[List[torch.tensor]]: A list of token chunks, where each chunk contains
                                    up to `context_length` tokens with overlapping.
        """
        tokenized_text = []
        end_of_text_token = self.tokenizer.encode(
            self.tokenizer.eos_token, return_tensors="pt", add_special_tokens=False
        )[0]  # Tokenize the end-of-text marker

        for file in Path(input_file_dir).iterdir():
            if file.stem.startswith("star-wars"):
                # append the token <|endoftext|> at the beginning of each file.
                tokenized_text.extend(end_of_text_token)

                with open(file, "r", encoding="utf-8") as file:
                    while True:
                        chunk = file.read(self.chunk_size)

                        if not chunk:
                            break

                        # preprocess the chunk
                        chunk = self._preprocess_text(chunk)

                        chunk_tokens = self.tokenizer.encode(
                            chunk, return_tensors="pt"
                        )[0]  # pytorch tensor of tokenized text for a chunk
                        tokenized_text.extend(chunk_tokens)

                # append the token of <|endoftext|> at the end of each file.
                tokenized_text.extend(end_of_text_token)

        # For distilgpt2 model tokenizer, (updated on 16/03/2025)
        # total tokens from all star wars text files after removing the page number: 386830
        # default context length: 512
        # number of chunks: 756, where the last chunk contains only 279 tokens.

        # Calculate step size for overlapping chunks
        overlap_tokens = int(self.context_length * overlap_ratio)
        step_size = self.context_length - overlap_tokens

        # Create overlapping chunks
        chunks = []
        for i in range(0, len(tokenized_text), step_size):
            chunk = tokenized_text[i : i + self.context_length]
            if len(chunk) > 0:  # Only add non-empty chunks
                chunks.append(chunk)
            # Stop if we've reached the end and don't have a full context_length chunk
            if i + self.context_length >= len(tokenized_text):
                break

        return chunks

    def load_dataset(
        self,
        input_file_dir: str | Path,
        split_ratio: float = 0.9,
        overlap_ratio: float = 0.1,
    ) -> Dataset:
        """
        Load Star Wars text files and return a HuggingFace Dataset.

        Args:
            input_file_dir (str | Path): Directory containing Star Wars transcripts text files

        Returns:
            Dataset: HuggingFace Dataset containing tokenized chunks
        """
        chunks = self.read_text_files_into_chunks_of_tokens(
            input_file_dir, overlap_ratio
        )
        dataset = Dataset.from_dict({"input_ids": chunks})

        # dataset = dataset.shuffle(seed=42)

        split_index = int(len(dataset) * split_ratio)
        train_dataset = dataset.select(range(split_index))
        validation_dataset = dataset.select(range(split_index, len(dataset)))

        return DatasetDict({"train": train_dataset, "validation": validation_dataset})


class StarWarsStringChunks:
    def __init__(
        self, context_length: int = 128, overlap: int = 50, chunk_size: int = 1024
    ):
        self.chunk_size = chunk_size
        self.context_length = context_length
        self.overlap = overlap

    def _preprocess_text(self, text: str) -> str:
        # remove the page number
        text = re.sub(r"^\s*\d+\.\s*$", "", text, flags=re.MULTILINE)

        # remove single new line symbol
        text = re.sub(r"(?<!\n)\n(?!\n)", "", text)

        # remove consecutives tabs
        text = re.sub(r"\t+", "", text)

        # replace multiple spaces with a single space: put it after removing single new line, as there could be \s\n\s
        text = re.sub(r" +", " ", text)

        # # replace double new lines symbol with only one new line symbol
        # text = re.sub(r"\n{2,}", "\n", text)
        # remove all new line symbols
        text = re.sub(r"\n", "", text)

        # add <|endoftext|> after scene break
        # text = re.sub(r"(^|\n)(\s*)?(\d+)?(\.?\s*)?(INT|EXT)(\.?\s*)?", f"\\1{self.tokenizer.eos_token}\\n\\3\\4\\5\\6", text)
        text = re.sub(
            r"(^|\s*)(\d+)?(\.?\s*)?(INT|EXT)(\.?\s*)?",
            "\\1 \\2\\3\\4\\5",
            text,
        )

        return text.strip()

    def _chunk_text_by_words(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.context_length
            chunk = words[start:end]
            chunks.append(" ".join(chunk))
            start += self.context_length - self.overlap
        return chunks

    def get_all_processed_strings(self, input_file_dir: str | Path):
        all_text = ""

        for file in Path(input_file_dir).iterdir():
            if file.stem.startswith("star-wars"):
                with open(file, "r", encoding="utf-8") as file:
                    while True:
                        chunk = file.read(self.chunk_size)

                        if not chunk:
                            break

                        cleaned_text = self._preprocess_text(chunk)
                        all_text += cleaned_text + " "

        return all_text

    def get_chunks_of_strings(self, input_file_dir: str | Path):
        all_text = self.get_all_processed_strings(input_file_dir)

        return self._chunk_text_by_words(all_text)
