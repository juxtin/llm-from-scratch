#!/usr/bin/env python
# Code generated from notebooks/project_gutenberg.ipynb by script/gen-py. DO NOT EDIT.

# coding: utf-8

# # Training on Project Gutenberg
# 
# The following few sections include the code necessary to:
# 1. Preprocess the [deepmind/pg19](https://huggingface.co/datasets/deepmind/pg19) dataset and wrap it in a Dataset class that we can use for training.
# 2. Sample some batches from the dataset to see what typical text looks like.
# 3. Train a GPTModel with a context size of 512 on this corpus.
# 
# If you want to do this, better get a big cup of coffee and about 60 gigs of
# space ready. Downloading the data from HuggingFace takes a while, then you have
# to expand it and then the LazyTokenDatasetPG19 class will create a cache containing
# tokenized versions of every text.
# 
# After that, training takes about 6-8 hours to reach a plateau on my machine with an NVidia 3080.

# In[1]:


import import_ipynb
import gpt  # type: ignore
from gpt import GPTModel  # type: ignore
from training import TrainingConfig, new_training_config, train  # type: ignore
import training  # type: ignore
import re
import glob
from pathlib import Path
from datasets import load_dataset
import textwrap
import torch
import os
from torch.utils.data import Dataset, DataLoader
import tiktoken
import openai


# In[2]:


class LazyTokenDatasetPG19(Dataset):
    """Preprocesses the dataset (assumed to be deepmind/pg19!) by creating a
    directory './tokens' containing pre-tokenized versions of all books in the
    dataset. This takes a while the first time you run it (maybe 20 minutes),
    but after that it's just a few seconds.

    The initialized object is suitable for passing to Dataloader."""

    GUTENBERG_END_RE = re.compile(r"(?i)end of (the )?project gutenberg.*", re.DOTALL)
    TOO_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
    LEADING_NEWLINES_RE = re.compile(r"^\n+")

    def __init__(self, context_len: int = 256):
        super().__init__()
        self.context_len = context_len
        self.preprocess()
        self.file_paths = glob.glob("tokens/*.pt")
        self.samples: list[tuple[int, int]] = []
        print("Loading data from tokens directory")
        for i, path in enumerate(self.file_paths):
            length = torch.load(path, map_location="cpu").shape[0]
            for j in range(0, length - context_len, context_len):
                self.samples.append((i, j))
            if i % 5_000 == 0:
                print(f"Loaded up to book {i}...")
        print("Loading complete")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        book_idx, start = self.samples[idx]
        tokens = torch.load(self.file_paths[book_idx], map_location="cpu")
        input_ids = tokens[start : start + self.context_len]
        target_ids = tokens[start + 1 : start + self.context_len + 1]
        return input_ids, target_ids

    def preprocess(self):
        os.makedirs("tokens", exist_ok=True)
        existing_filepaths = glob.glob("tokens/book_*.pt")
        if (
            len(existing_filepaths) >= 28_000
        ):  # the approx size of the expanded tokens dir
            print("Preprocessing not needed.")
            return
        ds = load_dataset("deepmind/pg19", split="train")
        print("Preprocessing data to tokens directory.")
        tokenizer = tiktoken.get_encoding("gpt2")
        for i, book in enumerate(ds):
            path = Path(f"tokens/book_{i}.pt")
            if path.exists():
                continue
            text = self.clean_text(book["text"])
            if len(text) < self.context_len + 1:
                continue
            tokens = tokenizer.encode(text)
            torch.save(torch.tensor(tokens, dtype=torch.long), path)
            if i % 1_000 == 0:
                print(f"Completed preprocessing book {i}")

    def clean_text(self, text: str) -> str:
        # Remove Gutenberg end matter
        text = self.GUTENBERG_END_RE.split(text)[0]
        # Remove leading newlines/whitespace
        text = self.LEADING_NEWLINES_RE.sub("", text)
        # Collapse 3+ newlines into exactly 2 (paragraph break)
        text = self.TOO_MANY_NEWLINES_RE.sub("\n\n", text)
        # Eliminate chapter:verse markings
        text = re.sub(r"\b\d+:\d+\b", "", text)
        # Unwrap lines in each paragraph, but preserve paragraphs
        paragraphs = text.split("\n\n")
        unwrapped_paragraphs = [re.sub(r"\n", " ", p) for p in paragraphs]
        text = "\n\n".join(unwrapped_paragraphs)
        # don't allow multiple spaces in a row
        text = re.sub(r"[^\S\n]+", " ", text)
        return text.strip()


# In[3]:


def sample_loader(dataloader, n):
    """Given a DataLoader and a number of samples, prints batches from the DataLoader."""
    tokenizer = tiktoken.get_encoding("gpt2")
    i = n
    for input_batch, target_batch in dataloader:
        if i == 0:
            break
        i -= 1
        text = tokenizer.decode(input_batch.tolist()[:64])
        print(text)
        print("----------------------------------------")


# Uncomment below to see what's in the sanitized pg19 dataset.
# sample_loader(DataLoader(ltds, shuffle=True)


# In[4]:


GPT_CONFIG_MEDIUM: gpt.GPTConfigDict = {
    **gpt.GPT_CONFIG_124M,
    "context_length": 512,
}  # 1024 is just too big to train locally
GPT_CONFIG_LARGE: gpt.GPTConfigDict = {**openai.GPT_CONFIG_774M, "context_length": 512}
ltds = LazyTokenDatasetPG19(context_len=512)
training_cfg: TrainingConfig = new_training_config(
    epochs=1,
    eval_freq=500,
    peak_lr=1.5e-3,  # 1.5e-3 for 774M, 1e-3 for 355M
    max_length=512,
    # max_validation_batches=4,
)
# model = GPTModel(GPT_CONFIG_MEDIUM)
model = GPTModel(GPT_CONFIG_LARGE)
model.to(gpt.get_device())
optimizer = training.default_optimizer(model, training_cfg)


# In[ ]:


def train_pg19(name: str, dataset: Dataset = ltds, force_refresh: bool = False):
    with open("walden.txt") as f:
        walden_txt = f.read()
    walden_ds = training.GPTDatasetV1(
        walden_txt, tokenizer=tiktoken.get_encoding("gpt2"), max_length=512, stride=256
    )
    if force_refresh:
        training.load(model, optimizer, name)

    batch_size = 12
    training_loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )
    validation_loader = DataLoader(
        walden_ds,  # type: ignore
        shuffle=True,
        batch_size=4,
        drop_last=True,
    )

    train(
        model=model,
        optimizer=optimizer,
        training_loader=training_loader,
        validation_loader=validation_loader,
        cfg=training_cfg,
        metrics=training.MLflowMetrics(),
        example_generator=training.SimpleCompletion(
            prompt="John held up his hands and", max_new_tokens=24, temperature=0.4
        ),
    )


# Uncomment below to actually train the model. You won't get good results until you do.
# training.load(model, optimizer, "pg19_medium")
training.load(model, optimizer, name="pg19_755M_partial", base_path="/workspace")
training_cfg["gradient_clipping"] = True
train_pg19("new_training_run")


# In[ ]:


def prompt(model: GPTModel, txt: str, max_tokens=128, temperature=0.8):
    result = model.prompt(txt, max_tokens=max_tokens, temperature=temperature)
    print(textwrap.fill(result, width=120))


# In[ ]:


# If the model is trained, you should see some... interesting results from this.
# Otherwise it'll just be gibberish.
# prompt(model, "Ere thrice the sun done salutation to the dawn,")


# In[ ]:


training.save(model, optimizer, name="pg19_755M_partial", base_path="/workspace")


# In[ ]:




