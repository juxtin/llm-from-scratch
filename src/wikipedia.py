#!/usr/bin/env python
# Code generated from notebooks/wikipedia.ipynb by script/gen-py. DO NOT EDIT.

# coding: utf-8

# # Wikipedia
# 
# This notebook provides helpers to train on English-language Wikipedia articles.

# In[1]:


# Allow imports from ../src, which holds the de-notebookified code files
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src"))


# In[ ]:


from datasets import load_dataset

# Quick smoke test on 1%:
# ds = load_dataset("google/wiki40b", "en", split="train[:1%]")
# Full split:
# ds_train_raw = load_dataset("google/wiki40b", "en", split="train")
# ds_val_raw = load_dataset("google/wiki40b", "en", split="validation")


# In[ ]:


import re

TOK = re.compile(r"_START_(ARTICLE|SECTION|PARAGRAPH)_\s*")

def normalize(example):
    x = example["text"]
    # title = text between _START_ARTICLE_ and the next _START_*
    m = re.search(r"_START_ARTICLE_\s*(.*?)\s*_START_", x, flags=re.S)
    title = m.group(1).strip() if m else ""
    plain = TOK.sub("\n", x).strip()
    plain = re.sub("_NEWLINE_", "\n\n", plain)
    return {"title": title, "text": plain}


# In[ ]:


# pip install datasets tiktoken torch
import os, re, math, itertools
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets

# ---- Config ----
CTX = 1024
OVERLAP = 96
IGNORE_INDEX = 50256
DOC_TOK   = "<doc>"
TITLE_TOK = "<title>"
SEC_TOK   = "<sec>"
SEP_TOK   = "<doc_sep>"
enc = tiktoken.get_encoding("gpt2")

# ---- Small helpers ----
def paragraphs(text: str):
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]

def header(title: str, section: str | None = None):
    if section:
        return f"{DOC_TOK} {TITLE_TOK} {title} </title> {SEC_TOK} {section} </sec>\n\n"
    return f"{DOC_TOK} {TITLE_TOK} {title} </title>\n\n"

def chunk_article(title: str, body: str, max_tokens=CTX, overlap=OVERLAP):
    # Paragraph-preserving; starts every chunk with the header; repeats small overlap.
    paras = paragraphs(body)
    chunks, cur_ids = [], []
    def flush():
        nonlocal cur_ids
        if not cur_ids: return
        h_ids = enc.encode(header(title))
        avail = max_tokens - len(h_ids)
        body_ids = cur_ids[:avail]
        chunks.append(h_ids + body_ids)
    for p in paras:
        p_ids = enc.encode(p + "\n\n")
        if len(cur_ids) + len(p_ids) > (max_tokens - 64):
            flush()
            tail = cur_ids[-overlap:] if overlap and cur_ids else []
            cur_ids = tail[:]  # next chunk starts with overlap tail
        cur_ids.extend(p_ids)
    flush()
    return chunks  # list[list[int]]

def pack_docs(doc_chunks, ctx=CTX):
    """Greedy pack chunks into ctx windows with <doc_sep> between docs."""
    sep = enc.encode("\n" + SEP_TOK + "\n")
    packs, cur = [], []
    for ch in doc_chunks:
        add = (sep if cur else []) + ch
        if len(cur) + len(add) > ctx:
            packs.append(cur)
            cur = ch[:]  # start next with a fresh chunk (already has header)
        else:
            cur.extend(add)
    if cur:
        packs.append(cur)
    return packs

# ---- Build map-style datasets from an HF split with columns: title, text ----
def build_map_style(hf_split, ctx=CTX, overlap=OVERLAP, limit = -1):
    # 1) chunk each article
    ds_chunks = hf_split.map(
        lambda ex: {"chunks": chunk_article(ex["title"], ex["text"], ctx, overlap)},
        remove_columns=hf_split.column_names,
        desc="Chunking articles",
        num_proc=os.cpu_count() or 8,
    )
    # 2) flatten chunks
    flat = []
    for chs in ds_chunks["chunks"]:
        if limit == 0:
            break
        limit -= 1
        flat.extend(chs)
    # 3) greedy-pack to 4k sequences
    packs = pack_docs(flat, ctx=ctx)

    class PackedTokens(Dataset):
        def __init__(self, packs, ignore_index=IGNORE_INDEX):
            self.packs = packs
            self.ignore_index = ignore_index
        def __len__(self): return len(self.packs)
        def __getitem__(self, i):
            ids = np.array(self.packs[i], dtype=np.int32)
            # labels for next-token loss: ignore last position
            labels = ids.copy()
            if labels.shape[0] > 1:
                labels[:-1] = ids[1:]
            labels[-1] = self.ignore_index
            return {
                "input_ids": torch.from_numpy(ids),
                "labels": torch.from_numpy(labels),
                "length": torch.tensor(len(ids), dtype=torch.int32),
            }

    return PackedTokens(packs)

def pad_collate(batch, pad_id=0, ignore_index=IGNORE_INDEX):
    # Sequences are already <= CTX; we’ll left-pad to uniform length for efficient stacking.
    max_len = max(x["input_ids"].shape[0] for x in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels    = torch.full((B, max_len), ignore_index, dtype=torch.long)
    for i, ex in enumerate(batch):
        L = ex["input_ids"].shape[0]
        input_ids[i, :L] = ex["input_ids"]
        labels[i, :L]    = ex["labels"]
    return (input_ids, labels)

# ---- Example usage (Wiki40B) ----
# ds_train_raw = load_dataset("google/wiki40b","en", split="train")
# ds_val_raw   = load_dataset("google/wiki40b","en", split="validation")
# If you already have a cleaned/normalized dataset with `title`, `text`, substitute it here.
# train_ds = build_map_style(ds_train_raw)
# val_ds   = build_map_style(ds_val_raw)

# train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4,
#                           pin_memory=True, persistent_workers=True, collate_fn=pad_collate)
# val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2,
#                           pin_memory=True, persistent_workers=True, collate_fn=pad_collate)


# In[ ]:


from functools import partial

def loaders(batch_size = 4):
    print("Loading datasets...")
    ds_train_raw = load_dataset("google/wiki40b", "en", split="train[:20%]")
    ds_val_raw = load_dataset("google/wiki40b", "en", split="validation[:10%]")
    print("Datasets loaded")

    print("Normalizing training data...")
    ds_train = ds_train_raw.map(normalize, remove_columns=ds_train_raw.column_names, num_proc=8)
    print("Normalizing validation data...")
    ds_val = ds_val_raw.map(normalize, remove_columns=ds_val_raw.column_names, num_proc=8)
    print("Data normalized")

    print("Building training loader...")
    train_ds = build_map_style(ds_train)
    print("Building validation loader...")
    val_ds = build_map_style(ds_val)
    print("Loaders built")

    collate = lambda batch: pad_collate(batch, pad_id=0, ignore_index=IGNORE_INDEX)
    custom_loader = partial(
        DataLoader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate
    )
    return (custom_loader(train_ds), custom_loader(val_ds))


# In[ ]:




