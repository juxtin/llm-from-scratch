#!/usr/bin/env python
# Code generated from notebooks/openai.ipynb by script/gen-py. DO NOT EDIT.

# coding: utf-8

# # Importing OpenAI Weights
# 
# In this notebook, I'll be attempting to import official trained model weights from OpenAI into my own GPT model.
# 
# I'll be importing code from [gpt.ipynb](./gpt.ipynb), so refer to that when necessary.
# 
# You'll need `tensorflow` for this notebook:
# 
# ```
# uv pip install --group dev
# ```

# In[14]:


import import_ipynb
# Import the notebook gpt.ipynb
import gpt # type: ignore
from gpt import get_device # type: ignore
import torch
import torch.nn as nn
import numpy as np
import tiktoken
import training # type: ignore

tokenizer = tiktoken.get_encoding("gpt2")


# ## Downloading the gpt_download.py script
# 
# This script was provided as part of the book Build a Large Language Model (From Scratch), which I'm following here.

# In[2]:


import urllib.request
from pathlib import Path

def ensure_script():
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch05/"
        "01_main-chapter-code/gpt_download.py"
    )
    filename = url.split('/')[-1]
    if Path(filename).exists():
        # nothing to do
        return
    print(f"Downloading {filename}")
    urllib.request.urlretrieve(url, filename)

ensure_script()


# ## Running gpt_download.py
# 
# This script will download the following files:
# - checkpoint
# - encoder.json
# - hparams.json
# - model.ckpt.data-00000-of-00001
# - model.ckpt.index
# - model.ckpt.meta
# - vocab.bpe

# In[3]:


from gpt_download import download_and_load_gpt2


# # Define the OpenAI model config
# 
# These are the basic hyperparameters that distinguish the various OpenAI GPT-2 models.
# We'll be focusing on the 124M version, at least initially, so we'll create `NEW_CONFIG`
# with the right settings.

# In[4]:


model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

GPT_CONFIG_124M: gpt.GPTConfigDict = gpt.GPT_CONFIG_124M.copy()
GPT_CONFIG_124M.update(model_configs["gpt2-small (124M)"])
# We had set the context_length to 256 before, but we need it back at 1024.
GPT_CONFIG_124M.update({"context_length": 1024})

# QKV Bias is not so popular anymore, but GPT-2 used it, so we will too.
GPT_CONFIG_124M.update({"qkv_bias": True})

GPT_CONFIG_355M: gpt.GPTConfigDict = gpt.GPT_CONFIG_124M.copy()
GPT_CONFIG_355M.update(model_configs["gpt2-medium (355M)"])
GPT_CONFIG_355M.update({"context_length": 1024, "qkv_bias": True})

GPT_CONFIG_774M: gpt.GPTConfigDict = gpt.GPT_CONFIG_124M.copy()
GPT_CONFIG_774M.update(model_configs["gpt2-large (774M)"])
GPT_CONFIG_774M.update({"context_length": 1024, "qkv_bias": True})

GPT_CONFIG_1558M: gpt.GPTConfigDict = gpt.GPT_CONFIG_124M.copy()
GPT_CONFIG_1558M.update(model_configs["gpt2-xl (1558M)"])
GPT_CONFIG_1558M.update({"context_length": 1024, "qkv_bias": True})


# # Create a new model based on GPT-2 and transfer weights
# 
# This could get long. We're using a helper to "safely" overwrite the weights in
# our model. There are a lot of layers to do this with.

# In[ ]:


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape},"
                         "Right: {right.shape}")
    return nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(model: gpt.GPTModel, params):
    # Restore the token embeddings and positional embeddings
    model.positional_embedding.weight = assign(model.positional_embedding.weight, params['wpe'])
    model.token_embedding.weight = assign(model.token_embedding.weight, params['wte'])

    # For each transformer block...
    for b in range(len(params["blocks"])):
        # ...restore the attention QKV weights
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        model.transformer_blocks[b].attention.w_query.weight = assign(
            model.transformer_blocks[b].attention.w_query.weight, q_w.T)
        model.transformer_blocks[b].attention.w_key.weight = assign(
            model.transformer_blocks[b].attention.w_key.weight, k_w.T)
        model.transformer_blocks[b].attention.w_value.weight = assign(
            model.transformer_blocks[b].attention.w_value.weight, v_w.T)

        # and the QKV biases
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        model.transformer_blocks[b].attention.w_query.bias = assign(
            model.transformer_blocks[b].attention.w_query.bias, q_b)
        model.transformer_blocks[b].attention.w_key.bias = assign(
            model.transformer_blocks[b].attention.w_key.bias, k_b)
        model.transformer_blocks[b].attention.w_value.bias = assign(
            model.transformer_blocks[b].attention.w_value.bias, v_b)

        # and the attention output projection
        model.transformer_blocks[b].attention.w_out.weight = assign(
            model.transformer_blocks[b].attention.w_out.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        model.transformer_blocks[b].attention.w_out.bias = assign(
            model.transformer_blocks[b].attention.w_out.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # and the FeedForward layer weights
        model.transformer_blocks[b].feedforward.layers[0].weight = assign(
            model.transformer_blocks[b].feedforward.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        model.transformer_blocks[b].feedforward.layers[0].bias = assign(
            model.transformer_blocks[b].feedforward.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        model.transformer_blocks[b].feedforward.layers[2].weight = assign(
            model.transformer_blocks[b].feedforward.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        model.transformer_blocks[b].feedforward.layers[2].bias = assign(
            model.transformer_blocks[b].feedforward.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # and the LayerNorm scale and shift weights
        model.transformer_blocks[b].layer_norm_1.scale = assign(
            model.transformer_blocks[b].layer_norm_1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        model.transformer_blocks[b].layer_norm_1.shift = assign(
            model.transformer_blocks[b].layer_norm_1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        model.transformer_blocks[b].layer_norm_2.scale = assign(
            model.transformer_blocks[b].layer_norm_2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        model.transformer_blocks[b].layer_norm_2.shift = assign(
            model.transformer_blocks[b].layer_norm_2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    # and finally, restore the final norm scale and shift layers
    model.layer_norm.scale = assign(model.layer_norm.scale, params["g"])
    model.layer_norm.shift = assign(model.layer_norm.shift, params["b"])

    # and the output head is also different in this version.
    model.output.weight = assign(model.output.weight, params["wte"])


def load_openai_model(config: gpt.GPTConfigDict, size: str) -> gpt.GPTModel:
    settings, params = download_and_load_gpt2(
        model_size=size, models_dir="gpt2"
    )
    model = gpt.GPTModel(config)
    model.eval()
    load_weights_into_gpt(model, params)
    print(f"{size} model loaded.")
    return model

# Uncomment one of the following to load that model
# model = load_openai_model(GPT_SMALL, "124M")
# model = load_openai_model(GPT_CONFIG_355M, "355M")
# model = load_openai_model(GPT_CONFIG_774M, "774M")
# model = load_openai_model(GPT_CONFIG_1558M, "1558M") # needs force_cpu=True on my system or it crashes


# In[17]:


import textwrap

def chat_gpt(model, prompt, temperature=1.5, max_tokens=128):
    base = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"
    tail = "\n\n### Response:\n"
    model_input = base + prompt + tail
    completion = training.text_completion_topk(
        model,
        model_input,
        max_new_tokens=max_tokens,
        context_size=1024,
        topk=50,
        temperature=temperature,
    )
    response = completion[len(model_input):].strip()
    print(textwrap.fill(response, width=120))


# In[ ]:


# chat_gpt(model, "Answer this question: what is 2+2?")


# In[ ]:




