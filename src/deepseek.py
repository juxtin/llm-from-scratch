#!/usr/bin/env python
# Code generated from notebooks/deepseek.ipynb by script/gen-py. DO NOT EDIT.

# coding: utf-8

# # DeepSeek
# 
# In this notebook, I'll be experimenting with the architectural enhancements that characterize DeepSeek.

# ## Attention
# 
# The attention mechanism used by DeepSeek is Multi-Head Latent Attention (MLA), so I'll be working towards that.
# But rather than immediately jumping to MLA, I'll build up to it in steps:
# 
# 1. MHA with a KV cache
# 2. Multi-Query Attention
# 3. Grouped-Query Attention
# 4. Multi-Head Latent Attention

# ### Naive KV Cache
# 
# The idea behind the KV cache is that the autoregressive nature of LLM text
# generation results in a ton of redundant calculations.
# 
# Specifically, for every token `x` in the input sequence, MHA starts with:
# 
# ```python
# queries = self.w_query(x)
# keys = self.w_key(x)
# values = self.w_value(x)
# ```
# 
# Each of the layers `w_{query,key,value}` is of size $\text{embedding\_dimension}^2$,
# or $1024^2 = 1,048,576$ in the GPT-2 medium model. In naive MHA, _every token_ in
# the context gets multiplied by that _every time_ we generate a new token.
# 
# That means that to generate 100 tokens, we need a cumulative $2\sum_{i=1}^{100} i*1024^2$ individual
# multiply-adds just to generate the K and V tensors. That's over _10 billion_.
# 
# If, on the other hand, we cache all previously-computed K and V tensors and just add the row
# for each new token in the sequence, we can reduce that to:
# 
# $2 * 100 * 1024^2 \approx 210\text{ million}$
# 
# We still need $\sum_{i=1}^{100} i*1024^2$ operations to calculate the Q tensors,
# but saving almost 10 billion operations is a huge win. There are strategies relating
# to the Q tensors that we can try out later.
# 
# #### the downside
# 
# Unfortunately, there is a catch. Assuming one byte per tensor element, the peak cache size 
# for 512 tokens in GPT-2 Medium is:
# 
# $\text{layers} * 2 * \text{tokens} * \text{embedding} = 24 * 2 * 512 * 1024 \approx 26\text{ MB} $
# 
# Pretty manageable, right? But what about GPT-3, with 12,288 embedding dimensions, 96 layers, and a
# context size of 2048?
# 
# $96 * 2 * 2048 * 12,288 \approx 4.8\text{ GB}$
# 
# That's really pushing it for a cache, and we're nowhere near the size of GPT-4. The bottom line is that
# a naive KV caching strategy is great for smaller models, but it doesn't scale well to larger sizes. For
# that, we'll need to get much more clever about it.

# In[ ]:


import torch
import torch.nn as nn


# 

# In[ ]:


class MultiHeadLatentAttentionV1(nn.Module):
    """A simple implementation of multi-head latent attention with low-rank
    key-value joint compression."""
    def __init__(
        self,
        emb_dim: int, # the embedding dimension
        d_latent: int,    # the latent dimension
        context_length: int,
        num_heads: int,
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("The number of heads must evenly divide d_out.")
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_width = emb_dim // num_heads

        # For now, the query weights are classic flavor
        self.w_query = nn.Linear(emb_dim, emb_dim)

        # The latent stuff
        self.w_dkv = nn.Linear(emb_dim, d_latent, bias=False)
        self.w_uk = nn.Linear(d_latent, emb_dim, bias=False)
        self.w_uv = nn.Linear(d_latent, emb_dim, bias=False)

        # We have our own LayerNorm now, unlike in GPT
        self.ln = nn.LayerNorm(d_latent, d_latent)

        # and the output projection, also trainable.
        self.w_out = nn.Linear(emb_dim, emb_dim, bias=False)

        # save a place for the absorbed K (w_query @ w_uk)
        self.register_buffer("absorbed_k", None)

        # and the mask, which prevents each token from "seeing" later ones
        mask = torch.triu(  # an upper triangular matrix
            torch.ones(context_length, context_length),  # consisting of ones
            diagonal=1,  # starting one row above the diagonal, leaving the diagonal itself as zeroes.
        )
        self.register_buffer(
            "mask", mask
        )  # register this tensor as non-trainable, but keep it on the same device
        self.mask: torch.Tensor  # to make the type-checker happy

    def forward(
        self, x: torch.Tensor, c_kv: torch.Tensor, past_tokens: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape # Batch, Sequence, Dimension (embedding)
        if self.absorbed_k is None:
            self.absorbed_k = torch.matmul(self.w_queries.weight, self.w_uk.weight).view(self.num_heads, self.head_width, self.d_latent)

        new_kv_rows = self.w_dkv(x)
        c_kv = torch.cat([c_kv, new_kv_rows])
        S_full = c_kv.size[1]

        values = self.w_uv(c_kv).view(B, self.num_heads, S_full, self.head_width)
        queries = x.view(B, S, self.num_heads, self.head_width) # no unique queries var because of absorbed_k
        # NOTE: no keys variable because of absorbed_k

        attention_scores = torch.zeros([B, self.num_heads, S, S_full], device=x.device) # new attention scores only
        for h in range(self.num_heads):
            attention_h = (queries[:, :, h] @ self.absorbed_k[h]).view(B, S, S_full)
            attention_scores[:, h] = torch.bmm(attention_h, c_kv)

        mask = torch.tril(torch.ones([S, S_full], device=x.device), diagonal=past_tokens)
        attention_scores = attention_scores.masked_fill(mask.view(1, 1, S, S_full) == 0, float("-inf")) / self.head_width ** 0.5
        attention_scores = torch.softmax(attention_scores, -1)

        out_heads = []
        for h in range(self.num_heads):
            context_h = torch.matmul(attention_scores[:, h], values[:, h])
            out_heads.append(context_h)

        out = torch.cat(out_heads, dim=-1)

        return self.w_out(out), c_kv


# In[77]:


import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src"))

import gpt

class DeepSeekConfigDict(gpt.GPTConfigDict):
    latent_dim: int # the size of the latent cache in MLA

DeepSeekSmall: DeepSeekConfigDict = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "latent_dim": 12,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


# In[78]:


class DeepSeekTransformerBlock(nn.Module):
    """
    A single DeepSeek transformer block.
    """

    def __init__(self, cfg: DeepSeekConfigDict):
        super().__init__()
        self.layer_norm_1 = gpt.LayerNorm(cfg["emb_dim"])
        self.attention = MultiHeadLatentAttentionV1(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            d_latent=cfg["latent_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
        )
        self.drop_rate = cfg["drop_rate"]
        self.layer_norm_2 = gpt.LayerNorm(cfg["emb_dim"])
        self.feedforward = gpt.FeedForward(cfg)
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm_2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + shortcut
        return x


# In[ ]:


class DeepSeekModel(nn.Module):
    """
    An in-progress rewrite of the GPTModel class using re-implementations of
    the DeepSeek architecture.
    """
    def __init__(self, cfg: DeepSeekConfigDict):
        """Initialize model with config."""
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.positional_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[gpt.TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.layer_norm = gpt.LayerNorm(cfg["emb_dim"])
        self.output = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def clear(self):
        pass

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass: input indices to logits."""
        batch_size, sequence_length = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)
        positional_embeddings = self.positional_embedding(
            # get the first N positional embeddings, where N is the sequence length
            torch.arange(sequence_length, device=in_idx.device)
        )

        x = token_embeddings + positional_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        logits = self.output(x)
        return logits

    def device(self) -> torch.device:
        return next(self.parameters()).device


# In[80]:


import tiktoken

def generate_text_simple(model, idx, max_new_tokens, context_size, device=gpt.get_device()):
    """
    A helper function used by smoke_test. It's easier to pass the prompt to smoke_test, rather than call this directly.
    """
    idx.to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probabilities, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def smoke_test(prompt):
    """
    Pass the prompt to the (untrained) GPT model with a manual seed. Should correspond to the expected output.
    """
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = DeepSeekModel(DeepSeekSmall)
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    model.eval()
    out = generate_text_simple(
        model, encoded_tensor, 6, DeepSeekSmall["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


if __name__ == "__main__":
    smoke_test(
        "Hello, I am"
    )  # should output "Hello, I am Featureiman Byeswickattribute argue"


# In[ ]:




