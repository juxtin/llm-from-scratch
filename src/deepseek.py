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

# In[67]:


import torch
import torch.nn as nn
from typing import Optional

class KVCache():
    def __init__(self):
        self.x: Optional[torch.Tensor] = None
        self.keys: Optional[torch.Tensor] = None
        self.vals: Optional[torch.Tensor] = None

    def cache_hit(self, x: torch.Tensor) -> bool:
        """As an extremely rough cut, just check that the x (param) looks kinda
        like the previous x plus a new row"""
        if self.x is None:
            return False
        _, cached_tokens, _ = self.x.shape
        _, incoming_tokens, _ = x.shape
        if cached_tokens != (incoming_tokens - 1):
            # Rather than require the caller to manually reset the cache, I'll just reset it
            # whenever the incoming data looks like it comes from a new sequence.
            self.reset()
            return False
        return True

    def save_keys(self, x: torch.Tensor, val: torch.Tensor):
        self.x = x
        self.keys = val

    def save_vals(self, x: torch.Tensor, val: torch.Tensor):
        self.x = x
        self.vals = val

    def get_keys(self, x: torch.Tensor) -> nn.Linear:
        assert(self.keys is not None)
        return self.keys

    def get_vals(self, x: torch.Tensor) -> nn.Linear:
        assert(self.vals is not None)
        return self.vals

    def reset(self):
        self.__init__()

class MultiHeadAttentionWithCache(nn.Module):
    def __init__(
        self,
        d_in: int,  # embedding dimension
        d_out: int, # embedding dimension
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("The number of heads must evenly divide d_out.")
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_width = d_out // num_heads
        self.qkv_bias = qkv_bias

        # construct the weights for Q, K, and V.
        # these will be registered as trainable parameters automatically.
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # create a KV cache
        self.kv_cache = KVCache()

        # and the output projection, also trainable.
        self.w_out = nn.Linear(d_out, d_out)

        # and the dropout layer. not trainable, just drops random values
        # to zero with a probability determined by the dropout parameter
        self.dropout = nn.Dropout(dropout)

        # and the mask, which prevents each token from "seeing" later ones
        mask = torch.triu(  # an upper triangular matrix
            torch.ones(context_length, context_length),  # consisting of ones
            diagonal=1,  # starting one row above the diagonal, leaving the diagonal itself as zeroes.
        )
        self.register_buffer(
            "mask", mask
        )  # register this tensor as non-trainable, but keep it on the same device
        self.mask: torch.Tensor  # to make the type-checker happy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, d_in = x.shape
        queries = self.w_query(x)

        if self.kv_cache.cache_hit(x):
            new_token: torch.Tensor = x[:, -1:, :]

            keys = self.kv_cache.get_keys(x)
            new_key_row: torch.Tensor = self.w_key(new_token)
            keys = torch.cat([keys, new_key_row], dim=1)

            values = self.kv_cache.get_vals(x)
            new_val_row: torch.Tensor = self.w_value(new_token)
            values = torch.cat([values, new_val_row], dim=1)
        else:
            keys = self.w_key(x)
            values = self.w_value(x)

        self.kv_cache.save_keys(x, keys)
        self.kv_cache.save_vals(x, values)

        # Split the last dimension of the tensors into multiple heads
        q_heads = queries.view(batch, num_tokens, self.num_heads, self.head_width)
        k_heads = keys.view(batch, num_tokens, self.num_heads, self.head_width)
        v_heads = values.view(batch, num_tokens, self.num_heads, self.head_width)

        #                                  [  0  ,     1     ,    2     ,      3    ]
        # {q,k,v}_heads now have the shape [batch, num_tokens, num_heads, head_width],
        # but we want them to be:          [batch, num_heads, num_tokens, head_width]
        q_heads = q_heads.transpose(1, 2)
        k_heads = k_heads.transpose(1, 2)
        v_heads = v_heads.transpose(1, 2)

        # now we need to calculate the raw dot-product attention scores between Q and K^T,
        # where K^T has the shape [batch, num_heads, head_width, num_tokens].
        # that gives attention_scores the shape [batch, num_heads, num_tokens, num_tokens]
        attention_scores = q_heads @ k_heads.transpose(2, 3)
        # and apply the causal mask
        mask = self.mask[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask == 1, float("-inf"))

        # and we construct the weights using softmax on the scaled final dimension
        attention_weights = torch.softmax(
            attention_scores / self.head_width**0.5, dim=-1
        )
        # and apply dropout
        attention_weights = self.dropout(attention_weights)

        #                                 [  0  ,     1    ,     2     ,     3     ]
        # attention_weights has the shape [batch, num_heads, num_tokens, num_tokens]
        # v_heads has the shape:          [batch, num_heads, num_tokens, head_width]
        # if we multiply them, we get:    [batch, num_heads, num_tokens, head_width]
        # but in the end, we want:        [batch, num_tokens, d_out]
        context = (
            attention_weights @ v_heads
        )  # [batch, num_heads, num_tokens, head_width]

        # so we need to first transpose and get [batch, num_tokens, num_heads, head_width]
        context = context.transpose(1, 2)
        # and then concatenate the last two dimensions together to get d_out
        context = context.contiguous().view(batch, num_tokens, self.d_out)
        # and multiply by the output projection
        return self.w_out(context)

    def reset(self):
        self.kv_cache.reset()


# 

# In[ ]:


class MultiHeadLatentAttentionV1(nn.Module):
    """A simple implementation of multi-head latent attention with low-rank
    key-value joint compression."""
    def __init__(
        self,
        d_in: int,  # embedding dimension
        d_out: int, # embedding dimension
        d_latent: int,    # the latent dimension
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("The number of heads must evenly divide d_out.")
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_width = d_out // num_heads
        self.qkv_bias = qkv_bias

        # construct the weights for Q, K, and V.
        # these will be registered as trainable parameters automatically.
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # The latent stuff
        self.w_dkv = nn.Linear(d_out, d_latent, bias=False)
        self.w_uk = nn.Linear(d_latent, d_out, bias=False)
        self.w_uv = nn.Linear(d_latent, d_out, bias=False)

        # create a KV cache (not used yet)
        self.ckv = torch.zeros([d_latent, d_latent]) 
        self.register_buffer(
            "ckv", self.ckv
        )

        # and the output projection, also trainable.
        self.w_out = nn.Linear(d_out, d_out)

        # and the dropout layer. not trainable, just drops random values
        # to zero with a probability determined by the dropout parameter
        self.dropout = nn.Dropout(dropout)

        # and the mask, which prevents each token from "seeing" later ones
        mask = torch.triu(  # an upper triangular matrix
            torch.ones(context_length, context_length),  # consisting of ones
            diagonal=1,  # starting one row above the diagonal, leaving the diagonal itself as zeroes.
        )
        self.register_buffer(
            "mask", mask
        )  # register this tensor as non-trainable, but keep it on the same device
        self.mask: torch.Tensor  # to make the type-checker happy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, d_in = x.shape
        queries = self.w_query(x)
        ckv = self.w_dkv(x)
        keys = self.w_uk(ckv)
        values = self.w_uv(ckv)

        # Split the last dimension of the tensors into multiple heads
        q_heads = queries.view(batch, num_tokens, self.num_heads, self.head_width)
        k_heads = keys.view(batch, num_tokens, self.num_heads, self.head_width)
        v_heads = values.view(batch, num_tokens, self.num_heads, self.head_width)

        #                                  [  0  ,     1     ,    2     ,      3    ]
        # {q,k,v}_heads now have the shape [batch, num_tokens, num_heads, head_width],
        # but we want them to be:          [batch, num_heads, num_tokens, head_width]
        q_heads = q_heads.transpose(1, 2)
        k_heads = k_heads.transpose(1, 2)
        v_heads = v_heads.transpose(1, 2)

        # now we need to calculate the raw dot-product attention scores between Q and K^T,
        # where K^T has the shape [batch, num_heads, head_width, num_tokens].
        # that gives attention_scores the shape [batch, num_heads, num_tokens, num_tokens]
        attention_scores = q_heads @ k_heads.transpose(2, 3)
        # and apply the causal mask
        mask = self.mask[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask == 1, float("-inf"))

        # and we construct the weights using softmax on the scaled final dimension
        attention_weights = torch.softmax(
            attention_scores / self.head_width**0.5, dim=-1
        )
        # and apply dropout
        attention_weights = self.dropout(attention_weights)

        #                                 [  0  ,     1    ,     2     ,     3     ]
        # attention_weights has the shape [batch, num_heads, num_tokens, num_tokens]
        # v_heads has the shape:          [batch, num_heads, num_tokens, head_width]
        # if we multiply them, we get:    [batch, num_heads, num_tokens, head_width]
        # but in the end, we want:        [batch, num_tokens, d_out]
        context = (
            attention_weights @ v_heads
        )  # [batch, num_heads, num_tokens, head_width]

        # so we need to first transpose and get [batch, num_tokens, num_heads, head_width]
        context = context.transpose(1, 2)
        # and then concatenate the last two dimensions together to get d_out
        context = context.contiguous().view(batch, num_tokens, self.d_out)
        # and multiply by the output projection
        return self.w_out(context)


# In[69]:


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


# In[ ]:


class DeepSeekTransformerBlock(nn.Module):
    """
    A single DeepSeek transformer block.
    """

    def __init__(self, cfg: DeepSeekConfigDict):
        super().__init__()
        self.layer_norm_1 = gpt.LayerNorm(cfg["emb_dim"])
        self.attention = MultiHeadLatentAttentionV1(
            cfg["emb_dim"],
            cfg["emb_dim"],
            cfg["context_length"],
            cfg["drop_rate"],
            cfg["n_heads"],
            cfg["qkv_bias"],
        )
        self.drop_rate = cfg["drop_rate"]
        self.layer_norm_2 = LayerNorm(cfg["emb_dim"])
        self.feedforward = FeedForward(cfg)
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


# In[ ]:




