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

# ### KV Cache
# 
# The idea behind the KV cache is that the autoregressive nature of LLM text
# generation results in a ton of redundant calculations.
# Since tokens are predicted one at a time, and only the final context vector in a
# sequence influences the prediction, we should only have to calculate one context
# vector at a time.
# 
# In order to make that possible, we cache previously-calculated context vectors
# and recall them when needed.

# In[ ]:


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
