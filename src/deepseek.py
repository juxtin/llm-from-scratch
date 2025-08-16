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

# In[17]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src"))
import gpt


# # RoPE
# 
# For each vector $x \in \mathbb{R}^{d}$ at position $i$, produce a rotated version $x^{(i)}$:
# 
#   - $x_\text{even}^{(i)} = x_\text{even} \cos \theta_i - x_\text{odd} \sin \theta_i$
# 
#   - $x_\text{odd}^{(i)} = x_\text{even} \sin \theta_i + x_\text{odd} \cos \theta_i$
# 
#   - where $\theta_i$ is a frequency-based position angle.

# In[18]:


class RoPE(nn.Module):
    def __init__(self, head_dim: int, context_length: int, base: int = 10_000):
        super().__init__()
        D = head_dim
        k = torch.arange(D//2) # [1, D//2]
        positions = torch.arange(context_length) # [1, context_length]
        freqs = base ** (- (2 * k) / D) # [1, D//2]
        angles = torch.outer(positions, freqs) # [context_length, D//2]
        self.register_buffer(
            "sin",
            torch.sin(angles) # [context_length, D//2]
        )
        self.register_buffer(
            "cos",
            torch.cos(angles) # [context_length, D//2]
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a rotation pair-wise to the final dimension of x (taken to be a
        tensor of [batch, seq, head_dim]) to encode positional information to the
        tokens.
        """
        B, S, D = x.shape

        x_even = x[..., 0::2] # [B, S, D//2]
        x_odd = x[..., 1::2] # [B, S, D//2]

        sin = self.sin[:S, :]
        cos = self.cos[:S, :]

        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_even * cos - x_odd * sin
        x_rotated[..., 1::2] = x_even * sin + x_odd * cos

        return x_rotated


# 

# In[19]:


class MultiHeadLatentAttentionWithRoPE(nn.Module):
    """A more complete implementation of MLA with low-rank joint key-value compression and RoPE."""
    def __init__(
        self,
        emb_dim: int, # the embedding dimension
        d_latent: int,    # the latent dimension
        num_heads: int,
        context_length: int,
        rope_k: Optional[RoPE] = None,
        rope_q: Optional[RoPE] = None, 
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("The number of heads must evenly divide d_out.")
        self.emb_dim = emb_dim
        self.d_latent = d_latent
        self.num_heads = num_heads
        self.head_width = emb_dim // num_heads

        # Latent down
        self.w_dkv = nn.Linear(emb_dim, d_latent, bias=False) # [d_latent, emb_dim]
        self.w_dq = nn.Linear(emb_dim, d_latent, bias=False)  # [d_latent, emb_dim]

        # Latent up
        self.w_uk = nn.Linear(d_latent, emb_dim, bias=False)  # [emb_dim, d_latent]
        self.w_uv = nn.Linear(d_latent, emb_dim, bias=False)  # [emb_dim, d_latent]
        self.w_uq = nn.Linear(d_latent, emb_dim, bias=False)  # [emb_dim, d_latent]

        # RoPE
        self.rope_k = rope_k or RoPE(self.head_width, context_length=context_length)
        self.rope_q = rope_q or RoPE(self.emb_dim, context_length=context_length)
        self.w_kr = nn.Linear(d_latent, self.head_width, bias=False)  # [head_width, d_latent]
        self.w_qr = nn.Linear(d_latent, emb_dim, bias=False)  # [emb_dim, d_latent]

        # and the output projection, also trainable.
        self.w_out = nn.Linear(emb_dim, emb_dim, bias=False) # [emb_dim, emb_dim]

        # We have our own LayerNorm now, unlike in GPT
        self.ln = nn.LayerNorm(d_latent)

        mask = torch.triu(
            torch.ones(context_length, context_length),
            diagonal=1,
        )
        self.register_buffer(
            "mask", mask
        )
        self.mask: torch.Tensor

    def forward(self, x: torch.Tensor, c_kv: Optional[torch.Tensor] = None, k_r: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (logits, c_kv, k_r)
        """
        B, S, D = x.shape # Batch, Sequence, Dimension (embedding)

        # Phase 1: No RoPE
        #  1. Absorption trick/compute q_c
        absorbed_q = self.w_dq.weight.T @ self.w_uq.weight.T @ self.w_uk.weight
        q_c = x @ absorbed_q # [B, S, d_latent]

        #  2. compute new KV rows and append to cache
        new_kv_rows = self.ln(self.w_dkv(x)) # [B, S, emb_dim] * [emb_dim, d_latent] = [B, S, d_latent]
        if c_kv is None:
            c_kv = new_kv_rows
        else:
            c_kv = torch.cat([c_kv, new_kv_rows], dim=1) # [B, S_full, d_latent]
        assert c_kv is not None
        S_full = c_kv.size(1)

        #  3. Multiply q_c with updated KV cache
        c_attn_scores = q_c @ c_kv.transpose(1, 2) # [B, S, d_latent] * [B, d_latent, S_full] = [B, S, S_full]

        # Phase 2: RoPE
        #  4. Compute k_r, taking the cache into account
        if k_r is None:
            head_kr = self.w_kr(c_kv) # [B, S_full, d_latent] * [d_latent, head_width] = [B, S_full, head_width]
            shared_kr = self.rope_k(head_kr)
            k_r = shared_kr.repeat(1, 1, self.num_heads) # [B, S_full, emb_dim]
        else:
            new_head_kr = self.w_kr(new_kv_rows)
            new_shared_kr = self.rope_k(new_head_kr)
            new_k_r = new_shared_kr.repeat(1, 1, self.num_heads)
            k_r = torch.cat([k_r, new_k_r], dim=1)
        assert k_r is not None

        #  5. Compute q_r
        c_q = self.w_dq(x)
        full_qr = self.w_qr(c_q)
        q_r = self.rope_q(full_qr) # [B, S, emb_dim]

        #  6. Calculate the rotated attention scores
        ro_attn_scores = q_r @ k_r.transpose(1, 2) # [B, S, S_full]

        # Phase 3: Bringing it together
        #  7. Add both attention score vectors and add causal mask
        attn_scores = c_attn_scores + ro_attn_scores # [B, S, S_full]
        mask = self.mask[:S_full, :S_full]
        mask = mask[-S:, :] # not so sure about this
        attn_scores = attn_scores.masked_fill(mask == 1, float("-inf"))

        #  8. Compute the attention weights
        attn_weights = torch.softmax( # [B, S, S_full]
            attn_scores * (self.head_width ** -0.5),
            dim=-1
        )

        #  9. Compute v_c
        v_c = self.w_uv(c_kv) # [B, S, D]

        #  10. Compute the context vector
        context_vector = attn_weights @ v_c # [B, S, S_full] * [B, S, D] = [B, S, D]

        #  11. Compute the final attention logits
        logits = self.w_out(context_vector)
        if not self.training:
            logits = logits[:, -1:, :] # during inference we only need the last token

        return logits, c_kv, k_r


# In[20]:


class Expert(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden_dim == 0:
            hidden_dim = emb_dim * 4
        self.layer = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim, bias=False),
            gpt.GELU(),
            nn.Linear(hidden_dim, emb_dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


# In[ ]:


class NoisyTopKRouter(nn.Module):
    def __init__(self, n_experts: int, top_k: int, emb_dim: int, u: float = 0.001):
        """
        n_experts: the total number of experts to route between
        top_k: the number of experts to select for any given token
        emb_dim: the token embedding dimension to use for routing
        u: the load balancing correction coefficient
        """
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.routing = nn.Linear(emb_dim, n_experts, bias=False)
        self.noise_linear = nn.Linear(emb_dim, n_experts, bias=False)
        self.register_buffer("bias", torch.zeros(n_experts, requires_grad=False))
        self.u = u

    def update_bias(self, expert_counts: torch.Tensor, total_tokens: int):
        expected = total_tokens / self.n_experts
        load = expert_counts.float()
        bias_update = self.u * (expected - load) # TODO: this is technically wrong. instead of (expected - load), we want either -1 or 1, depending on the sign of (expected - load).
        self.bias += bias_update

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.routing(x)
        if self.training:
            noise_logits = self.noise_linear(x)
            noise = torch.randn_like(logits)*F.softplus(noise_logits)
            logits = logits + noise
        else:
            logits = logits

        logits = logits + self.bias

        top_k_experts, top_k_indices = logits.topk(self.top_k, dim=-1)

        if self.training:
            with torch.no_grad():
                mask = torch.zeros_like(logits)
                mask.scatter_(-1, top_k_indices, 1)
                expert_counts = mask.sum(dim=(0, 1))
                total_tokens = x.size(0) * x.size(1)
                self.update_bias(expert_counts, total_tokens)

        routing = torch.full_like(logits, float('-inf')).scatter_(-1, top_k_indices, top_k_experts)
        expert_selector_weight_matrix = F.softmax(routing, dim=-1)
        return (expert_selector_weight_matrix, top_k_indices)


# In[22]:


class FineGrainedMoE(nn.Module):
    def __init__(self, emb_dim: int, n_experts: int, expert_m: int, top_k: int, dropout: float = 0.1):
        super().__init__()
        if emb_dim % expert_m != 0:
            raise(AssertionError("expert_m must evenly divide emb_dim"))
        self.top_k = top_k
        self.router = NoisyTopKRouter(n_experts=n_experts, top_k=top_k, emb_dim=emb_dim)
        self.n_experts = n_experts
        self.shared_expert = Expert(emb_dim=emb_dim, dropout=dropout)
        self.experts = nn.ModuleList([
            Expert(emb_dim=emb_dim, hidden_dim=emb_dim//expert_m, dropout=dropout) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor): # x: [B, S, D]
        B, S, D = x.shape
        # gating_output: [B, top_k] (float)
        # indices: [B, top_k] (int)
        gating_output, indices = self.router(x)
        # final_output: [B, S, D]
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        x_flat = x.view(-1, D) # [B*S, D]
        indices_flat = indices.view(-1, self.top_k) # [B*S, top_k]
        gating_flat = gating_output.view(-1, self.n_experts) # [B*S, n_experts]
        final_output_flat = final_output.view(-1, D) # [B*S, D]

        # Get the shared expert output and add it to the result
        shared_expert_output = self.shared_expert(x_flat)
        final_output_flat += shared_expert_output

        for i, expert in enumerate(self.experts):
            # Find tokens routed to expert i
            expert_mask = (indices_flat == i)
            token_idx, expert_pos = torch.where(expert_mask)
            if token_idx.numel() == 0:
                continue

            expert_input = x_flat[token_idx]
            expert_output = expert(expert_input)

            gating_scores = gating_flat[token_idx, i]
            weighted_output = expert_output * gating_scores.unsqueeze(-1)

            final_output_flat[token_idx] += weighted_output

        return final_output_flat.view(B, S, D)


# In[23]:


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


# In[24]:


class DeepSeekTransformerBlock(nn.Module):
    """
    A single DeepSeek transformer block.
    """

    def __init__(self, cfg: DeepSeekConfigDict, rope_k: Optional[RoPE] = None, rope_q: Optional[RoPE] = None):
        super().__init__()
        self.clear() # to init the cache values
        self.layer_norm_1 = gpt.LayerNorm(cfg["emb_dim"])
        self.attention = MultiHeadLatentAttentionWithRoPE(
            emb_dim=cfg["emb_dim"],
            d_latent=cfg["latent_dim"],
            num_heads=cfg["n_heads"],
            context_length=cfg["context_length"],
            rope_k=rope_k,
            rope_q=rope_q,
        )
        self.drop_rate = cfg["drop_rate"]
        self.layer_norm_2 = gpt.LayerNorm(cfg["emb_dim"])
        expert_m = 4 # arbitrarily chosen
        num_experts = max(4, cfg["emb_dim"] // 256) * expert_m
        self.feedforward = FineGrainedMoE(emb_dim=cfg["emb_dim"], n_experts=num_experts, expert_m=expert_m, top_k=2*expert_m)
        self.dropout = nn.Dropout(self.drop_rate)

    def clear(self):
        self.c_kv = None
        self.k_r = None

    def forward_prefill(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.layer_norm_1(x)        
        x, _, _ = self.attention(x, c_kv=None, k_r=None)

        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm_2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + shortcut
        return x

    def forward_cache(self, x: torch.Tensor) -> torch.Tensor:
        start_idx = 0 if self.c_kv is None else -1
        x_in = x[:, start_idx:, :]
        shortcut = x[:, start_idx:, :]

        x = self.layer_norm_1(x_in)
        x, c_kv, k_r = self.attention(x, c_kv=self.c_kv, k_r=self.k_r)
        self.c_kv = c_kv.detach()
        self.k_r = k_r.detach()

        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm_2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + shortcut
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.forward_prefill(x)
        else:
            return self.forward_cache(x)


# In[ ]:


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, epsilon: float = 1e-8):
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return x / rms

class SimpleMTP(nn.Module):
    def __init__(self):
        pass


# In[25]:


class ClearableSequential(nn.Sequential):
    def __init__(self, *args: nn.Module):
        super().__init__(*args)
        self.args = args

    def clear(self):
        for m in self.args:
            m.clear()

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
        self.dropout = nn.Dropout(cfg["drop_rate"])

        # MLA allows us to share these objects for efficiency
        head_width = cfg["emb_dim"] // cfg["n_heads"]
        rope_k = RoPE(head_width, context_length=cfg["context_length"])
        rope_q = RoPE(cfg["emb_dim"], context_length=cfg["context_length"])

        self.transformer_blocks = ClearableSequential(
            *[DeepSeekTransformerBlock(cfg, rope_k=rope_k, rope_q=rope_q) for _ in range(cfg["n_layers"])]
        )
        self.layer_norm = gpt.LayerNorm(cfg["emb_dim"])
        self.output = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def clear(self):
        self.transformer_blocks.clear()

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass: input indices to logits."""
        batch_size, sequence_length = in_idx.shape
        x = self.token_embedding(in_idx)
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        logits = self.output(x)
        return logits

    def device(self) -> torch.device:
        return next(self.parameters()).device


# In[26]:


import tiktoken

def generate_text_simple(model: DeepSeekModel, encoded_input, max_new_tokens, context_size, device=gpt.get_device()):
    """
    A helper function used by smoke_test. It's easier to pass the prompt to smoke_test, rather than call this directly.
    """
    model.eval()
    model.clear()
    encoded_input.to(device)

    initial_input = encoded_input[:, -context_size:]
    with torch.no_grad():
        _ = model(initial_input)

    current_ids = initial_input
    for _ in range(max_new_tokens):
        last_token = current_ids[:, -1:]
        with torch.no_grad():
            logits = model(last_token)
        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probabilities, dim=-1, keepdim=True)
        current_ids = torch.cat((current_ids, idx_next), dim=1)

    model.clear()
    return current_ids


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


# In[27]:


import urllib.request
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import training


# Download the text if it's not yet available, then return it as a string
def the_verdict() -> str:
    """Returns the text of the short story \"The Verdict\". Uses the local filesystem for caching."""
    file_path = Path("the-verdict.txt")
    if not file_path.exists():
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
            with open(file_path, "w") as f:
                f.write(text_data)
            return text_data
    with open(file_path, "r") as f:
        return f.read()

tokenizer = tiktoken.get_encoding("gpt2")

def text_training_loaders(
    text: str, cfg: training.TrainingConfig
) -> tuple[DataLoader, DataLoader]:
    """Turn the given text into two Dataloaders: one for training and one for validation."""
    split_idx = int(len(text) * cfg["train_percent"])

    # Use partials with the Dataset and Dataloader classes to declutter and enforce consistency
    custom_dataset = partial(
        training.GPTDatasetV1,
        tokenizer=tokenizer,
        max_length=cfg["max_length"],
        stride=cfg["stride"],
    )
    custom_dataloader = partial(
        DataLoader, batch_size=4, shuffle=True, drop_last=True, num_workers=0
    )

    # raw text portions
    train_portion = text[:split_idx]
    validation_portion = text[split_idx:]

    # tokenized datasets
    train_dataset = custom_dataset(train_portion)
    validation_dataset = custom_dataset(validation_portion)

    # completed dataloaders
    train_loader = custom_dataloader(train_dataset)
    validation_loader = custom_dataloader(validation_dataset)
    return (train_loader, validation_loader)


def cross_entropy_loss_for_batch(
    model: DeepSeekModel,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    classification: bool = False,
) -> torch.Tensor:
    """Returns the model's loss for the given batch. The loss can be used to train the model.
    Supports classification and completion modes. Usually, you want completion (classification=False)."""
    device = model.device()
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    if classification:
        logits = logits[:, -1, :]
        return nn.functional.cross_entropy(logits, target_batch)
    else:
        return nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )  # TODO: explain why flatten


def calc_loss_loader(
    model: DeepSeekModel,
    data_loader: DataLoader,
    num_batches=None,
    classification: bool = False,
) -> float:
    """Calculates the model's total loss over a number of batches for the given
    Dataloader. This helper is used for validation only."""
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = cross_entropy_loss_for_batch(
                model, input_batch, target_batch, classification=classification
            )
            total_loss += loss.item()
            model.clear()
        else:
            break

    return total_loss / num_batches


def train_simple_text(model: DeepSeekModel, text: str, cfg: training.TrainingConfig) -> float:
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["peak_lr"], weight_decay=cfg["weight_decay"]
    )
    training_loader, validation_loader = text_training_loaders(text, cfg)
    last_validation_loss = 0.0

    for epoch in range(cfg["epochs"]):
        model.train()
        for input_batch, target_batch in training_loader:
            optimizer.zero_grad()
            loss = cross_entropy_loss_for_batch(
                model, input_batch=input_batch, target_batch=target_batch
            )
            loss.backward()
            optimizer.step()
            model.clear()

        # calculate the validation loss for this epoch
        model.eval()
        if len(validation_loader) == 0:
            raise ValueError("Ooops, no validation data")
        with torch.no_grad():
            validation_loss = calc_loss_loader(model, validation_loader)
            print(f"Validation loss for epoch {epoch}: {validation_loss}")
            last_validation_loss = validation_loss

    return last_validation_loss


def train_verdict(model: DeepSeekModel, epochs: int = 10) -> float:
    torch.manual_seed(123)
    text = the_verdict()
    verdict_training_config = training.new_training_config(
        train_percent=0.85, peak_lr=5e-4, max_length=256, epochs=epochs
    )
    return train_simple_text(model=model, text=text, cfg=verdict_training_config)


# In[32]:


model = DeepSeekModel(cfg=DeepSeekSmall)
model.to(gpt.get_device())

# v3: 2m2.5s
# v4: crash
# fine: 
train_verdict(model, epochs=20)


# In[33]:


def text_to_token_ids(
    text: str, tokenizer: tiktoken.Encoding, device: torch.device = gpt.get_device()
) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor.to(device)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def trained_example(model: DeepSeekModel, start_context, new_tokens = 10):
    torch.manual_seed(123)
    model.eval()
    model.clear()
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = gpt.generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=new_tokens,
        context_size=128,
    )

    print("Output text (trained):\n", token_ids_to_text(token_ids, tokenizer))

trained_example(model, "Jack thought", new_tokens=43)


# In[ ]:




