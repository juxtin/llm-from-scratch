#!/usr/bin/env python
# Code generated from notebooks/gpt.ipynb by script/gen-py. DO NOT EDIT.

# coding: utf-8

# # LLM From Scratch
# 
# This is a notebook I'm using to re-create the GPT-2 style architecture from the book "Build a Large Language Model (From Scratch)."
# I'm trying to do as much as possible from memory, other than having some notes on what classes and methods to implement.
# 
# **Required classes:**
# 1. `LayerNorm`
# 2. `GELU`
# 3. `GPT_CONFIG_124M`
# 4. `FeedForward`
# 5. `MultiHeadAttention`
# 6. `TransformerBlock`
# 7. `GPTModel`

# In[40]:


import tiktoken
import torch
import torch.nn as nn


# ## 1. LayerNorm
# 
# This class is responsible for layer normalization, which takes place _multiple times_ in the GPT architecture.
# Its purpose is to keep gradient magnitudes within a certain range, to avoid the problems of vanishing gradients and exploding gradients.
# The concrete goal is to adjust the outputs to have a mean of zero and a variance of one.
# 
# To accomplish this, we need two values:
# - the mean: $\mu = \frac{(x_1 + x_2 + ... + x_n)}{n}$
# - the variance: $v = \frac{(x_1 + \mu)^2 + (x_2 + \mu)^2 + ... + (x_n + \mu)^2}{n} + \epsilon$
# 
# The normalized vector is then: $[\frac{(x_1 - µ)}{\sqrt{v}}, \frac{(x_2 - µ)}{\sqrt{v}}, ..., \frac{(x_n - µ)}{\sqrt{v}}]$
# 
# NOTE: we're dividing by both n and $\sqrt{v}$ and we need to make sure we never divide by zero. We know that n (the embedding dimension) will never be zero, but the variance could be. For that reason, we add a miniscule value epsilon to the variance.

# In[41]:


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.epsilon = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False) + self.epsilon
        norm = (x - mean) / torch.sqrt(variance)
        return self.scale * norm + self.shift


# ## 2. GELU
# 
# GELU, or Gaussian Error Linear Unit, is the activation function we'll be using. It's similar to RELU, but it's differentiable everywhere (even at zero, where RELU has a sharp corner discontinuity). GELU is also slightly negative between -2 and 0, rather than flatly zero like RELU. This provides a richer range of values for the network to train on.
# 
# Calculating the GELU for real would take us out of closed-form math, so we'll use a very close approximation here instead.

# In[42]:


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


# ## 3. GPT_CONFIG_124M
# The configuration paramters for our GPT-2 implementation. These come directly from the book.

# In[43]:


from typing import TypedDict

class GPTConfigDict(TypedDict):
    vocab_size: int        # the number of tokens in the vocabulary
    context_length: int    # the maximum number of token vectors to consider at once
    emb_dim: int           # the width of the token vectors
    n_heads: int           # the number of heads to use for multi-head attention
    n_layers: int          # the number of transformer layers to use
    drop_rate: float       # the dropout percentage rate
    qkv_bias: bool         # whether to use the bias setting for the KQV matrices.

GPT_CONFIG_124M: GPTConfigDict = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


# ## 4. FeedForward
# 
# The feed-forward network (or multi-layer perceptron) is the fundamental neural network used in the GPT model.
# It expands the number of outputs in a hidden layer before shrinking back down to the original size for the output.
# This allows the network to explore a richer space, while preserving the input and output dimensions to keep the overall architecture simple.
# 
# In this example, we'll expand the dimensions by a factor of 4 for the internal layer. I would normally say that should be configurable, but the book just has it fixed at 4. Anyway, that means that our 768 parameters will expand to 3,072, then shrink back down to 768 for output.
# 
# ### How many layers?
# 
# If you look at a diagram of a feed-forward network, you'll see three layers:
# 1. a left-most layer with n weights
# 2. a middle layer with n*4 weights (or some other factor)
# 3. a right-most layer with n weights again.
# 
# However, if you look at the implementation below, it kind of seems like there are two linear layers.
# Well, as you might guess, the middle layer is really the connection between the first and the second layers.
# The first layer has `dim_internal` outputs, and the second layer has `dim_internal` inputs. These represent overlapping,
# connected points—just as you might see in the diagram.
# 
# You could think about like this: each `nn.Linear` has two sides, and of the four total sides there are two that overlap in the center. Thus you get three layers!

# In[44]:


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfigDict): 
        super().__init__()
        expansion_factor = 4
        dim_external = cfg["emb_dim"]
        dim_internal = expansion_factor * dim_external
        self.layers = nn.Sequential(
            nn.Linear(dim_external, dim_internal),
            GELU(),
            nn.Linear(dim_internal, dim_external),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ## 5. MultiHeadAttention
# 
# This is the heart of what makes GPT different to earlier language models. The attention mechanism tweaks context vectors in response to earlier tokens in the sequence, shifting their "meaning" to become much richer and more specific than a single word could be.
# 
# ### Motivating Examples
# 
# For example, take the sentence "the cat sat on the mat because it was warm." The word "it" has one particular vector embedding in the vocabulary, which might relate loosely to concepts like "noun" and "non-human." That's not enough to capture the meaning of "it" in this sentence, where it most likely refers to "mat." Attention allows the system to change the vector for the "it" token to resemble the vector for "mat," clarifying its meaning in the context of the sentence.
# 
# That's about the simplest possible example, but in reality each token is pushed and pulled in much more subtle ways by many more tokens in the sequence, so that by the end it somehow represents the meaning of the entire sequence of text. Ultimately, the attention-modulated vector of the final token in the sequence is _the only input needed_ to predict the next token. That's pretty wild.
# 
# For a more contrived example of what this means, take another example sequence: "This gritty, romantic, windswept, ornate, melancholic city is none other than". The word "than" has nothing to do with any particular city or place, but by the time its vector is modulated by this long series of words preceding it, it will be something that appears close (in embedding space) to cities like Lisbon and Istanbul. Indeed, those are the two most likely predictions for the final word in the sequence from GPT-3.
# 
# ### Implementation
# 
# Multi-head attention was first described in "Attention is All You Need" (2017), in sections 3.2.1 (scaled dot-product attention) and 3.2.2 (extending to multiple heads). I'll be using that paper as a reference for the following two sections.
# 
# #### Scaled Dot-Product Attention
# 
# Each attention head is an instance of something called "scaled dot-product attention," which is given by:
# 
# $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
# 
# That is, the attention weights given matrices K, Q, and V are the result of applying softmax to the product of Q times K-transpose over the square root of the embedding size of K, all multiplied by V.
# 
# I'll try to break that down a bit more:
# - Q, K, and V are trainable matrix parameters with the same dimensions as the token embedding vectors. They are short for Query, Key, and Value.
#   - I think of the Query parameter as representing what a token is "looking for" to know if another token is worth attending to.
#   - To continue that metaphor, the Key parameter is what other tokens "look like" to the Query.
#   - The Value is the real identity of the tokens that are found, their deeper reality beneath the appearance presented by the Key.
#   - To sum up, a token's Query is used to examine every other token's Key to see if it's a good match. If it is, we use that token's Value in attention weight.
# - Multiplying Q by the transpose of K gives us the dot product of every Query row against every Key row. In other words, it tells us how aligned every Query is with every Key.
# - We scale that by the inverse square root of the Key dimensions to counteract a known issue with dot-product attention: "for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients." ("Attention is All You Need," p. 4). In other words, the dot product of two rows is going to tend to get larger the more columns you have, and these large values make it hard for training to adjust weights effectively. Scaling by the square root of the number of columns helps to solve this.
# - Applying softmax turns these scaled dot products into weights.
# - Multiplying by V translates the weights by Key into weights by Value.
# 
# Note: it's not described in detail in the paper, but there's an important step carried out here called masking. Essentially, we only want Queries to find Keys that _precede_ them in the sequence. We accomplish this by zeroing out values above the main diagonal. To make sure that these values are zero _after_ softmax, we first set them to minus-infinity.
# 
# #### Multi-Head Attention
# 
# In single-headed dot-product attention, Q, K, and V all have the same dimensions as the input and output embeddings. To use multiple heads, we divide the width of each parameter by the number of heads and concatenate them together. This results in the same overall dimensions, but with different sets of columns relating to different Value vectors:
# 
# $\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O$
# 
# $\text{ where } head_i = \text{Attention}(Q_iW_i^Q, K_iW_i^K, V_iW_i^V)$
# 
# $\text{ where } W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $ W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, $W_i^O \in \mathbb{R}^{hd_{model} \times d_{model}}$
# 

# In[45]:


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias: bool=False):
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

        # and the output projection, also trainable.
        self.w_out = nn.Linear(d_out, d_out)

        # and the dropout layer. not trainable, just drops random values
        # to zero with a probability determined by the dropout parameter
        self.dropout = nn.Dropout(dropout)

        # and the mask, which prevents each token from "seeing" later ones
        mask = torch.triu( # an upper triangular matrix
            torch.ones(context_length, context_length), # consisting of ones
            diagonal=1, # starting one row above the diagonal, leaving the diagonal itself as zeroes.
        )
        self.register_buffer("mask", mask) # register this tensor as non-trainable, but keep it on the same device
        self.mask: torch.Tensor # to make the type-checker happy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, d_in = x.shape
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

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
        attention_scores = attention_scores.masked_fill(mask == 1, float('-inf'))

        # and we construct the weights using softmax on the scaled final dimension
        attention_weights = torch.softmax(attention_scores / self.head_width**0.5, dim=-1)
        # and apply dropout
        attention_weights = self.dropout(attention_weights)

        #                                 [  0  ,     1    ,     2     ,     3     ]
        # attention_weights has the shape [batch, num_heads, num_tokens, num_tokens]
        # v_heads has the shape:          [batch, num_heads, num_tokens, head_width]
        # if we multiply them, we get:    [batch, num_heads, num_tokens, head_width]
        # but in the end, we want:        [batch, num_tokens, d_out]
        context = attention_weights @ v_heads # [batch, num_heads, num_tokens, head_width]

        # so we need to first transpose and get [batch, num_tokens, num_heads, head_width]
        context = context.transpose(1, 2)
        # and then concatenate the last two dimensions together to get d_out
        context = context.contiguous().view(batch, num_tokens, self.d_out)
        # and multiply by the output projection
        return self.w_out(context)


# ## 6. TransformerBlock
# 
# This version of the transformer block is loosely based on "Attention is All You Need" section 3, but includes _only_ the decoder stack. The encoder stack is omitted from the GPT architecture, and thus from the Build a Large Language Model (From Scratch) book.
# 
# The transformer block goes a little something like this:
# ```
# Tokenized Text -> LayerNorm 1 -> MultiHeadAttention -> Dropout -> (+) -> LayerNorm 2 -> FeedForward -> Dropout -> (+) -> Output
# ```
# 
# Where `(+)` represents a shortcut connection, where a previous state is added back in to reinforce weights that are getting very small.
# 
# As far as requirements:
# - I've already implemented the the LayerNorm, MultiHeadAttention, and FeedForward classes.
# - `nn.Dropout` is provided by PyTorch.
# - Shortcut connections just use ordinary variables and addition.
# 
# So we're all set to put these elements together below.

# In[46]:


class TransformerBlock(nn.Module):
    """
    A single GPT-2 transformer block.
    """
    def __init__(self, cfg: GPTConfigDict):
        super().__init__()
        self.layer_norm_1 = LayerNorm(cfg["emb_dim"])
        self.attention = MultiHeadAttention(
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


# ## 7. GPTModel
# 
# This is the big one, where everything comes together.
# The hard parts are all pretty much done, this is going to be just a bit more glue.
# 
# The flow here goes like:
# ```
# Tokenized Text -> Token Embedding Layer -> Positional Embedding Layer -> Dropout -> TransformerBlocks -> LayerNorm -> Output
# ```
# 
# Or, in detail:
# 1. **Tokenized Text**: the tokenizer is outside of this module; we'll get to that later.
# 2. **Token Embedding Layer**: this is a trainable `nn.Embedding` layer that starts out with random weights. It maps tokens to the embedding space.
# 3. **Positional Embedding Layer**: very similar to the Token Embedding Layer, but encodes positional information rather than "semantic" content.
# 4. **Dropout**: provided by `nn.Dropout` with a configurable drop rate.
# 5. **TransformerBlocks**: implemented above. We'll have a number of these set by config, and they run in serial.
# 6. **LayerNorm**: also implemented above. This keeps all values in the tensors in a range of [-1, 1], with a mean of 0.
# 7. **Output**: the outputs are called "logits," and they represent the likelihood that the following token will be the one with a given ID. In order to project these from the previous LayerNorm, we'll need the size to be $\text{emb\_dim} \times \text{vocab\_size}$
# 
# This model is just a _module_. In PyTorch, modules are the basic building blocks
# of neural networks. That is to say, they need some additional machinery to actually be
# useful.
# 
# Other than the smoke test below, this file is now complete. Next up is [training](./training.ipynb)!

# In[49]:


class GPTModel(nn.Module):
    """
    Minimal GPT module. It can return logits via the `forward` method, but it
    can't do training on its own. That said, we will be training this later
    using some functions defined in training.ipynb.
    """
    def __init__(self, cfg: GPTConfigDict):
        """Initialize model with config."""
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.positional_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.layer_norm = LayerNorm(cfg["emb_dim"])
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


# # Smoke Test
# 
# If everything above has worked, then we should be able to exactly replicate the results from the book as long as we use the same seed (123).
# 
# Use the `smoke_test` function below to get the predicted completion for a given prompt from the _untrained_ LLM.
# 
# Note: because the LLM is still untrained, the result will be total garbage.

# In[ ]:


def get_device() -> torch.device:
    if torch.cuda.is_available(): # type: ignore[attr-defined]
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # type: ignore[attr-defined]
        return torch.device("mps:0")
    else:
        return torch.device("cpu")

def generate_text_simple(model, idx, max_new_tokens, context_size, device=get_device()):
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
    model = GPTModel(GPT_CONFIG_124M)
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    model.eval()
    out = generate_text_simple(
        model,
        encoded_tensor,
        6,
        GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

if __name__ == "__main__":
    smoke_test("Hello, I am") # should output "Hello, I am Featureiman Byeswickattribute argue"

