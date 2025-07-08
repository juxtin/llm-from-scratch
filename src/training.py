#!/usr/bin/env python
# Code generated from notebooks/training.ipynb by script/gen-py. DO NOT EDIT.

# coding: utf-8

# # Training a smaller GPT-2
# 
# Having created the GPT model in [gpt.ipynb](./gpt.ipynb), it's time to try training it.

# In[ ]:


import import_ipynb
import gpt # type: ignore
from torch.utils.data import Dataset, DataLoader
import math
import mlflow
import os
import urllib.request
import tiktoken
import torch
import torch.optim as optim
import torch.nn as nn
import torch.cuda as cuda
import torch.backends
from pathlib import Path
from functools import partial
from typing import TypedDict, Optional
from collections.abc import Callable
from abc import ABC, abstractmethod

tokenizer = tiktoken.get_encoding("gpt2")

def get_device() -> torch.device:
    if cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps:0")
    else:
        return torch.device("cpu")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ## The mini config
# 
# While we're just setting things up and experimenting, it's better to set a lower context size. If we kept it at 1024, training would take up a lot more memory and a lot more time, slowing down development.
# 
# `GPT_CONFIG_MINI` sets a smaller context size of 256, which is enough to see things start to work.

# In[2]:


GPT_CONFIG_MINI: gpt.GPTConfigDict = {**gpt.GPT_CONFIG_124M, "context_length": 256}


# ## Convenience Functions and Example
# 
# We're adding a few functions to make it easier to interact with the model:
# - `text_to_token_ids`: represents the very first stage in running the model.
# - `token_ids_to_text`: represents the very last stage in running the model.
# 
# Also, another quick example of how they work.
# 
# Some things that are new:
# - `simplified_model.eval();` this disables the training functionality of the model, letting us run inference much faster. It will skip dropout and discard gradients. The semicolon just discards the return value, which seems pointless in this example.
# - `squeeze` and `unsqueeze`: These add or remove (respectively) an empty dimension to the outside of a tensor.
#   - squeeze: `[[x]]` -> `[x]` (usually to remove the batch dimension when batch_size=1)
#   - unsqueeze: `[x]` => `[[x]]` (to add a batch dimension where batch_size=1)

# In[3]:


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding, device:torch.device=get_device()) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor.to(device)

def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

# example:
def untrained_example(start_context:str="Every effort moves you"):
    torch.manual_seed(123)
    model = gpt.GPTModel(GPT_CONFIG_MINI)
    model.to(get_device())
    model.eval(); # disables dropout and gradients
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = gpt.generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer).to(get_device()),
        max_new_tokens=10,
        context_size=GPT_CONFIG_MINI["context_length"],
    )

    print("Output text (untrained):\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == "__main__":
    untrained_example()


# ## Dataset
# 
# `Dataset` is a class that resembles an iterator, but specialized for passing to
# a Dataloader. Like an iterator, it exposes `__len__` and `__getitem__` methods.
# 
# The specialization is that it returns input-target _pairs_ rather than single
# items. This is what makes it specifically useful for training.
# 
# In this case, it also encapsulates the tokenization step and implements the
# length/stride logic, but that's not always the case.

# In[4]:


class GPTDatasetV1(Dataset):
    def __init__(self, text: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            start = i
            end = start + max_length
            input_chunk = token_ids[start:end]
            target_chunk = token_ids[start+1:end+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


# ## Minimal training example
# 
# Initially I had written a big wrapper class that enclosed a `SimplifiedGPT` instance and encapsulated a ton of training logic.
# 
# That approach had some advantages and disadvantages, but in the end I think it was a mistake. It resulted in highly complected code
# with more and more configuration and more and more conditionals, although it did save some amount of passing arguments around.
# 
# Instead, I think it's preferable to mainly focus on functions that operate on the `GPTModel` class and/or its output logits.
# 
# To start, I'm going to focus on the bare minimum to train a model on [The Verdict](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt). That means I need to be able to do the following:
# 
# 1. download the text of The Verdict and save it locally.
# 2. split text into training and validation portions, then return a `training_loader` and `validation_loader`.
# 3. measure a model's loss given a batch (i.e., a set of input-output pairs).
# 4. implement the minimal training flow using the above and a crucial call to `loss.backward()`.

# ### Training config
# 
# Training involves a lot of parameters, and they need to be referenced by a few
# different functions. At the same time, most of them can have sensible defaults.
# 
# Rather than adding the parameters one by one to every function that might need them,
# I'm going to create a dataclass that we can pass around.

# In[ ]:


class TrainingConfig(TypedDict):
    train_percent: float                   # If the training corpus is text, the percentage to use for training. The rest is used for validation.
    max_length: int                        # the maximum length of a given training batch
    stride: int                            # the distance between starting points of training batches
    epochs: int                            # the number of times to train on one set of data
    initial_lr: float                      # the initial learning rate used by the optimizer
    peak_lr: float                         # the highest learning rate to be used by the optimizer
    weight_decay: float                    # the weight decay used by the optimizer
    gradient_clipping: bool                # whether to force gradient norms down to after every step 1.0
    temperature: float                     # the temperature for token generation
    topk: int                              # the number of logits to select for top-k token generation
    eval_freq: int                         # evaluate the model every [this number] of steps
    max_validation_batches: Optional[int]  # the maximum number of batches to use for validation
    classification: bool                   # allows the model to run in classification mode. If false, assumes completion mode.

def new_training_config(
        train_percent:float=0.9,
        max_length:int=1024,
        stride:Optional[int]=None,
        epochs:int=1,
        initial_lr:float=0.0001,
        peak_lr:float=0.01,
        weight_decay:float=0.1,
        gradient_clipping:bool=False,
        temperature:float=0.8,
        topk:int=50,
        eval_freq:int=0,
        max_validation_batches:Optional[int]=None,
        classification:bool=False,
) -> TrainingConfig:
    if stride is None:
        stride = max_length // 2
    return TrainingConfig(
        train_percent=train_percent,
        max_length=max_length,
        stride=stride,
        epochs=epochs,
        initial_lr=initial_lr,
        peak_lr=peak_lr,
        weight_decay=weight_decay,
        gradient_clipping=gradient_clipping,
        temperature=temperature,
        topk=topk,
        eval_freq=eval_freq,
        max_validation_batches=max_validation_batches,
        classification=classification,
    )


# ### Training on "The Verdict"
# 
# Training on this short story is probably the last time we'll pass a text file in
# directly for training, so some of these functions will be short-lived.
# 
# The most important function here is `train_simple_text`, which illustrates in
# the most minimal possible way how the training flow works. Most of that boils
# down to this one little stanza:
# 
# ```python
#     for epoch in range(cfg['epochs']):
#         model.train()
#         for input_batch, target_batch in training_loader:
#             optimizer.zero_grad()
#             loss = cross_entropy_loss_for_batch(model, input_batch=input_batch, target_batch=target_batch)
#             loss.backward()
#             optimizer.step()
# ```
# 
# To break that down in excruciating detail:
# 1. each epoch gets an entire run through the data loader, so we have a top-level loop here.
# 2. `model.train()` puts the model into training mode, where it preserves gradients for back propagation.
# 3. then we loop throuhg input-target pairs from the training loader.
# 4. `optimizer.zero_grad()` clears the optimizer state so that we don't contaminate this batch with the results from a previous batch.
# 5. we calculate the loss next. It's common to think of the loss as a scalar score, but in this case it encapsulates the entire gradient.
# 6. hence, we can call `loss.backward()` to backpropagate and update the model's weights.
# 7. finally, `optimizer.step()` updates the optimizer's parameters for one step.
# 

# In[ ]:


# Download the text if it's not yet available, then return it as a string
def the_verdict() -> str:
    """Returns the text of the short story \"The Verdict\". Uses the local filesystem for caching."""
    file_path = Path("the-verdict.txt")
    if not file_path.exists():
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
            with open(file_path, "w") as f:
                f.write(text_data)
            return text_data
    with open(file_path, "r") as f:
        return f.read()

def text_training_loaders(text: str, cfg: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    """Turn the given text into two Dataloaders: one for training and one for validation."""
    split_idx = int(len(text) * cfg['train_percent'])

    # Use partials with the Dataset and Dataloader classes to declutter and enforce consistency
    custom_dataset = partial(GPTDatasetV1, tokenizer=tokenizer, max_length=cfg['max_length'], stride=cfg['stride'])
    custom_dataloader = partial(DataLoader, batch_size=4, shuffle=True, drop_last=True, num_workers=0)

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

def cross_entropy_loss_for_batch(model: gpt.GPTModel, input_batch: torch.Tensor, target_batch: torch.Tensor, classification: bool = False) -> torch.Tensor:
    """Returns the model's loss for the given batch. The loss can be used to train the model.
    Supports classification and completion modes. Usually, you want completion (classification=False)."""
    device = model.device()
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    if classification:
        logits = logits[:, -1, :]
        return nn.functional.cross_entropy(logits, target_batch)
    else:
        return nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten()) # TODO: explain why flatten

def calc_loss_loader(model: gpt.GPTModel, data_loader: DataLoader, num_batches=None, classification: bool = False) -> float:
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
            loss = cross_entropy_loss_for_batch(model, input_batch, target_batch, classification=classification)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

def train_simple_text(model: gpt.GPTModel, text: str, cfg: TrainingConfig) -> float:
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg['peak_lr'], weight_decay=cfg['weight_decay']
    )
    training_loader, validation_loader = text_training_loaders(text, cfg)

    for epoch in range(cfg['epochs']):
        model.train()
        for input_batch, target_batch in training_loader:
            optimizer.zero_grad()
            loss = cross_entropy_loss_for_batch(model, input_batch=input_batch, target_batch=target_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    if len(validation_loader) == 0:
        raise ValueError("Ooops, no validation data")
    with torch.no_grad():
        validation_loss = calc_loss_loader(model, validation_loader)
        return validation_loss

def train_verdict(model: gpt.GPTModel) -> float:
    torch.manual_seed(123)
    text = the_verdict()
    verdict_training_config = new_training_config(train_percent=0.85, peak_lr=5e-4, max_length=256, epochs=10)
    return train_simple_text(model=model, text=text, cfg=verdict_training_config)


# ### Test out the training loop
# 
# Below you'll see the untrained output, the final training loss, and the trained output.
# 
# The trained output looks _much_ better, but there is a little bit of a trick
# here. Training 10 epochs on such a small and homogenous dataset means that we're
# _massively_ overfitting. The model is basically just memorizing phrases from the text
# and spitting them out.

# In[7]:


def trained_example(model: gpt.GPTModel, start_context):
    torch.manual_seed(123)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = gpt.generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_MINI["context_length"],
    )

    print("Output text (trained):\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == "__main__":
    start_context = "He never"
    untrained_example(start_context=start_context)
    model = gpt.GPTModel(GPT_CONFIG_MINI)
    model.to(get_device())
    val_loss = train_verdict(model)
    print(f"\nValidation loss after training: {val_loss:.3f}\n")
    trained_example(model, start_context=start_context)


# # Better text generation
# 
# So far, we've been using `gpt.generate_text_simple`, which chooses the next token by just finding the logit with the highest score.
# 
# That's good enough for illustration early in the process, but it honestly sucks. It results in super boring and repetitive output.
# 
# Let's implement a more typical Top-K multinomial algorithm:
# 1. Limit our choices to the top k logits, where k is a constant of our choosing. This prevents extremely unlikely tokens from _ever_ being generated.
# 2. Divide the logit probabilities by `temperature`. A low temperature emphasizes logits with higher probabilities, a higher temperature creates chaos by emphasizing lower probabilities.
# 3. Randomly choose from the remaining logits, weighted by their respective probabilities. Logits with higher probabilities will be chosen more often, but even unlikely ones will be chosen sometimes.

# In[8]:


END_OF_TEXT = 50256

def choose_from_topk(logits: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    top_logits, top_pos = torch.topk(logits, topk)
    filtered = torch.full_like(
        logits, -torch.inf
    )
    filtered.scatter_(dim=1, index=top_pos, src=top_logits) #huh?
    scaled = filtered / temperature
    probabilities = torch.softmax(scaled, dim=-1) # note: might have trouble with device
    if torch.any(torch.isnan(probabilities)) or torch.any(probabilities < 0):
        print("Bad probabilities:", probabilities)
        print("Logits:", logits)
        raise ValueError("NaNs or invalid values in probabilities")
    return torch.multinomial(probabilities, num_samples=1)

def generate_text_topk(model: gpt.GPTModel, token_ids: torch.Tensor, max_new_tokens: int, context_size: int, topk: int, temperature: float):
    for _ in range(max_new_tokens):
        idx_cond = token_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = choose_from_topk(logits, topk, temperature)
        if idx_next.item() == END_OF_TEXT:
            break
        token_ids = torch.cat((token_ids, idx_next), dim=1)
    return token_ids

def text_completion_topk(model, initial_context: str, max_new_tokens:int=10, context_size:int=256, topk:int=50, temperature:float=1.5):
    device = model.device()
    encoded = tokenizer.encode(initial_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    model.eval()
    out = generate_text_topk(
        model,
        encoded_tensor,
        context_size=context_size,
        max_new_tokens=max_new_tokens,
        topk=topk,
        temperature=temperature,
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    return decoded_text


# In[9]:


if __name__ == "__main__":
    start_context = "He never"
    model = gpt.GPTModel(GPT_CONFIG_MINI)
    model.to(get_device())
    train_verdict(model)
    print(text_completion_topk(model, start_context, temperature=0.4))


# # Saving and loading model state
# 
# 

# In[ ]:


def save(model: gpt.GPTModel, optimizer: Optional[optim.Optimizer], name: str, overwrite:bool=False, base_path: str = "."):
    if len(name) == 0:
        raise ValueError("name can't be empty")
    path = Path(f"{base_path}/{name}.pth")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists and overwrite is set to False")

    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state,
    }, path)

def load(model: gpt.GPTModel, optimizer: Optional[optim.Optimizer], name: str, base_path: str = ".", device: Optional[torch.device] = None):
    if len(name) == 0:
        raise ValueError("name can't be empty")
    path = Path(f"{base_path}/{name}.pth")
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    if optimizer_state_dict is not None and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])




# ## Default optimizer
# 
# An unfortunate aspect of the current architecture is that callers have to manage
# their own optimizers to some extent.
# 
# This helper takes on a little bit of that burden.

# In[20]:


def default_optimizer(model, cfg: TrainingConfig):
    return optim.AdamW(
        model.parameters(), lr=cfg['peak_lr'], weight_decay=cfg['weight_decay']
    )


# ## A more mature training loop
# 
# A complete training function has to represent a set of somewhat orthogonal concerns:
# 
# 1. Configuring the learning rate
# 2. Reporting metrics
# 3. Generating samples
# 4. The training loop itself
# 
# The `train_simple_text` function attempts to focus as much as possible on 4, barely touching the other points.
# Now that we're moving on, that's going to have to change. The problem is that we need some amount of extensibility
# and I don't think a gigantic configuration dict and a mess of conditionals is the way to go.
# 
# What follows are my attempts to make 1-3 configurable while keeping the training function as light as reasonably possible
# and providing good defaults.

# ### 1. Configuring the learning rate
# 
# I've decided to go with an _optionally_ configurable learning rate. The default
# is the linear warmup with cosine decay, and I expect that's going to be what's
# used probably every time. But if I ever want to experiment with a different type of scheduling, I can do
# that easily. 
# 
# The extension is responsible for returning two things in a tuple:
# 1. A `LearningRateFunction`, which takes the step number and returns a learning
#    rate multiplier. Note: the function does _not_ return the new learning rate! It
#    scales the base learning rate.
# 2. The number of warmup steps. This used to be individually configurable, but now it's
#    owned entirely by the learning rate scheduling abstraction. This number is used in the
#    training loop to make sure we don't do gradient clipping during warmup.
# 
# The default implementation is returned by `cosine_decay_lr`. In order to set reasonable values for `warmup_steps`, etc,
# it needs to peek at the training loader. This is basically unavoidable, since the scheduler _only_ passes the step number
# and not anything else that would be useful, like the total number of steps or anything like that.

# In[ ]:


LearningRateFunction = Callable[[int], float]

def cosine_decay_lr(cfg: TrainingConfig, training_loader: DataLoader) -> tuple[LearningRateFunction, int]:
    """Returns a LambdaLR function that closes over the cfg and implements a basic linear warmup with cosine decay.
    The number of warmup steps is also returned."""
    total_steps = len(training_loader) * cfg['epochs']
    warmup_steps = int(0.1 * total_steps)
    warmup_steps = max(warmup_steps, 1)
    decay_steps = total_steps - warmup_steps
    decay_steps = max(decay_steps, 1)
    def lambda_lr(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / decay_steps
            progress = min(progress, 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))
    return (lambda_lr, warmup_steps)


# ### 2. Reporting metrics
# 
# I really like using MLflow for visualizing and tracking metrics, but it's not
# something that I want to require for every training run. It would be great
# to have other metrics implementations that can be swapped in at will.
# 
# So I'm going to create a `Metrics` abstract base class that will have the following implementations:
# 1. MLflow, as a basic passthrough.
# 2. stdout, the default since it's convenient and doesn't require a separate service.
# 3. pushover (at some point in the future), for when I have longer running jobs in the cloud.
# 
# The class below is modeled after the MLflow methods that I used before.

# In[12]:


class Metrics(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def start_run(self):
        return self

    def error(self, msg: str):
        pass

    def finish(self, msg: str):
        pass

    @abstractmethod
    def log_param(self, name: str, val):
        pass

    @abstractmethod
    def log_metric(self, name: str, val, step: int):
        pass

    @abstractmethod
    def log_example(self, name: str, contents: str, step: int):
        pass

class StdoutMetrics(Metrics):
    def __init__(self, print_interval: int=1):
        self.print_interval = print_interval

    def error(self, msg: str):
        print(f"Training finished with error: {msg}")

    def finish(self, msg: str):
        print(f"Training finished successfully: {msg}")

    def log_param(self, name: str, val):
        print(f"Parameter: \"{name}\"={val}")

    def log_metric(self, name: str, val, step: int):
        if step % self.print_interval == 0:
            print(f"[{step}] Metric: \"{name}\"={val}")

    def log_example(self, name: str, contents: str, step: int):
        # I'm going to assume we always want to log these, regardless of the step
        print(f"[{step}] Example ({name}): \"{contents}\"")

class MLflowMetrics(Metrics):
    def __init__(self, tracking_uri: str = "http://localhost:5000", artifact_dir: str = "examples"):
        mlflow.set_tracking_uri(tracking_uri)
        os.makedirs(artifact_dir, exist_ok=True)
        self.artifact_dir = artifact_dir

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.run.__exit__(exc_type, exc_val, exc_tb)

    def start_run(self):
        if hasattr(self, "run"):
            raise RuntimeError("MLflow run has already been started")
        self.run = mlflow.start_run()
        return self

    def log_param(self, name: str, val):
        mlflow.log_param(name, val)

    def log_metric(self, name: str, val, step: int):
        mlflow.log_metric(name, val, step = step)

    def log_example(self, name: str, contents: str, step: int):
        mlflow.log_text(contents, artifact_file=f"{self.artifact_dir}/{name}_{step}.txt")


# ### 3. Generating samples
# 
# My earlier training method took a `sample_prompt` parameter and used that to
# generate a completion to add to the logs. That works as long as your focused on
# text completion, but it's not really adequate in general. Just in this project,
# I've done three types of generation:
# 
# 1. Text completion, as I just described
# 2. Classification, where the output is lower dimensional and not based on tokens
# 3. Instruction responses, where it's best to strip the full instruction preamble and only show the generated output
# 
# I had written three different training methods for those three scenarios, and that always bugged me. Instead, I will
# write another abstract base class that can provide a way to satisfy those three use cases (and others), hopefully
# with minimal extra code.

# In[16]:


class ExampleGenerator(ABC):
    @abstractmethod
    def generate(self, model: gpt.GPTModel) -> str:
        pass

class SimpleCompletion(ExampleGenerator):
    def __init__(
            self,
            prompt: str = "It is good",
            tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2"),
            max_new_tokens: int = 10,
            context_size: int = 128,
            topk: int = 50,
            temperature: float = 0.8,
        ):
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.context_size = context_size
        self.topk = topk
        self.temperature = temperature

    def generate(self, model: gpt.GPTModel) -> str:
        return text_completion_topk(
            model,
            self.prompt,
            max_new_tokens=self.max_new_tokens,
            context_size=self.context_size,
            topk=self.topk,
            temperature=self.temperature
        )


# ### 4. The training loop itself
# 
# This section ties it all together. The core training code is just a few lines,
# so most of this is actually taken up by looping, metrics, and error
# handling.
# 
# For now, the periodic evaluation logic relies on some basic configuration. There's
# no option yet to do something like generating and logging an example completion,
# although I'll probably add that soon.

# In[ ]:


def train(
        model: gpt.GPTModel,
        optimizer: optim.Optimizer,
        training_loader: DataLoader,
        validation_loader: Optional[DataLoader],
        cfg: TrainingConfig,
        lr_schedule_fn: Optional[LearningRateFunction] = None,
        metrics: Metrics = StdoutMetrics(print_interval=20),
        example_generator: ExampleGenerator = SimpleCompletion(),
        ):
    warmup_steps = 1
    if lr_schedule_fn is None:
        lr_schedule_fn, warmup_steps = cosine_decay_lr(cfg, training_loader)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule_fn)

    tokens_seen = 0
    global_step = 0
    total_steps = cfg['epochs'] * len(training_loader)
    loss_val = 0.0

    try:
        with metrics.start_run():
            # Initial metrics (job-level params)
            metrics.log_param("epoch size", len(training_loader))
            metrics.log_param("epochs", cfg['epochs'])
            metrics.log_param("total training size", len(training_loader) * cfg['epochs'])
            metrics.log_param("validation size", len(validation_loader or []))
            metrics.log_param("gradient clipping", cfg['gradient_clipping'])

            for epoch in range(cfg['epochs']):
                clear_cache()
                for input_batch, target_batch in training_loader:
                    model.train()
                    optimizer.zero_grad()

                    # Actual training
                    loss = cross_entropy_loss_for_batch(model, input_batch=input_batch, target_batch=target_batch, classification=cfg["classification"])
                    loss_val = loss.item()
                    loss.backward()
                    tokens_seen += input_batch.numel()
                    global_step += 1

                    # Gradient clipping if enabled and not warming up
                    if cfg['gradient_clipping'] and global_step >= warmup_steps:
                            nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0,
                        )

                    optimizer.step()
                    scheduler.step()

                    # Per-batch metrics
                    metrics.log_metric("training loss", loss_val, step=global_step)
                    metrics.log_metric("learning rate", scheduler.get_last_lr()[0], step=global_step)
                    metrics.log_metric("tokens seen", tokens_seen, step=global_step)
                    metrics.log_metric("epoch", epoch, step=global_step)
                    metrics.log_metric("progress percent", (global_step / total_steps) * 100, step=global_step)

                    # Periodic evaluation
                    if cfg['eval_freq'] > 0 and global_step % cfg['eval_freq'] == 0: # validation_loader not required
                        clear_cache()
                        model.eval()
                        with torch.inference_mode():
                            example = example_generator.generate(model)
                            metrics.log_example("example", example, step=global_step)
                            if validation_loader is not None:
                                validation_loss = calc_loss_loader(model, validation_loader, num_batches=cfg['max_validation_batches'], classification=cfg['classification'])
                                metrics.log_metric("validation loss", validation_loss, step=global_step)
                        clear_cache()
            # Final metrics
            metrics.log_metric("tokens seen", tokens_seen, step=global_step)
            if validation_loader is not None:
                clear_cache()
                model.eval()
                with torch.inference_mode():
                    total_validation_loss = calc_loss_loader(model, validation_loader, classification=cfg['classification'])
                    metrics.finish(f"final validation loss {total_validation_loss:.3f}")
            else:
                metrics.finish(f"final training loss {loss_val:.3f}")
    # Outer catchall block
    except Exception as error:
        metrics.error(f"{error}")
        raise error


# In[19]:


if __name__ == "__main__":
    model = gpt.GPTModel(GPT_CONFIG_MINI)
    model.to(get_device())
    torch.manual_seed(123)
    text = the_verdict()
    verdict_training_config = new_training_config(train_percent=0.85, initial_lr=0.0001, peak_lr=0.001, weight_decay=0, max_length=256, epochs=10, eval_freq=20)
    training_loader, validation_loader = text_training_loaders(text, verdict_training_config)

    optimizer = optim.AdamW(
        model.parameters(), lr=verdict_training_config['peak_lr'], weight_decay=verdict_training_config['weight_decay']
    )

    train(model, optimizer, training_loader, validation_loader, verdict_training_config)

