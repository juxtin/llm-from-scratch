#!/usr/bin/env python
# Code generated from notebooks/chat.ipynb by script/gen-py. DO NOT EDIT.

# coding: utf-8

# # Fine tuning the model to make a chat bot
# 
# This is the big guacamole at the end of the rainbow. We'll be fine tuning one of the OpenAI models to be able to respond sort of like ChatGPT. I think there's an example of trying to do this on the foundation model in `openai.ipynb` without fine-tuning, and right now it _sucks_.

# In[ ]:


import import_ipynb
import openai # type:ignore
import gpt # type:ignore
from gpt import get_device # type: ignore
import torch
import urllib
import ssl
import os
import json
from pprint import pprint
from typing import TypedDict
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial
import textwrap
from datasets import load_dataset
import training # type: ignore

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


# ## Download the instruction training data
# 
# This is 1,100 instruction-response pairs (actually some have a third field called input) that were made specifically for the book.

# In[3]:


class InstructionExample(TypedDict):
    instruction: str  # A description of the task to be performed
    input: str        # Optional parameter for the task
    output: str       # The expected result of performing the task

def download_and_load_file(file_path: str, url: str) -> list[InstructionExample]:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url, context=ssl_context) as response: # type:ignore
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))
print("Example:")
pprint(data[1])


# ## Convert the examples to Stanford Alpaca format
# 
# The [format](https://github.com/tatsu-lab/stanford_alpaca) looks like this:
# 
# ```
# Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# 
# ### Instruction:
# {instruction}
# 
# ### Input:
# {input}
# 
# ### Response:
# ```
# 
# Or, if there's no input:
# 
# ```
# Below is an instruction that describes a task. Write a response that appropriately completes the request.
# 
# ### Instruction:
# {instruction}
# 
# ### Response:
# ```

# In[ ]:


def format_input(entry: InstructionExample, include_response:bool=True) -> str:
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    response_text = f"\n\n### Response:\n{entry['output']}" if include_response else ""

    return instruction_text + input_text + response_text

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]


# In[5]:


class InstructionDataset(Dataset):
    def __init__(self, data: list[InstructionExample], tokenizer: tiktoken.Encoding):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            full_text = format_input(entry)
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index) -> list[int]:
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# ## Custom collate function
# 
# Passing in a custom collate function lets us easily pad out shorter sequences in each batch to match the longest one.
# Initially, the padding token will be `<|endoftext|>`, but we'll eventually set it up so that there's only one EOT token
# and the padding will be done with `-100`.
# 
# The collate function is responsible for:
# 1. Finding the longest sequence in the batch
# 2. Padding and preparing inputs
# 3. Removing the extra EOT tokens
# 4. Converting the token list to a tensor and transferring it to the target device.
# 
# 
# ### We're not masking the instructions
# 
# We could use `-100` to mask out the instructions from each example. That would avoid rewarding the model for memorizing
# worthless bits like "Below is a task…", and some people think that's helpful. But it's controversial, and there's at least
# one paper, ["Instruction Tuning with Loss Over Instructions,"](https://arxiv.org/abs/2405.14394) that argues that it's
# better to train on the whole thing.
# 
# Maybe I'll try adding instruction masking later, but for now it's not recommended.

# In[6]:


def custom_collate_fn(
        batch: list[list[int]],
        pad_token_id: int=50256, # i.e., <|endoftext|>
        ignore_index: int=-100, # this is the default ignore index for torch.nn.CrossEntropyLoss
        allowed_max_length: int|None=None,
        device: str|torch.device="cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_max_length = max([len(item)+1 for item in batch])

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id # tensor([bool * max_length])
        indices = torch.nonzero(mask).squeeze() # type:ignore
        if indices.numel() > 1:
            # Note: we only do this -100 thing in the targets tensor
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)

    return inputs_tensor.to(device), targets_tensor.to(device)

customized_collate_fn = partial(custom_collate_fn, device=get_device(), allowed_max_length=1024)


# ## Create the Datasets and DataLoaders
# 
# As a reminder:
# - **Dataset**: a class that exposes `__getitem__` and `__len__`, so it's like
#   a list or a vector. It's not really specialized for anything in particular, it's
#   just a convenient way to wrap data from some source.
# - **DataLoader**: a class that encapsulates logic for ordering (shuffle or not),
#   associating inputs with targets, batching, parallelization, etc.

# In[7]:


num_workers = 0
batch_size = 8

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

custom_loader = partial(DataLoader, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=True, drop_last=True, num_workers=num_workers)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = custom_loader(train_dataset)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = custom_loader(val_dataset)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = custom_loader(test_dataset)


# In[8]:


class LlamaExampleGenerator(training.ExampleGenerator):
    def __init__(self, instruction: str, input: str = ""):
        self.prompt = format_input({
            'instruction': instruction,
            'input': input,
            'output': ""
        }, include_response=True)

    def generate(self, model: gpt.GPTModel) -> str:
        result = training.text_completion_topk(model, initial_context=self.prompt, max_new_tokens=128, context_size=512, topk=50, temperature=1.5)
        return result[len(self.prompt):].strip()

def train_model_on_small_example_set(model: gpt.GPTModel):
    example_prompt = format_input({
        'instruction': "Convert this sentence to passive voice.",
        'input': 'The chef cooked the meal.',
        'output': ''}, include_response=True)

    training_config = training.new_training_config(
        gradient_clipping = False,
        epochs = 2,
        peak_lr = 5e-5,
        initial_lr = 4e-6,
        eval_freq = 30,
    )

    optimizer = training.default_optimizer(model, training_config)

    training.train(
        model=model,
        optimizer=optimizer,
        training_loader=train_loader,
        validation_loader=val_loader,
        cfg=training_config,
        metrics=training.StdoutMetrics(print_interval=30),
        example_generator=LlamaExampleGenerator(instruction="Use this word in a sentence", input="fascinating")
    )

# gpt355m = openai.load_openai_model(openai.GPT_CONFIG_355M, "355M")
# train_model_on_small_example_set(gpt355m)
# model = gpt355m


# In[ ]:


# GPT_CONFIG_MEDIUM: gpt.GPTConfigDict = {**gpt.GPT_CONFIG_124M, "context_length": 512} # Equivalent to the project gutenberg config
# model = gpt.GPTModel(GPT_CONFIG_MEDIUM)

# training.load(model=model, optimizer=None, name="pg19_runpod_355m_396")
# train_model_on_small_example_set(model)


# In[12]:


# instruct_str can be used in cases where you need to keep the string around
def instruct_str(model: gpt.GPTModel, instruction: str, input='', temperature=0.8) -> str:
    prompt = format_input({
        'instruction': instruction,
        'input': input,
        'output': '',
    }, include_response=True)
    result = training.text_completion_topk(model, initial_context=prompt, max_new_tokens=1024, context_size=512, topk=50, temperature=temperature)
    return result[len(prompt):].strip()

# instruct is just used interactively, so it prints the result nicely
def instruct(trainer: gpt.GPTModel, instruction: str, input='', temperature=0.8):
    result = instruct_str(trainer, instruction, input, temperature)
    print(textwrap.fill(result, width=120))


# In[ ]:


# Should output: The healthier of the two foods is carrots.

# instruct(model, "Describe a pastoral scene")


# # Training on Alpaca
# 
# The [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset is about 52k rows, so almost 50x bigger than the one we just trained on. That should give 50x better results, right???

# In[15]:


alpaca = load_dataset("tatsu-lab/alpaca", split='train')


# In[16]:


def token_len(txt: str):
    tks = tokenizer.encode(txt)
    return len(tks)

alpaca: list[InstructionExample] = [x for x in alpaca if token_len(x['text']) <= 323] # type: ignore


# In[17]:


batch_size = 1

alpaca_train_portion = int(len(alpaca) * 0.95)
alpaca_val_portion = int(len(alpaca) * 0.002) # about 100 examples
alpaca_test_portion = len(alpaca) - alpaca_train_portion - alpaca_val_portion

alpaca_train_data = alpaca[:alpaca_train_portion]
alpaca_test_data = alpaca[alpaca_train_portion:alpaca_train_portion + alpaca_test_portion]
alpaca_val_data = alpaca[alpaca_train_portion + alpaca_test_portion:]

alpaca_train_dataset = InstructionDataset(alpaca_train_data, tokenizer)
alpaca_test_dataset = InstructionDataset(alpaca_test_data, tokenizer)
alpaca_val_dataset = InstructionDataset(alpaca_val_data, tokenizer)

print(f"Train: {len(alpaca_train_data)}")
print(f"Val: {len(alpaca_val_data)}")
print(f"Test: {len(alpaca_test_data)}")

alpaca_train = DataLoader(
    alpaca_train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

alpaca_test = DataLoader(
    alpaca_test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

alpaca_val = DataLoader(
    alpaca_val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)


# In[ ]:


def train_model_on_big_example_set(model: gpt.GPTModel):
    example_prompt = format_input({
        'instruction': "Convert this sentence to passive voice.",
        'input': 'The chef cooked the meal.',
        'output': ''}, include_response=True)

    training_config = training.new_training_config(
         gradient_clipping = False,
         epochs = 2,
         peak_lr = 5e-5,
         initial_lr = 4e-6,
         eval_freq = 200,
     )

    optimizer = training.default_optimizer(model, training_config)

    training.train(
        model=model,
        optimizer=optimizer,
        training_loader=alpaca_train,
        validation_loader=alpaca_val,
        cfg=training_config,
        metrics=training.StdoutMetrics(print_interval=30),
        example_generator=LlamaExampleGenerator(instruction="Use this word in a sentence", input="fascinating"),
    )

# gpt355m = openai.load_openai_model(openai.GPT_CONFIG_355M, "355M")
# train_model_on_big_example_set(gpt355m)
# training.save(model=model, optimizer=None, name="fine-tuned-355m-alpaca")

# train_model_on_big_example_set(model)


# In[ ]:


# llm = gpt.GPTModel(cfg=openai.GPT_CONFIG_355M, training_cfg=gpt.DEFAULT_TRAINING_CONFIG)
# llm.load("fine-tuned-355m-alpaca")


# In[ ]:


# instruct(llm, "Are you conscious?", temperature=0.8)


# In[ ]:


def save_example_responses(model: gpt.GPTModel, examples: list[InstructionExample], temperature:float=0.8, file_path:str="model_output.json"):
    results = []
    for ex in examples:
        model_response = instruct_str(model, ex['instruction'], ex['input'], temperature=temperature)
        results.append({
            'instruction': ex['instruction'],
            'input': ex['input'],
            'output': ex['output'],
            'model_output': model_response,
        })
    json_body = json.dumps(results, indent=4)
    with open(file_path, 'w') as f:
        f.write(json_body)


# In[ ]:


# save_example_responses(model=model, examples=test_data, file_path="small_training_output.json")


# In[ ]:




