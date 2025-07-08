#!/usr/bin/env python
# Code generated from notebooks/classifier.ipynb by script/gen-py. DO NOT EDIT.

# coding: utf-8

# # Fine tuning for spam classification
# 
# This follows chapter 6 of the book [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch).

# In[1]:


import import_ipynb
import openai # type: ignore
import gpt # type: ignore
from gpt import get_device # type: ignore
import training # type: ignore
import pandas as pd
import urllib.request
import ssl
import zipfile
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tiktoken
import time
from functools import partial


# ## Download and preprocess the UCI spam data
# 
# The fine folks at the University of California at Irvine have provided a nice little data set for SMS spam.
# Let's download that and save it in a convenient CSV format.

# In[2]:


url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    ssl_context = ssl._create_unverified_context()

    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)    
    print(f"File downloaded and saved as {data_file_path}")


# The data set contains 4825 ham messages and only 747 spam messages. Since we want an equal number of both, we'll have to take 747 ham messages at random and discard the rest.

# In[3]:


def create_balanced_dataset(df):
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df


# Now we want to create the following splits:
# - 70% for training
# - 10% for validation
# - 20% for testing

# In[4]:


def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


# In[5]:


def save_csv():
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None) # type:ignore
    validation_df.to_csv("validation.csv", index=None) # type:ignore
    test_df.to_csv("test.csv", index=None) # type:ignore


# ## SpamDataset
# 
# This class:
# 1. Pre-tokenizes the texts from the dataset
# 2. Truncates any sequences that are longer than the maximum length (or the longest text, if no maximum is set).
# 3. Pads any sequences shorter than the max length.

# In[6]:


class SpamDataset(Dataset):
    def __init__(self, csv_file: Path, tokenizer: tiktoken.Encoding, max_length:int|None=None, pad_token_id:int=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            # truncate sequences that are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # pad the sequences
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.data.iloc[idx]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max([len(txt) for txt in self.encoded_texts])


# ## Creating the ClassifierGPT class
# 
# The ClassifierGPT class wraps and modifies a normal SimplifiedGPT model.
# Maybe it would be better to use a function that explicitly modifies its argument?
# 
# ### What's the difference between this and SimplifiedGPT?
# 
# Truthfully, not much: 
# - The final output layer has dimensions $\text{context\_length}\times\text{classifications}$, rather than $\text{context\_length}\times\text{vocabulary}$.
# - We discard the gradients for the inner layers, since those are adequately trained already.
# 

# In[7]:


class ClassifierGPT(nn.Module):
    """Wraps a SimplifiedGPT model and bases a classification model on it.
    Note that the model argument WILL BE MODIFIED."""
    def __init__(self, model: gpt.GPTModel, classifications:int):
        super().__init__()
        self.model = model
        cfg = model.cfg
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.output = nn.Linear(cfg["emb_dim"], classifications)
        for param in self.model.transformer_blocks[-1].parameters():
            param.requires_grad = True
        for param in self.model.layer_norm.parameters():
            param.requires_grad = True

    def device(self):
        return self.model.device()

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        return self.model(in_idx)


# In[8]:


def spam_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    save_csv()
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create the datasets from the CSV files
    train_dataset = SpamDataset(
        csv_file=Path("train.csv"),
        tokenizer=tokenizer,
    )
    max_length = train_dataset.max_length
    custom_dataset = partial(SpamDataset, tokenizer=tokenizer, max_length=max_length)
    val_dataset = custom_dataset(csv_file=Path("validation.csv"))
    test_dataset = custom_dataset(csv_file=Path("test.csv"))

    # Create the DataLoaders from the datasets
    num_workers = 0
    batch_size = 8
    custom_dataloader = partial(DataLoader, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    train_loader = custom_dataloader(dataset=train_dataset)
    val_loader = custom_dataloader(dataset=val_dataset)
    test_loader = custom_dataloader(dataset=test_dataset)

    return (train_loader, val_loader, test_loader)


# In[9]:


def calc_accuracy_loader(dataloader: DataLoader, model: ClassifierGPT, device: torch.device, num_batches:int|None=None) -> float:
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


# In[10]:


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: ClassifierGPT, device: torch.device) -> torch.Tensor:
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = nn.functional.cross_entropy(logits, target_batch)
    return loss


# In[11]:


def calc_loss_loader(dataloader: DataLoader, model: ClassifierGPT, device: torch.device, num_batches:int|None=None) -> float:
    total_loss = 0.
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# In[12]:


def evaluate_model(model: ClassifierGPT, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, eval_iter:int):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# In[13]:


def train_classifier_simple(model: ClassifierGPT, train_loader: DataLoader, val_loader: DataLoader, optimizer, device: torch.device, num_epochs:int, eval_freq:int, eval_iter:int):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end ="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# In[14]:


def classify(text: str, model: ClassifierGPT, tokenizer: tiktoken.Encoding, device: torch.device, max_length:int=0, pad_token_id:int=50256) -> str:
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.model.cfg['context_length']
    if max_length == 0:
        max_length = supported_context_length

    # truncate if too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    # pad if too short
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"


# In[15]:


class SpamExample(training.ExampleGenerator):
    def __init__(self, device: torch.device):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.msg = "YOU WIN A TRUCK! Text 666 and give us all your money right now to claim ur prize!"
        self.device = device

    def generate(self, model: gpt.GPTModel) -> str:
        classification = classify(self.msg, model, self.tokenizer, self.device)
        return f"{classification}: \"{self.msg}\""

def train_classifier(
        model: ClassifierGPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_epochs: int,
        eval_freq: int,
        eval_iter: int
    ):
    model.to(device)
    cfg = training.new_training_config(
        epochs=num_epochs,
        eval_freq=eval_freq,
        max_validation_batches=eval_iter, # I think?
        peak_lr=5e-5,
        weight_decay=0.1,
        classification=True,
    )
    training.train(
        model=model,
        optimizer=optimizer,
        training_loader=train_loader,
        validation_loader=val_loader,
        cfg=cfg,
        metrics=training.StdoutMetrics(print_interval=100),
        example_generator=SpamExample(device)
    )


# In[16]:


def model_accuracy(model: ClassifierGPT, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: torch.device):
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")


# In[18]:


def training_run(model: ClassifierGPT, train_loader: DataLoader, val_loader: DataLoader):
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    train_classifier(
        model, train_loader, val_loader, optimizer, get_device(), num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

tokenizer = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    gpt124m = openai.load_openai_model(openai.GPT_CONFIG_124M, "124M")
    clas = ClassifierGPT(gpt124m, 2).to(get_device())
    train_loader, val_loader, test_loader = spam_dataloaders()
    training_run(clas, train_loader, val_loader)

    # model_accuracy(clas, train_loader, val_loader, test_loader, get_device())
    max_length = 120 # the maximum length (in tokens) of the texts in the training loader
    sample = "hey dude, I promise this isn't spam."
    classify(sample, clas, tokenizer, get_device(), max_length=max_length)


# In[ ]:




