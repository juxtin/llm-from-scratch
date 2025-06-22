# LLM (From Scratch)

In this repo, I'm building a large language model comparable to GPT-2, using the following resources:
1. The book [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch), by [@rasbt](https://github.com/rasbt).
2. The YouTube playlist [Building LLMs from scratch](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu) by [Vizuara](https://www.youtube.com/@vizuara).
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.

# Notebooks:
- The basic GPT model implementation and my notes/explanations are in [gpt.ipynb](./gpt.ipynb).
  This file covers everything through chapter 5, section 3 of the book ("Decoding strategies to control randomness").
  It also has a bunch of original code for downloading a Project Gutenberg dataset, pre-processing it, and training on it.
- Sections 4 and 5 of chapter 5, "Loading and saving model weights in PyTorch" and "Loading pretrained weights
  from OpenAI," are in [openai.ipynb](./openai.ipynb).
- Chapter 6, "Fine-Tuning For Classification," is in [classifier.ipynb](./classifier.ipynb).

# Current Status

- Base GPT model is complete.
- Training with learning rate warmup, cosine decay, and gradient clipping is complete.
- The model can be loaded with pretrained weights from OpenAI, up to gpt2-xl.
- Fine-tuning for spam classification is in progress.

# To run

- `uv venv && source .venv/bin/activate && uv pip sync pyproject.toml`
- for CUDA support, `uv pip install --group cuda`
- Open in Jupyter Notebook