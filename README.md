# LLM (From Scratch)

In this repo, I'm building a large language model comparable to GPT-2, using the following resources:
1. The book [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch), by [@rasbt](https://github.com/rasbt).
2. The YouTube playlist [Building LLMs from scratch](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu) by [Vizuara](https://www.youtube.com/@vizuara).
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.

The code and my notes/explanations are in [gpt.ipynb](./gpt.ipynb).

# Current Status

This now includes all the code to train a model on Project Gutenberg and try out some completions.
The results aren't great, but they're interesting.

# To run

- `uv venv && source .venv/bin/activate && uv pip sync pyproject.toml`
- for CUDA support, `uv pip install --group cuda`
- Open in Jupyter Notebook