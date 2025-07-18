#!/usr/bin/env python
# Code generated from notebooks/grading_with_ollama.ipynb by script/gen-py. DO NOT EDIT.

# coding: utf-8

# # Using Ollama to evaluate our instruction responses
# 
# The most practical method we have to really assess the quality of our instruction-following model is to use a bigger, better model to judge it.
# 
# I'm keeping this code in its own notebook so that I _don't_ have to import `torch` and everything else. I'm going to assume a `model_output.json` file was already generated by `chat.ipynb`.

# In[1]:


import json
from typing import TypedDict
import psutil
import urllib.request
from urllib.error import URLError
import tqdm


# In[2]:


# this is almost the same as the one defined in chat.ipynb, but it has model_output as well.
class InstructionExampleResponse(TypedDict):
    instruction: str  # A description of the task to be performed
    input: str  # Optional parameter for the task
    output: str  # The expected result of performing the task
    model_output: str  # The actual result of performing the task
    assessment: str | None  # The assessment provided by the judge LLM, if any
    score: int | None  # The score assigned by the judge LLM, if any


# In[8]:


def check_if_running(process_name: str):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


ollama_running = check_if_running("ollama")
if not ollama_running:
    raise RuntimeError("Ollama is not running. Launch ollama before proceeding.")

print("Ollama is running.")


# In[9]:


def query_model(
    prompt: str,
    model: str = "qwen3:4b",
    url: str = "http://localhost:11434/api/chat",
) -> str:
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 123, "temperature": 0, "num_ctx": 2048},
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            while True:
                line = response.readline().decode("utf-8")
                if not line:
                    break
                response_json = json.loads(line)
                response_data += response_json["message"]["content"]
    except URLError as error:
        print("Timed out on request:")
        print("---")
        print(prompt)
        print("---")

    return response_data


# In[10]:


def response_to_score(response: str) -> int:
    lines = response.split("\n")
    last_line = lines[-1].strip()
    # despite my careful prompting, the model will often reply with "Score: 20"
    last_word = last_line.split(" ")[-1]
    try:
        return int(last_word)
    except ValueError:
        print("Invalid response\n")
        print(response)
        return 0


def judge_example(example: InstructionExampleResponse) -> InstructionExampleResponse:
    preface = (
        "Below you'll find the output from a new LLM. Please "
        'evaluate the quality of the "model_output" compared to the reference '
        '"output". Explain the rationale for your judgment, then end with a new line '
        "containing only an integer score out of 100. The final line should contain only "
        "a single integer. Factually incorrect responses must be scored below 50, with higher scores indicating that the answer is close to the truth. \n"
        "Ungrammatical responses must be scored below 50, with lower scores indicating nonsense.\n"
        "Other than that, please judge based on how close the response is to being complete, accurate, and relevant to the prompt.\n"
        "If the model_output is functionally identical to the reference output, the score should be 100.\n"
        "Please note that the last line of your response must be a single integer with no other characters.\n"
        "Example:\n\n"
        "The model's response incorrectly identifies the capital of Poland as Portland, but it is otherwise grammatically correct.\n"
        "While Portland is a city, it is not in Poland and it is not a capital city.\n25\n\n"
        "Remember that your score will be parsed with the assumption that the final line contains only a single integer value.\n"
        "A valid score looks like:\n85\n\n"
        "An invalid score looks like:\n**100**\n"
        "or:\nScore: 100 out of 100\n"
        "or: \\box{100}\n\n"
        "Result to evaluate:\n"
    )
    prompt = preface + json.dumps(example)
    response = query_model(prompt)
    score = response_to_score(response)
    result = example.copy()
    result["assessment"] = response
    result["score"] = score
    return result


# In[11]:


with open("pg19_774M_responses.json", "r") as f:
    examples = json.load(f)

examples[1]
judge_example(examples[45])


# In[12]:


def score_examples(examples: list[InstructionExampleResponse], out_path: str):
    for example in tqdm.tqdm(examples, desc="Scoring examples"):
        result = judge_example(example)
        with open(out_path, "a") as f:
            f.write(json.dumps(result))
            f.write("\n")


# In[13]:


score_examples(examples, "graded_pg19_774M_responses.lsv")


# In[14]:


def average_score(tsv_file: str):
    with open(tsv_file, "r") as f:
        lines = f.readlines()
    results = [json.loads(x) for x in lines]
    scores = [x["score"] for x in results]
    avg = sum(scores) / len(scores)
    return avg


# In[15]:


def scroll_results(tsv_file: str):
    with open(tsv_file, "r") as f:
        lines = f.readlines()
    results: list[InstructionExampleResponse] = [json.loads(x) for x in lines]
    for result in results:
        print(f"Instruction: {result['instruction']}")
        print(f"Input: {result['input']}") if result["input"] else None
        print(f"Output: {result['model_output']}")
        print(f"Score: {result['score']}")
        print("-------------")


# In[ ]:


scroll_results("graded_small_training.lsv")


# In[33]:


scroll_results("graded_355m_alpaca.lsv")


# # pg19 774M Alpaca
# 
# ## Introduction to the experiment
# 
# In the sections above, you can see what happens when you fine-tune one of OpenAI's old pre-trained models on instruction-response pairs.
# The results are fairly ok, and I'm sure they would have been very impressive by 2017 standards as a proof of concept.
# 
# My assumption was that fine-tuning made it possible to make explicit some of the "knowledge" that was implicit in the OpenAI training data,
# which covers a wide range of sources and topics. But what would happen if we tried fine tuning a model that was trained only on public domain
# English books?
# 
# Well, I found out.
# 
# ## Procedure
# 
# The results below were a result of the following procedure:
# 1. Create a randomly-initialized model based on GPT 2 Large (774 million parameters)
# 2. Train the model on [Project Gutenberg](https://huggingface.co/datasets/deepmind/pg19) books until training levels out after a few hundred million tokens (see [project_gutenberg.ipynb](./project_gutenberg.ipynb)).
# 3. Fine-tune the model on the tiny dataset of instruction-response pairs from the book.
# 4. Fine-tune the model _again_ on the much larger set of 52,000 instruction-response pairs from [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca). See [chat.ipynb](./chat.ipynb).
# 5. Save the results of 110 example prompts and evaluate them using Qwen3.
# 
# ## Results
# 
# I would call the results below comically bad. There were only two that I would consider somewhat correct.
# For the first "correct" response, it gave an answer that does have a more positive connotation, but clearly
# isn't a normal way to complete the task:
# 
# ```
# Instruction: Rewrite the given sentence to describe the same thing in a positive way.
# Input: The food was not good.
# Output: The food was not perfect.
# ```
# 
# The second correct response went a little beyond what was asked, but it did technically also fulfill the request:
# 
# ```
# Instruction: Rewrite the sentence to use a negative adverb.
# Input: She always remembers to call.
# Output: She never forgot to call, no matter what.
# ```
# 
# The other 108 responses were generally bizarre, repetitive, irrelevant, nonsensical, etc.

# In[17]:


scroll_results("graded_pg19_774M_responses.lsv")


# In[ ]:




