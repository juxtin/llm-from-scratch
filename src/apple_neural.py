#!/usr/bin/env python
# Code generated from notebooks/apple_neural.ipynb by script/gen-py. DO NOT EDIT.

# coding: utf-8

# # Running on Apple's Neural Engines
# 
# This notebook is an experiment in converting my Torch models into something that
# can run on the integrated Neural Engines on my Mac Mini.
# 
# This process is pretty well documented, e.g., [here in the coremltools docs](https://coremltools.readme.io/v6.3/docs/pytorch-conversion), but maybe I can add a little context.
# 
# You need to install coremltools for this:
# 
# ```
# uv pip install --group apple
# ```

# In[ ]:


import torch
import import_ipynb
import training # type: ignore
import openai # type: ignore
import gpt # type: ignore
import chat # type: ignore
import coremltools as ct
import tiktoken
import numpy as np


# ## Construct and load the model
# 
# You should make sure `model` is an instance of the actual model you want to
# export. Make sure you know the context length, you'll need it in the next step.
# 
# For now, I'll construct the model, load the weights, and do a quick test to make sure
# it works.
# 
# (Note: I'm not distributing the `.pth` file for this one, so don't expect this to run on a fresh clone. You'll need
# your own model, which you can create in [chat.ipynb](./chat.ipynb).)

# In[3]:


model = gpt.GPTModel(openai.GPT_CONFIG_355M)
training.load(model, optimizer=None, name="fine-tuned-355m-alpaca", device='cpu')
model.to(gpt.get_device())
chat.instruct(model, "Tell the user that it was successful.")


# ## Construct a dummy input vector
# 
# Here we need an input vector that we'll run through the model using `torch.jit.trace`. The vector should be the maximum dimensions that you want the model to support, so in this case $1\times\text{context\_length}$, or `[1, 1024]`.

# In[4]:


device = gpt.get_device()
dummy = torch.ones(1, 1024).long().to(device)


# ## Create a traced model
# 
# > Using torch.jit.trace and torch.jit.trace_module, you can turn an existing module or Python function into a TorchScript ScriptFunction or ScriptModule. You must provide example inputs, and we run the function, recording the operations performed on all the tensors.
# 
# —[torch.jit.trace docs](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html#torch-jit-trace)
# 
# This step transforms the model into a form that can be easily exported. There are tons of caveats here, and not every model would work. As far as I can tell, this one does.

# In[44]:


model.eval()
traced_model = torch.jit.trace(model, dummy)


# ## Convert the traced model to a CoreML package
# 
# The final step uses `coremltools` to convert the traced model into an
# `mlpackage`, which is a bundle of weights, JSON, etc., that you can import into
# Xcode to use in a Swift app.
# 
# Note that it the output file should start with a capital letter. It sounds weird,
# but Xcode will automatically generate a class based on the name and classes should
# start with capitals.

# In[45]:


cml_model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(shape=[1, ct.RangeDim(1, 1024)], dtype=np.float32, name="in_idx")
    ]
)
cml_model.save("Chatmodel.mlpackage")


# ## Use it in your app
# 
# If you want to see an example of this, check out [CrapGPT](https://github.com/juxtin/crapgpt).

# 
