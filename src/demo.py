import gpt
import training
import openai
import torch
import tiktoken
import sys

def load_model() -> gpt.GPTModel:
    model = gpt.GPTModel(openai.GPT_CONFIG_355M)
    training.load(model, optimizer=None, name="355M-alpaca", base_path="models", device=gpt.get_device())
    model.to(gpt.get_device())
    return model

END_OF_TEXT = 50256
tokenizer = tiktoken.get_encoding("gpt2")

ConfidenceToken = tuple[float, torch.Tensor]

def token_len(str) -> int:
    tokenized = tokenizer.encode(str)
    return len(tokenized)

def choose_from_topk(logits: torch.Tensor, topk: int, temperature: float):
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
    token_tensor = torch.multinomial(probabilities, num_samples=1)
    probability = probabilities.gather(dim=1, index=token_tensor)
    return (probability, token_tensor)

def generate_text_topk(model: gpt.GPTModel, token_ids: torch.Tensor, max_new_tokens: int, context_size: int, topk: int, temperature: float):
    probabilities = torch.ones(1, token_ids.size(1), device=gpt.get_device())
    for _ in range(max_new_tokens):
        idx_cond = token_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        (p, idx_next) = choose_from_topk(logits, topk, temperature)
        if idx_next.item() == END_OF_TEXT:
            break
        token_ids = torch.cat((token_ids, idx_next), dim=1)
        probabilities = torch.cat((probabilities, p), dim=1)
    return (probabilities, token_ids)

def text_completion_topk(model, initial_context: str, max_new_tokens:int=10, context_size:int=256, topk:int=50, temperature:float=1.5) -> list[tuple[float, str]]:
    device = model.device()
    encoded = tokenizer.encode(initial_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    model.eval()
    (probabilities, token_ids) = generate_text_topk(
        model,
        encoded_tensor,
        context_size=context_size,
        max_new_tokens=max_new_tokens,
        topk=topk,
        temperature=temperature,
    )
    token_probs = probabilities.tolist()
    decoded_tokens = [tokenizer.decode([tok]) for tok in token_ids.squeeze(0).tolist()]
    annotated_tokens = list(zip(token_probs[0], decoded_tokens))
    return annotated_tokens

RESET      = "\033[0m"
RED        = "\033[31m"
BOLD_RED   = "\033[1;31m"
GREEN      = "\033[32m"
BOLD_GREEN = "\033[1;32m"
BLACK      = "\033[30m"  # usually the default terminal color

def print_annotated(pairs: list[tuple[float, str]]):
    for p, tok in pairs:
        if p < 0.45:
            # low prob → red; very low → bold red
            color = BOLD_RED if p < 0.20 else RED
        elif p > 0.55:
            # high prob → green; very high → bold green
            color = BOLD_GREEN if p > 0.80 else GREEN
        else:
            # mid-range → default/black
            color = BLACK
        print(f"{color}{tok}{RESET}", end="")
    print()  # newline at end

def ask_model(model: gpt.GPTModel, instruction: str) -> list[tuple[float, str]]:
    prompt = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n"
        f"{instruction}\n\n"
        f"### Response:\n"
    )
    pairs = text_completion_topk(
        model,
        initial_context=prompt,
        max_new_tokens=1024,
        context_size=1024,
        topk=50,
        temperature=1.5,
    )
    prompt_len = token_len(prompt)
    return pairs[prompt_len:]

def intro():
    print("Tokens in the response are color-coded to represent their probability:")
    space = (0.0, " ")
    print_annotated([
        (0.0, "terrible"),
        space,
        (0.44, "low"),
        space,
        (0.5, "medium"),
        space,
        (0.56, "good"),
        space,
        (1.0, "great")
    ])

def chat_loop(model: gpt.GPTModel):
    instruction = input(">> ").strip()
    pairs = ask_model(model, instruction)
    print_annotated(pairs)
    chat_loop(model)

def main():
    print("Loading model...", end=""); sys.stdout.flush()
    model = load_model()
    print("done.")
    intro()
    chat_loop(model)

if __name__ == "__main__":
    main()