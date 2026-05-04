import torch
import torch.nn.functional as F
import sentencepiece as spm
from model.gpt import GPT

#config
device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_size = 8000
d_model = 384
n_heads = 6
n_layers = 6
block_size = 64

#load tokenizer
sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")

#load model
model = GPT(vocab_size, d_model, n_heads, n_layers, block_size)
model.load_state_dict(torch.load("model.pth"))
model = model.to(device)
model.eval()


def generate(start_text, max_tokens=50):
    tokens = sp.encode(start_text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_tokens):
        tokens_cond = tokens[:, -block_size:]

        logits, _ = model(tokens_cond)
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

    return sp.decode(tokens[0].tolist())

#test
print(generate("Once upon a time"))