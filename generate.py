import torch
import torch.nn.functional as F
import sentencepiece as spm
from model.gpt import GPT

#config
device = "cuda" if torch.cuda.is_available() else "cpu"

d_model = 512
n_heads = 8
n_layers = 8
block_size = 128

#load tokenizer
sp = spm.SentencePieceProcessor(model_file="tokenizer/tiny.model")

vocab_size = sp.get_piece_size()
print("vocab_size:", vocab_size)

#load model
model = GPT(vocab_size, d_model, n_heads, n_layers, block_size)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model = model.to(device)
model.eval()


def generate(
    model,
    sp,
    start_text,
    max_tokens=50,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    device="cuda"
):
    model.eval()

    tokens = sp.encode(start_text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_tokens):
        tokens_cond = tokens[:, -block_size:]

        tokens_cond = torch.clamp(tokens_cond, 0, vocab_size - 1)

        logits, _ = model(tokens_cond)
        logits = logits[:, -1, :]

        # 🔥 Temperature
        logits = logits / temperature

        # 🔥 Repetition penalty (global)
        for token in tokens[0]:
            logits[0, token] /= 1.35

        # 🔥 Strong penalty for recent tokens
        recent_tokens = tokens[0][-10:]
        for token in recent_tokens:
            logits[0, token] /= 1.5

        # 🔥 Softmax
        probs = F.softmax(logits, dim=-1)

        # 🔥 Top-K
        top_k_values, top_k_indices = torch.topk(probs, top_k)
        top_k_probs = top_k_values / top_k_values.sum(dim=-1, keepdim=True)

        # 🔥 Sample
        next_token = torch.multinomial(top_k_probs, num_samples=1)
        next_token = torch.gather(top_k_indices, -1, next_token)

        # 🔥 Append
        tokens = torch.cat((tokens, next_token), dim=-1)

        # 🔥 Stop condition (NOW correct place)
        if next_token.item() == sp.eos_id():
            break

    return sp.decode(tokens[0].tolist())

#test
print(
    generate(
        model,
        sp,
        "Once upon a time",
        max_tokens=100,
        temperature=0.65,
        top_k=40,
        top_p=0.9,
        device=device
    )
)