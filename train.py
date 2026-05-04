import torch
from model.gpt import GPT
from data.dataloader import get_batch
import os

#-----------
#CONFIG
#-----------

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 8
block_size = 64

vocab_size = 8000
d_model = 384
n_heads = 6
n_layers = 6

learning_rate = 1e-4
max_iters = 10000
eval_interval = 100

#-----------
#MODEL
#-----------

model = GPT(vocab_size, d_model, n_heads, n_layers, block_size)

if os.path.exists("model.pth"):
    print("Loading existing model...")
    model.load_state_dict(torch.load("model.pth"))

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

print("Training on:", device)

#-----------
#TRAINING LOOP
#-----------

for step in range(max_iters):

    x, y = get_batch(batch_size, block_size, device)

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")
        
        # save checkpoint
        torch.save(model.state_dict(), "model.pth")
        print("Model checkpoint saved")

#save the model
torch.save(model.state_dict(), "model.pth")
print("Model saved!")