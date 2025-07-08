# train.py
import torch
from model import NanoGPT, device, batch_size, block_size, learning_rate, eval_iters
from tokenizer import encode, decode, vocab_size
import torch.nn.functional as F

# --- Carrega o dataset
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Inicializa o modelo
model = NanoGPT(vocab_size=vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Treinando com vocab_size = {vocab_size}")
print(f"Parâmetros totais: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

best_val_loss = float('inf')
max_iters = 1000
eval_interval = 100

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"[{iter}] Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), "nanogpt_energy.pth")
            print("✅ Novo melhor modelo salvo!")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Treinamento finalizado.")
