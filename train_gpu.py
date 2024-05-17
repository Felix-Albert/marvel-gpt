import torch

from dataset import create_next_token_dataloader
from model_gpu import GPTModel
from tokenizer import TiktokenTokenizer

filename = "soccer_conversations.txt"
batch_size = 32
context_len = 64
n_heads = 6
n_layers = 3
emb_dim = n_heads * 16
drop_rate = 0.1

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open(filename, "r", encoding="utf-8") as f:
    data = f.read()

n_split = int(0.9 * len(data))
train_data = data[:n_split]
val_data = data[n_split:]
print(len(train_data), len(val_data))

tokenizer = TiktokenTokenizer()
vocab_size = tokenizer.vocab_size

train_token_ids = tokenizer.encode(train_data)
val_token_ids = tokenizer.encode(val_data)

print("Create dataloaders")
train_dataloader = create_next_token_dataloader(
    token_ids=train_token_ids,
    batch_size=batch_size,
    context_len=context_len,
    shuffle=True,
    drop_last=True
)

val_dataloader = create_next_token_dataloader(
    token_ids=val_token_ids,
    batch_size=batch_size,
    context_len=context_len,
    shuffle=True,
    drop_last=True
)

print("Create model")
model = GPTModel(
    vocab_size=vocab_size, emb_dim=emb_dim, drop_rate=drop_rate, n_layers=n_layers, context_len=context_len,
    n_heads=n_heads, qkv_bias=False
).to(device)  # Move model to the device (GPU if available)

print("Compute first loss")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train_loop(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    n_epochs=10
)