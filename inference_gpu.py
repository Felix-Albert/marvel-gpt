import torch

from model_gpu import GPTModel
from tokenizer import TiktokenTokenizer


batch_size = 32
context_len = 64
n_heads = 6
n_layers = 3
emb_dim = n_heads * 16
drop_rate = 0.1

tokenizer = TiktokenTokenizer()
vocab_size = tokenizer.vocab_size

model = GPTModel(
    vocab_size=vocab_size, emb_dim=emb_dim, drop_rate=drop_rate, n_layers=n_layers, context_len=context_len,
    n_heads=n_heads, qkv_bias=False
)

token_ids = tokenizer.encode("Real Madrid is")
model.load_state_dict(torch.load("models/model_9.pth"))
model.eval()
generated_ids = model.generate_token_ids(torch.tensor(token_ids), max_new_tokens=50, context_len=context_len)

print(tokenizer.decode(generated_ids.tolist()))