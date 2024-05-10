import torch

from model import GPTModel
from tokenizer import TiktokenTokenizer

batch_size = 64

context_len = 40

n_heads = 8 

emb_dim = n_heads * 20

drop_rate = 0.1

n_layers = 2

tokenizer = TiktokenTokenizer()
vocab_size = tokenizer.vocab_size

model = GPTModel(
    vocab_size=vocab_size, emb_dim=emb_dim, drop_rate=drop_rate, n_layers=n_layers, context_len=context_len,
    n_heads=n_heads, qkv_bias=False
)

token_ids = tokenizer.encode("Tony Stark: I am")
model.load_state_dict(torch.load("model_8.pth"))
model.eval()
generated_ids = model.generate_token_ids(torch.tensor(token_ids), max_new_tokens=50, context_len=context_len)

print(tokenizer.decode(generated_ids.tolist()))
