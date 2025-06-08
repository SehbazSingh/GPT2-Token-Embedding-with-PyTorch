import tiktoken
import torch

tokenizer = tiktoken.get_encoding("gpt2")
with open("the-verdict.txt", "r") as f:
    text = f.read()

# text = input("Enter text to encode: ")
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
input_ids = torch.tensor(ids)
vocab_size = tokenizer.n_vocab
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer(input_ids))
