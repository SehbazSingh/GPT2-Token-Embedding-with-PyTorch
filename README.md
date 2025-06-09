# GPT2 Token Embedding with PyTorch

This repository demonstrates how to use the GPT-2 tokenizer from OpenAI's `tiktoken` library to tokenize text data, and then apply a simple PyTorch embedding layer on the tokenized input.

## 🧠 What It Does

- Reads text from a file (`the-verdict.txt`)
- Tokenizes the text using GPT-2 tokenizer (`tiktoken`)
- Converts token IDs into PyTorch tensors
- Applies a randomly initialized embedding layer on the token tensor
- Prints out the resulting embeddings

## 📁 File Structure

.<br>
├── the-verdict.txt # Input text file<br>
├── token-embedding.py # Python script


## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch tiktoken
```

### 2. Prepare Your Text File

Ensure you have a the-verdict.txt file in the same directory. You can replace this with any text you'd like to process.
### 3. Run the Script

```bash
python token-embedding.py
```
You will see output like:

```
tensor([[ 1.3966e+00, -9.9491e-01, -1.5822e-03],
        [-1.1659e+00,  1.3834e-01, -9.8013e-01],
        [ 1.0122e+00, -6.5152e-01,  8.6052e-02],
        ...,
        [-1.6160e+00, -1.0162e-01, -1.1510e+00],
        [-8.8438e-01, -6.7229e-02,  2.7670e+00],
        [-2.8280e-03,  2.8986e-01,  3.1343e-01]], grad_fn=<EmbeddingBackward0>)
```

🧩 Code Explanation

vocab_size = tokenizer.n_vocab  # Total size of GPT-2 vocabulary<br>
output_dim = 3                  # Size of embedding vector<br>
<br>
torch.manual_seed(123)         # Ensures reproducibility<br>
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)<br>
print(embedding_layer(input_ids))<br>

📌 Notes

    torch.manual_seed(123) sets the random seed for reproducibility.

    allowed_special={"<|endoftext|>"} allows GPT-2 special tokens to be used.

    The embedding dimension is set to 3 for simplicity; feel free to increase this in real projects.

📜 License

MIT License — free to use and modify.
