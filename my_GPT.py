import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters.
block_size = 128 # What is the maximum context length for predictions?
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
# How many batches we use each time we evaluate
d_model = 128
n_head = 6 # This implied that each head has a dimension for the key, query, and values of d_model / 6.
n_layer = 6 # This implies we have 6 turns to mix the embeddigs; this is "Nx" in the paper
dropout = 0.2
# ------------

class Head(nn.Module):
    """
    This class represents one head of self-attention
    Note that since this is a Decoder, this is masked-self-attention
    There is no Encoder, so there is no cross-self-attention
    """

    def __init__(self, d_head):
        super().__init__()
        self.d_head = d_head
        # Map each key, query, or value in to a d_head dimensional model.
        # Each should be matrices from d_model to d_head
        self.W_K = nn.Linear(d_model, self.d_head, bias=False)
        self.W_Q = nn.Linear(d_model, self.d_head, bias=False)
        self.W_V = nn.Linear(d_model, self.d_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, T, d_model)
        # B = batch_size, T = block_size in the below
        B,T,d = x.shape
        # Get the key and query representations from the embedding x
        # (B,T,d_head)
        k = self.W_K(x)
        # (B,T,d_head)
        q = self.W_Q(x)
        # (B,T,d_head)
        v = self.W_V(x)

        # Compute attention scores, and get the new representations for this head

        # (B T, d_head) @ (B, d_head, T) = (B, T, T)
        scores = q @ torch.transpose(k, 1, 2) / (self.d_head ** 0.5)

        # (B, T, T)
        # Apply a mask to scores, making all scores above the diagonal -inf
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # self.tril[:T, :T] == 0 creates a tensor with True above the diagonal and False on and below diagonal. scores will be replaced with float('-inf'), where self.tril is True!

        # (B, T, T)
        # Apply softmax to the final dimension of scores
        a =  F.softmax(scores, dim=-1)

        # Apply dropout
        a = self.dropout(a)
        # Perform the weighted aggregation of the values
        # Using a and v, get the new representations
        # (B, T, T) @ (B, T, d_head) -> (B, T, d_head)
        out = a @ v
        # For each token, return the weighted sum of the values
        return out

class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel
    You can have just sequential code below
    """

    def __init__(self, num_heads, d_head):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_head) for _ in range(num_heads)])
        # This is to project back to the dimension of d_model. In this case, it is just a learned linear map
        self.W_O = nn.Linear(num_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the different representations per head along the last dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # Project the concatenation and apply dropout; this is the W_O in "Attention is all you need"
        out = self.dropout(self.W_O(out))
        return out

class FeedFoward(nn.Module):
    """
    A simple linear layer followed by a non-linearity; this is applied at the token level
    """

    def __init__(self, d_model):
        super().__init__()
        d_ff = 4 * d_model
        # Map each token via a linear map to d_ff, apply ReLU, map back to d_model, and then apply dropout
        # This can be done with nn.Sequential
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.ff(x)

class DecoderBlock(nn.Module):
    """
    Transformer decoder block: communication followed by computation
    These are stacked on top of each other one after another
    """

    def __init__(self, d_model, n_head):
        super().__init__()
        # Each head gets a smaller dimensional representation of the data
        # Assume each head gets a representation of dimension d_head and d_model is divisible by n_head
        d_head = d_model // n_head
        self.sa = MultiHeadAttention(n_head, d_head)
        self.ff = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        This is different from the originl transformer paper
        In the "Attention is all you need" paper, we had
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        See Figure 1 here, and mimic that: https://arxiv.org/pdf/2002.04745.pdf

        Here, you can also do:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, decode):
        super().__init__()
        self.vocab_size = vocab_size
        self.decode = decode
        # Each token directly reads off the logits for the next token from a lookup table
        # Token embeddings are from vocab_size to d_model
        self.token_embedding_table = nn.Embedding(self.vocab_size, d_model)
        # Position embeddings are from block_size (T) to d_model
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        # This should be n_sequential applications of a DecoderBlock
        # This is the "Nx" piece in the paper
        self.blocks = nn.Sequential(
            *[DecoderBlock(d_model, n_head) for i in range(n_layer)]
        )
         # Final layer norm
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, self.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # (B,T,d_model)
        tok_emb = self.token_embedding_table(idx)
        # (T,d_model)
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=device))
        # Add positional encodings to encodings
        # (B,T,d_model)
        x = tok_emb + pos_emb

        # Mix up the token representations over and over via the blocks
        # (B,T,d_model)
        x = self.blocks(x)

        # Apply layer norm
        # (B,T,d_model)
        x = self.ln(x)

        # Apply the final linear map, to get to dimension vocab_size
        # (B,T,vocab_size)
        logits = self.ff(x)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, incremental_print=False):
        """
        idx is (B, T) array of indices in the current context
        This will generate B total paths in parrallel
        We will just geenrate 1 batch below
        """
        self.eval()
        for i in range(max_new_tokens):
            if incremental_print and i == 0:
                print(self.decode(idx[0].tolist()), end="")
            # crop idx to the last block_size tokens
            # The model only has kowledge of the context of maximum size block_size
            # Get the newest (B, T) data; T = block_size
            idx_cond = idx[:, -block_size:]


            # Get the predictions
            # (B, T, vocab_size)
            logits, loss = self(idx_cond)
            # print(idx_cond.shape)

            # Focus only on the last time step, get the logits
            # (B, vocab_size)
            logits = logits[:,-1,:]

            # Apply softmax to get probabilities
            # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution proporttional to probs
            # (B, 1)
            idx_next = torch.multinomial(probs, 1)

            if incremental_print:
                print(self.decode(idx_next[0].tolist()), end="", flush=True)

            # Append sampled index to the running sequence
            # (B, T+1)
            idx = torch.cat((idx, idx_next), -1)
        self.train()
        return idx