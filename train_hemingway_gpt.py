import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange
import my_GPT

# Hyperparameters.
batch_size = 64 # How many independent sequences will we process in parallel?
max_iters = 10000 # Max iterations we run the optimization
# How often we evaluate across the optimization; every 500 iterations
eval_interval = 500
# How many batches we use each time we evaluate
eval_iters = 200
learning_rate = 3e-4
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
# ------------

torch.manual_seed(1)

# uncomment below if running on colab and need to download hemingway.txt training data
# !gdown 'https://drive.google.com/uc?export=download&id=1RlmRmXiWVKpZq98ftdtOIdM2lsA1uw3j'

"""As usual, we read the text file and then get two dictionaries from char to idx and in reverse. char embeddings is what we will use here."""

with open('training_data/hemingway.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: take a list of integers, output a string

# Train and Test splits
data = torch.tensor(encode(text), dtype=torch.long)
train_val_split_idx = int(0.9*len(data)) # First 90% will be train, rest val
train_data = data[:train_val_split_idx]
val_data = data[train_val_split_idx:]

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # Randomly select batch_size rows from data's row indices
    ix = torch.randint(high=data.shape[0]-my_GPT.block_size-1, size=(batch_size,))
    # Select batch_size chunks of text each of size block_size; stack them
    xb = torch.stack([data[i:i+my_GPT.block_size] for i in ix])
    # Do the same for y, but make sure that this is shifted over by 1
    yb = torch.stack([data[i+1:i+my_GPT.block_size+1] for i in ix])
    # I.e. if you select xb (1, 2, 3, 4), yb should be (2, 3, 4, 5)
    xb, yb = xb.to(device), yb.to(device)
    # Each of xb, yb should be (batch_size, block_size)
    return xb, yb

@torch.no_grad()
def estimate_loss(model):
    out = {}
    # Put the model in eval mode here
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # Initilize an array of tensor of zeros of size eval_iters
        for k in range(eval_iters):
            # Get a batch of data
            xb, yb = get_batch(split)
            # Get the mean and loss
            logits, loss = model(xb, yb)

            # Get the loss for this batch
            losses[k] = loss.item()
        # Insert the mean estimate for the loss, based on the split you are in
        out[split] = losses.mean()
    # Put the model in train mode here
    model.train()

    return out

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) / train_loss > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True

model = my_GPT.GPT(vocab_size, decode).to(device)
# Print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
early_stopping = EarlyStopping(tolerance=1, min_delta=0.2)

for iter in trange(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        if iter:
          scheduler.step()
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        early_stopping(losses['train'], losses['val'])
        if early_stopping.early_stop:
          print("We stop at epoch {}".format(iter))
          break


    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

'''
1.213758 M parameters
  0%|          | 1/10000 [00:08<24:59:35,  9.00s/it]step 0: train loss 4.2890, val loss 4.2803
  5%|▌         | 503/10000 [00:53<2:46:34,  1.05s/it]step 500: train loss 2.0488, val loss 2.0475
 10%|█         | 1003/10000 [01:35<2:30:27,  1.00s/it]step 1000: train loss 1.6644, val loss 1.6775
 15%|█▌        | 1503/10000 [02:16<2:20:52,  1.01it/s]step 1500: train loss 1.4702, val loss 1.5175
 20%|██        | 2003/10000 [02:58<2:15:02,  1.01s/it]step 2000: train loss 1.3527, val loss 1.4311
 25%|██▌       | 2503/10000 [03:40<2:05:27,  1.00s/it]step 2500: train loss 1.2713, val loss 1.3739
 30%|███       | 3003/10000 [04:21<1:57:38,  1.01s/it]step 3000: train loss 1.2112, val loss 1.3384
 35%|███▌      | 3503/10000 [05:03<1:47:25,  1.01it/s]step 3500: train loss 1.1679, val loss 1.3206
 40%|████      | 4001/10000 [05:44<2:21:01,  1.41s/it]step 4000: train loss 1.1296, val loss 1.3066
 45%|████▌     | 4503/10000 [06:25<1:32:35,  1.01s/it]step 4500: train loss 1.0997, val loss 1.2955
 50%|█████     | 5003/10000 [07:08<1:37:42,  1.17s/it]step 5000: train loss 1.0758, val loss 1.2885
 55%|█████▌    | 5500/10000 [07:49<06:24, 11.70it/s]step 5500: train loss 1.0521, val loss 1.2832
We stop at epoch 5500
'''

# Start the model with a new line, generate up to 10000 tokens
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=100, incremental_print=False)[0].tolist()))

torch.save(model.state_dict(), '/models/gpt-hemingway-1M.pt')

