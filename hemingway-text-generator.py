import torch
import my_GPT

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

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


model = my_GPT.GPT(vocab_size, decode).to(device)
model.load_state_dict(torch.load('models/gpt-hemingway-1M.pt', map_location=device))

print("Welcome to Hemingway text generator. How many characters long do you want your generated Hemingway story to be?")
n = int(input())
print("This Hemingway text generator will complete a story in the style of Hemingway given a first sentence. Please input your first sentence (or a few words):")
x = input()
x = encode(x)
print('\n')

# Start the model with the inputted sentence x, generate up to n tokens
context = torch.tensor([x], dtype=torch.long, device=device)
model.generate(context, max_new_tokens=n, incremental_print=True)

print('\n\nGeneration Complete!')