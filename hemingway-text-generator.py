import torch
import train_hemingway_gpt.py

model = GPT()
model.load_state_dict(torch.load('models/gpt-hemingway-1M.pt'))
model.eval()

print("Welcome to Hemingway text generator. Please input ")