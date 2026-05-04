import torch

#loading tokenized data
data = torch.load("data/train.pt")

def get_batch(batch_size, block_size, device):

    #generate random starting indices for the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))

    #the input is a sequence of tokens of length block_size
    x = torch.stack([data[i:i+block_size] for i in ix])

    #the target is the input shifted by one position
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)