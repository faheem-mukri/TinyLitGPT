from data.dataloader import get_batch
x, y = get_batch(batch_size = 2, block_size = 8, device = "cpu")

print("X:",x)
print("Y:",y)