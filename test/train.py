import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch import nn, optim
import time

# Complex CNN model for example purposes
class MyComplexModel(nn.Module):
    def __init__(self):
        super(MyComplexModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Assuming input images are 32x32
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 classes for example
        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(nn.functional.relu(self.fc1(x)))  # Apply dropout
        x = self.dropout(nn.functional.relu(self.fc2(x)))  # Apply dropout
        x = self.fc3(x)  # Final output
        return x

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set device for this process

def cleanup():
    # Clean up the process group
    dist.destroy_process_group()

def train(rank, world_size, epochs=100):
    setup(rank, world_size)
    
    # Create model and move it to the GPU corresponding to the rank
    model = MyComplexModel().to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])

    # Dummy dataset - Random images and labels
    data = torch.randn(10000, 3, 32, 32)  # 3 channels, 32x32 images
    labels = torch.randint(0, 10, (10000,))  # Random labels for 10 classes
    dataset = torch.utils.data.TensorDataset(data, labels)

    # Use DistributedSampler for distributed training
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=128, sampler=sampler)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer

    # Training loop
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)  # Shuffle data every epoch for better training
        
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(rank)
            batch_labels = batch_labels.to(rank)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
        print(f"Rank {rank}, Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = 8  # Number of GPUs/nodes
    rank = int(os.environ['RANK'])  # This should be set by the launching mechanism
    print(f"rank: {rank}")

    start = time.time()
    train(rank, world_size)
    end = time.time()
    
    total = end - start
    print(f"Total time: {total:.2f} seconds")
