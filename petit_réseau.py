# -*- coding: utf-8 -*-
"""Improved Network — MNIST digit classifier

Optimized for best accuracy with a simple fully-connected architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# 1. Define transformations (convert image to tensor and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Section 2: Network configuration
# Modify these parameters, then re-run from this cell onward.

digits_to_identify = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # classes
num_hidden_layers = 2  # hidden layers
neurons_per_hidden_layer = 256  # neurons per layer
num_epochs = 15  # training epochs
learning_rate = 0.001  # Adam LR
dropout_rate = 0.2  # dropout regularization
batch_size = 64

# Section 3: Training and testing

# 2. Download MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


def filter_digits(dataset):
    """Keep only the digits listed in digits_to_identify and re-label them 0..N-1."""
    indices = dataset.targets == digits_to_identify[0]
    dataset.targets[dataset.targets == digits_to_identify[0]] = 0
    for ii, digit in enumerate(digits_to_identify[1:]):
        indices = indices | (dataset.targets == digit)
        dataset.targets[dataset.targets == digit] = ii + 1
    return Subset(dataset, torch.where(indices)[0])


train_subset = filter_digits(train_dataset)
test_subset = filter_digits(test_dataset)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)


class FlexibleNet(nn.Module):
    def __init__(self, num_hidden_layers=1, hidden_size=128, output_size=10, dropout=0.2):
        super(FlexibleNet, self).__init__()

        self.layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer (784 -> hidden_size)
        self.layers.append(nn.Linear(784, hidden_size))
        self.batchnorms.append(nn.BatchNorm1d(hidden_size))
        self.dropouts.append(nn.Dropout(dropout))

        # Hidden layers (hidden_size -> hidden_size)
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.batchnorms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(dropout))

        # Output layer (hidden_size -> output_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, 784)

        for linear, bn, drop in zip(self.layers, self.batchnorms, self.dropouts):
            x = linear(x)
            x = bn(x)  # batch normalization (stabilizes + speeds up training)
            x = torch.relu(x)
            x = drop(x)  # dropout (prevents overfitting)

        x = self.output_layer(x)
        return x


# Create model, loss, optimizer, and LR scheduler

model = FlexibleNet(num_hidden_layers, neurons_per_hidden_layer,
                    len(digits_to_identify), dropout_rate)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
    scheduler.step()
    avg_loss = running_loss / num_batches
    print(f"Epoch {epoch + 1}/{num_epochs} — Avg loss: {avg_loss:.4f}")

# Evaluation

model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

accuracy = round(100 * correct / len(test_subset), 2)
print(f"\nTest accuracy: {accuracy}%")

# Show misclassified examples

misclassified_images = []
true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        errors = (predicted != labels)
        if errors.any():
            misclassified_images.extend(images[errors])
            true_labels.extend(labels[errors])
            predicted_labels.extend(predicted[errors])

if len(misclassified_images) > 0:
    print(f"Misclassified images: {len(misclassified_images)}")
    plt.figure(figsize=(12, 5))
    num_to_display = min(len(misclassified_images), 8)
    for i in range(num_to_display):
        plt.subplot(1, num_to_display, i + 1)
        img = misclassified_images[i].squeeze() * 0.3081 + 0.1307
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {digits_to_identify[true_labels[i]]}\n"
                  f"Pred: {digits_to_identify[predicted_labels[i]]}")
        plt.axis('off')
    plt.show()
else:
    print("No errors on the test set!")
