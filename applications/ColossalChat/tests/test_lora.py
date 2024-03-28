import torch
import torch.nn as nn
import torch.optim as optim
from coati.models import convert_to_lora_module
from torch.utils.data import DataLoader, TensorDataset


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def test_overfit():
    input_size = 1000
    hidden_size = 200
    num_classes = 5
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 200

    # Synthesized dataset
    X = torch.randn(batch_size, input_size)
    Y = torch.randint(0, num_classes, (batch_size,))

    # Convert to DataLoader
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build and convert model
    model = SimpleNN(input_size, hidden_size, num_classes)
    weight_to_compare = model.fc1.weight.detach().clone()
    model = convert_to_lora_module(model, lora_rank=30)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for _ in range(num_epochs):
        for i, (inputs, labels) in enumerate(loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(loss)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Check if model has overfitted
    outputs = model(X)
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == Y).sum().item()
    assert (correct / total > 0.95, "The model has not overfitted to the synthesized dataset")
    assert (weight_to_compare - model.fc1.weight).sum() < 0.01


if __name__ == "__main__":
    test_overfit()
