"""CNN model implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List

from core.interfaces import ModelInterface

class SimpleCNN(nn.Module, ModelInterface):
    """Simple CNN model for CIFAR-10 classification"""
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weights(self) -> List[torch.Tensor]:
        """Get model weights as a list of tensors"""
        return [p.data.clone().cpu() for p in self.parameters()]

    def set_weights(self, weights: List[torch.Tensor]) -> None:
        """Set model weights from a list of tensors"""
        with torch.no_grad():
            for p, w in zip(self.parameters(), weights):
                p.copy_(w.to(p.device))

    def train_step(self, data: torch.Tensor, labels: torch.Tensor) -> float:
        """Perform one training step and return loss"""
        self.train()
        criterion = nn.CrossEntropyLoss()
        outputs = self(data)
        loss = criterion(outputs, labels)
        return loss.item()
    
    def evaluate(self, test_loader, device):
        """Evaluate model on test data"""
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy