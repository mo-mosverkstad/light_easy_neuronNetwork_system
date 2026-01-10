import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) 数据与加载器
transform = transforms.Compose([
    transforms.ToTensor(),                 # [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值/方差
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

# 2) 模型（简单 MLP 或 CNN）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),             # 28->14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),             # 14->7
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# 3) 损失与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4) 训练函数
def train_one_epoch(epoch):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    print(f"Epoch {epoch} | Train Loss: {total_loss/total_samples:.4f} "
          f"| Train Acc: {total_correct/total_samples:.4f}")

# 5) 测试函数
def evaluate():
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
    print(f"Test Acc: {total_correct/total_samples:.4f}")

# 6) 训练与评估
for epoch in range(1, 6):
    train_one_epoch(epoch)
    evaluate()
