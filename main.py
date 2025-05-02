import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from models.model import efficientnet_b0_condconv  # Make sure this exists

# Set random seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ===== CONFIG =====
config = {
    "use_dropout": 1,        # 0: no dropout, 1: fixed dropout, 2: random dropout
    "dropout_prob": 0.5,     # only used if use_dropout = 1
    "use_mixup": True,
    "mixup_alpha": 1.0,
    "use_autoaugment": True,
    "use_condconv": True     # Set to False to use standard EfficientNet-B0
}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
epochs = 50
learning_rate = 0.001
num_classes = 100
num_experts = 12

# ===== Transforms =====
transform_list = [
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
]

if config["use_autoaugment"]:
    transform_list.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))

transform_list += [
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
]

transform_train = transforms.Compose(transform_list)
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# ===== Datasets and Loaders =====
# train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
# test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ===== Model Wrapper =====
from torchvision.models import efficientnet_b0

class EfficientNetWrapper(nn.Module):
    def __init__(self, num_classes, use_condconv=True, num_experts=4, use_dropout=2, dropout_prob=0.5):
        super().__init__()
        if use_condconv:
            self.model = efficientnet_b0_condconv(num_experts=num_experts)
        else:
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        # Dropout control
        self.use_dropout = use_dropout
        if self.use_dropout == 0:
            self.dropout = nn.Identity()
        elif self.use_dropout == 1:
            self.dropout = nn.Dropout(p=dropout_prob)
        elif self.use_dropout == 2:
            self.dropout = nn.Dropout(p=random.uniform(0, 0.4))  # Random dropout prob each run

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x

# ===== Model, Loss, Optimizer =====
model = EfficientNetWrapper(
    num_classes=num_classes,
    use_condconv=config["use_condconv"],
    num_experts=num_experts,
    use_dropout=config["use_dropout"],
    dropout_prob=config["dropout_prob"]
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===== Mixup Functions =====
def mixup_data(x, y, alpha=1.0):
    lam = random.betavariate(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ===== Training Loop =====
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    progress_bar = tqdm(loader, desc="Training", unit="batch", ncols=100)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Mixup if enabled
        if config["use_mixup"]:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=config["mixup_alpha"])
        else:
            targets_a, targets_b, lam = targets, targets, 1.0

        optimizer.zero_grad()
        outputs = model(inputs)

        if config["use_mixup"]:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets_a).sum().item()

        progress_bar.set_postfix(loss=running_loss / (total + 1e-6))

    acc = 100. * correct / total
    avg_loss = running_loss / total
    return avg_loss, acc

# ===== Evaluation Loop =====
def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_total = 0, 0, 0.0

    progress_bar = tqdm(loader, desc="Evaluating", unit="batch", ncols=100)
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_total += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix(loss=loss_total / (total + 1e-6))

    acc = 100. * correct / total
    avg_loss = loss_total / total
    return avg_loss, acc

# ===== Run Training and Save Results =====
results = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
}

for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

    results["epoch"].append(epoch)
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

# ===== Save Results to CSV =====
dropout_type = {0: "nodrop", 1: "fixdrop", 2: "randdrop"}[config["use_dropout"]]
dropout_prob = config["dropout_prob"]
mixup_flag = "mixup" if config["use_mixup"] else "nomixup"
autoaug_flag = "autoaug" if config["use_autoaugment"] else "noautoaug"
condconv_flag = "condconv" if config["use_condconv"] else "baseline"

filename = f"results_{condconv_flag}_{dropout_type}_p{dropout_prob}_{mixup_flag}_{autoaug_flag}.csv"

df = pd.DataFrame(results)
df.to_csv(filename, index=False)
print(f"\nResults saved to {filename}")
