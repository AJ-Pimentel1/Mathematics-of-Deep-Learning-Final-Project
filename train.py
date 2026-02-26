import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
import numpy as np
from model_create import *

model_cifar = NCResNet18(dataset_name = 'cifar10')
model_mnist = NCResNet18(dataset_name = 'mnist')
model_fmnist = NCResNet18(dataset_name = 'fmnist')


def get_balanced_dataset(dataset, samples_per_class=5000):
    # Subsamples the dataset to 5,000 examples per class to perfectly balance it.
    targets = torch.tensor(dataset.targets)
    indices = []
    for class_id in range(10): 
        class_indices = (targets == class_id).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_indices), generator=torch.Generator().manual_seed(42))
        selected_indices = class_indices[perm[:samples_per_class]]
        indices.extend(selected_indices.tolist())
    return Subset(dataset, indices)

def load_data(dataset_name='mnist', batch_size=128):
    # Point directly to your existing data folder
    dataset_path = './data'
    
    # Pre-process images by subtracting the mean and dividing by the standard deviation.
    # No data augmentation is used.
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        full_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train = datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
        
    elif dataset_name == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        full_train = datasets.FashionMNIST(root=dataset_path, train=True, download=True, transform=transform)

    # Subsample to 5,000 examples per class.
    balanced_train = get_balanced_dataset(full_train, samples_per_class=5000)
    
    loader = DataLoader(balanced_train, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader

@torch.no_grad()
def compute_nc_metrics(model, dataloader, num_classes=10):
    model.eval()
    device = next(model.parameters()).device
    W = model.backbone.fc.weight.detach()
    
    all_features, all_labels = [], []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, features = model(inputs, return_features=True)
        all_features.append(features)
        all_labels.append(targets)
        
    H = torch.cat(all_features, dim=0)
    Y = torch.cat(all_labels, dim=0)
    
    mu_G = torch.mean(H, dim=0, keepdim=True)
    mu_c = torch.zeros(num_classes, H.shape[1], device=device)
    
    for c in range(num_classes):
        class_mask = (Y == c)
        mu_c[c] = torch.mean(H[class_mask], dim=0)
        
    M_dot = (mu_c - mu_G).T 
    
    # NC1: Variability Collapse
    Sigma_W = torch.zeros(H.shape[1], H.shape[1], device=device)
    for c in range(num_classes):
        class_mask = (Y == c)
        H_c = H[class_mask] - mu_c[c]
        Sigma_W += torch.mm(H_c.T, H_c) / len(Y)
        
    Sigma_B = torch.mm(M_dot, M_dot.T) / num_classes
    Sigma_B_pinv = torch.linalg.pinv(Sigma_B)
    nc1_metric = torch.trace(torch.mm(Sigma_W, Sigma_B_pinv)).item() / num_classes

    # NC2: Convergence to Simplex ETF
    M_dot_normalized = F.normalize(M_dot, p=2, dim=0)
    cosine_matrix = torch.mm(M_dot_normalized.T, M_dot_normalized)
    mask = ~torch.eye(num_classes, dtype=torch.bool, device=device)
    off_diagonals = cosine_matrix[mask]
    nc2_metric = torch.std(off_diagonals).item()

    # NC3: Convergence to Self-Duality
    W_normalized = W.T / torch.norm(W, p='fro')
    M_normalized = M_dot / torch.norm(M_dot, p='fro')
    nc3_metric = torch.norm(W_normalized - M_normalized, p='fro').item() ** 2

    # NC4: Simplification to Nearest Class-Center
    model_preds = torch.argmax(torch.mm(H, W.T) + model.backbone.fc.bias, dim=1)
    distances = torch.cdist(H, mu_c)
    ncc_preds = torch.argmin(distances, dim=1)
    nc4_metric = (model_preds != ncc_preds).float().mean().item()
    
    model.train()
    return nc1_metric, nc2_metric, nc3_metric, nc4_metric

# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================
def main():
    # Configuration
    DATASET = 'cifar10' # Change to 'mnist' or 'fmnist'
    
    # Training parameters based on the paper's optimization methodology
    EPOCHS = 350 #
    BATCH_SIZE = 128 #
    INITIAL_LR = 0.05 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Initialization
    train_loader = load_data(dataset_name=DATASET, batch_size=BATCH_SIZE)
    model = NCResNet18(dataset_name=DATASET).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer settings from paper: SGD with momentum 0.9 and weight decay 5e-4.
    optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate drops by factor of 10 at 1/3 and 2/3 of epochs.
    milestones = [int(EPOCHS * (1/3)), int(EPOCHS * (2/3))]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    print(f"Starting Training for {EPOCHS} Epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        # At the end of the epoch, compute NC metrics
        nc1, nc2, nc3, nc4 = compute_nc_metrics(model, train_loader)
        
        print(f"Epoch [{epoch+1:03d}/{EPOCHS}] "
              f"LR: {scheduler.get_last_lr()[0]:.4f} | "
              f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:05.2f}% || "
              f"NC1: {nc1:.4f} | NC2: {nc2:.4f} | NC3: {nc3:.4f} | NC4: {nc4:.4f}")

if __name__ == '__main__':
    main()