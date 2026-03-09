import os
import glob
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Import your custom NCResNet18 from your model_create.py file
from model_create import *

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def get_balanced_dataset(dataset, samples_per_class=5000):
    targets = torch.tensor(dataset.targets)
    indices = []
    for class_id in range(10): 
        class_indices = (targets == class_id).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_indices), generator=torch.Generator().manual_seed(42))
        selected_indices = class_indices[perm[:samples_per_class]]
        indices.extend(selected_indices.tolist())
    return Subset(dataset, indices)

def load_data(dataset_name='mnist', batch_size=128):
    dataset_path = './data'
    
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

    balanced_train = get_balanced_dataset(full_train, samples_per_class=5000)
    loader = DataLoader(balanced_train, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader

# ==========================================
# 2. NEURAL COLLAPSE METRICS ENGINE
# ==========================================
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
    Sigma_B_pinv = torch.linalg.pinv(Sigma_B.cpu()).to(device)
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
    # Loop over all three datasets
    DATASETS = ['mnist', 'fmnist', 'cifar10']
    EPOCHS = 350 
    BATCH_SIZE = 128 
    INITIAL_LR = 0.05 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Create a directory for outputs
    os.makedirs('results', exist_ok=True)

    for dataset_name in DATASETS:
        print(f"\n{'='*50}")
        print(f"BEGINNING BASE MODEL TRAINING FOR: {dataset_name.upper()}")
        print(f"{'='*50}")

        train_loader = load_data(dataset_name=dataset_name, batch_size=BATCH_SIZE)
        model = NCResNet18(dataset_name=dataset_name).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9, weight_decay=5e-4)
        
        milestones = [int(EPOCHS * (1/3)), int(EPOCHS * (2/3))]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        # Setup CSV Logging
        csv_filename = f"results/{dataset_name}_base_training_history.csv"
        
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Epoch', 'LR', 'Loss', 'Accuracy', 'NC1', 'NC2', 'NC3', 'NC4'])

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
                current_lr = scheduler.get_last_lr()[0]
                
                # Compute NC metrics
                nc1, nc2, nc3, nc4 = compute_nc_metrics(model, train_loader)
                
                # Log to CSV
                csv_writer.writerow([epoch+1, current_lr, epoch_loss, epoch_acc, nc1, nc2, nc3, nc4])
                
                print(f"Epoch [{epoch+1:03d}/{EPOCHS}] "
                      f"LR: {current_lr:.4f} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:05.2f}% || "
                      f"NC1: {nc1:.4f} | NC2: {nc2:.4f} | NC3: {nc3:.4f} | NC4: {nc4:.4f}")

                # STOP TRAINING & SAVE MODEL ON 100% ACCURACY
                if epoch_acc >= 100.0:
                    model_save_path = f"results/{dataset_name}_100_acc_model.pt"
                    torch.save(model.state_dict(), model_save_path)
                    print(f">>> 100% Accuracy Reached for {dataset_name.upper()}! <<<")
                    print(f">>> Saved base model to {model_save_path}. Moving to next dataset. <<<")
                    break # Breaks out of the epoch loop and moves to the next dataset

if __name__ == '__main__':
    main()