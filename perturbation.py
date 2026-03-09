import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_create import NCResNet18
from train import load_data, compute_nc_metrics

# ==========================================
# 1. DEEPFOOL ALGORITHM 
# ==========================================
def deepfool(image, model, num_classes=10, overshoot=0.02, max_iter=50):
    """
    Calculates the minimal perturbation required to cross the decision boundary.
    Fixed: Uses torch.autograd.grad to prevent contamination and applies correct squared-norm step scaling.
    """
    device = image.device
    image = image.clone().detach().requires_grad_(True)
    
    output = model(image)
    _, original_pred = torch.max(output, 1)
    
    current_image = image.clone().detach().requires_grad_(True)
    perturbation = torch.zeros_like(image).to(device)
    
    for _ in range(max_iter):
        fs = model(current_image)[0]
        _, current_pred = torch.max(fs.unsqueeze(0), 1)
        
        if current_pred != original_pred:
            break 
            
        # FIX 1: Safe gradient extraction for the original class
        grad_orig = torch.autograd.grad(fs[original_pred], current_image, retain_graph=True)[0]
        
        min_dist = float('inf')
        w_min = None
        f_min = None
        
        for k in range(num_classes):
            if k == original_pred:
                continue
                
            # FIX 1: Safe gradient extraction for class k
            grad_k = torch.autograd.grad(fs[k], current_image, retain_graph=True)[0]
            
            w_k = grad_k - grad_orig
            f_k = fs[k] - fs[original_pred]
            
            dist_k = abs(f_k.item()) / (torch.norm(w_k.flatten()) + 1e-8)
            
            if dist_k < min_dist:
                min_dist = dist_k
                w_min = w_k
                f_min = f_k
                
        # FIX 2: Correct DeepFool step scaling using squared norm
        step = (abs(f_min.item()) / (torch.norm(w_min.flatten())**2 + 1e-8)) * w_min * (1 + overshoot)
        
        perturbation += step
        current_image = (image + perturbation).clone().detach().requires_grad_(True)
        
    return perturbation

def calculate_robustness_metric(model, dataloader, num_samples=1000):
    model.eval()
    device = next(model.parameters()).device
    total_rho = 0.0
    samples_processed = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        for i in range(inputs.size(0)):
            if samples_processed >= num_samples:
                break
                
            single_img = inputs[i:i+1] 
            r_x = deepfool(single_img, model)
            
            r_norm = torch.norm(r_x.flatten()).item()
            x_norm = torch.norm(single_img.flatten()).item()
            
            total_rho += (r_norm / (x_norm + 1e-8))
            samples_processed += 1
            
        if samples_processed >= num_samples:
            break
            
    model.train()
    return total_rho / samples_processed

# ==========================================
# 2. VISUALIZATION ENGINE
# ==========================================
def denormalize(tensor, dataset_name):
    """Reverts normalization for plotting publishable images."""
    tensor = tensor.clone().detach().cpu().squeeze(0)
    if dataset_name == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    elif dataset_name == 'mnist':
        mean = torch.tensor([0.1307]).view(1, 1, 1)
        std = torch.tensor([0.3081]).view(1, 1, 1)
    else: # fmnist
        mean = torch.tensor([0.2860]).view(1, 1, 1)
        std = torch.tensor([0.3530]).view(1, 1, 1)
        
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    if tensor.shape[0] == 1: # Grayscale
        return tensor.squeeze(0).numpy(), 'gray'
    else: # RGB
        return tensor.permute(1, 2, 0).numpy(), None

def save_adversarial_examples(model, dataloader, dataset_name, phase_name):
    """Saves a 300 DPI figure showing 1 adversarial example per class."""
    print(f"Generating publishable adversarial figures for {phase_name}...")
    model.eval()
    device = next(model.parameters()).device
    classes_found = set()
    examples = {}

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        for i in range(inputs.size(0)):
            label = targets[i].item()
            if label not in classes_found:
                single_img = inputs[i:i+1]
                
                # Get the perturbation
                r_x = deepfool(single_img, model)
                adv_img = single_img + r_x
                
                # Get the new classification
                new_pred = torch.argmax(model(adv_img)).item()
                
                examples[label] = (single_img, r_x, adv_img, new_pred)
                classes_found.add(label)
                
            if len(classes_found) == 10:
                break
        if len(classes_found) == 10:
            break

    fig, axes = plt.subplots(10, 3, figsize=(10, 25))
    fig.suptitle(f"{dataset_name.upper()} - {phase_name} Phase Adversarial Examples", fontsize=16)
    
    for label in range(10):
        orig, r_x, adv, new_pred = examples[label]
        
        orig_img, cmap = denormalize(orig, dataset_name)
        adv_img, _ = denormalize(adv, dataset_name)
        
        # Amplify perturbation for visual clarity in paper
        pert_img, _ = denormalize(r_x * 10, dataset_name) 

        axes[label, 0].imshow(orig_img, cmap=cmap)
        axes[label, 0].set_title(f"Original: Class {label}")
        axes[label, 0].axis('off')
        
        axes[label, 1].imshow(pert_img, cmap=cmap)
        axes[label, 1].set_title("Perturbation (x10)")
        axes[label, 1].axis('off')
        
        axes[label, 2].imshow(adv_img, cmap=cmap)
        axes[label, 2].set_title(f"Adversarial: Classified as {new_pred}")
        axes[label, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f"results/{dataset_name}_{phase_name}_adv_figures.png", dpi=300, bbox_inches='tight')
    plt.close()
    model.train()

# ==========================================
# 3. EXPERIMENT WORKFLOW
# ==========================================
def run_tpt_experiment(dataset_name='mnist', extra_epochs=300, robustness_freq=25):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"STARTING TPT EXPERIMENT FOR: {dataset_name.upper()}")
    print(f"{'='*50}")

    train_loader = load_data(dataset_name=dataset_name, batch_size=128)
    model = NCResNet18(dataset_name=dataset_name).to(device)
    base_model_path = f"results/{dataset_name}_100_acc_model.pt"
    
    if not os.path.exists(base_model_path):
        print(f"Error: {base_model_path} not found.")
        return

    model.load_state_dict(torch.load(base_model_path, map_location=device))
    
    # --- PHASE 1: Baseline Measurements ---
    initial_rho = calculate_robustness_metric(model, train_loader, num_samples=500)
    save_adversarial_examples(model, train_loader, dataset_name, "Base_100_Acc")
    
    # --- PHASE 2: TPT Training & Tracking ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    
    csv_filename = f"results/{dataset_name}_tpt_history.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Loss', 'NC1', 'NC2', 'NC3', 'NC4', 'Rho_Adv'])

        current_rho = initial_rho 

        for epoch in range(extra_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(train_loader.dataset)
            nc1, nc2, nc3, nc4 = compute_nc_metrics(model, train_loader)
            
            # Recalculate robustness periodically to save time
            if (epoch + 1) % robustness_freq == 0:
                current_rho = calculate_robustness_metric(model, train_loader, num_samples=250)
                
            csv_writer.writerow([epoch+1, epoch_loss, nc1, nc2, nc3, nc4, current_rho])
            print(f"TPT Epoch [{epoch+1:03d}/{extra_epochs}] Loss: {epoch_loss:.6f} | NC1: {nc1:.4f} | Rho_Adv: {current_rho:.6f}")

    # --- PHASE 3: Final Measurements ---
    save_adversarial_examples(model, train_loader, dataset_name, "Fully_Collapsed")
    torch.save(model.state_dict(), f"results/{dataset_name}_collapsed_model.pt")

if __name__ == '__main__':
    DATASETS = ['mnist', 'fmnist', 'cifar10']
    for dataset in DATASETS:
        run_tpt_experiment(dataset_name=dataset, extra_epochs=300, robustness_freq=50)