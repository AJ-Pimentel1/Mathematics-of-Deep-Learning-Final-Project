import os
import glob
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Fix MNIST mirror issue
datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]

def clean_up_archives(dataset_path):
    """
    Finds and deletes leftover .gz and .tar.gz files 
    after PyTorch has finished extracting them.
    """
    search_path = os.path.join(dataset_path, '**', '*.gz')
    for file_path in glob.glob(search_path, recursive=True):
        try:
            os.remove(file_path)
            print(f"Cleaned up raw archive: {file_path}")
        except Exception as e:
            pass

def get_balanced_dataset(dataset, samples_per_class=5000):
    """
    Subsamples a dataset to have exactly n samples per class.
    """
    indices = []
    # Use torch.tensor to avoid numpy versioning issues
    targets = torch.tensor(dataset.targets)
    
    for class_id in range(10): 
        class_indices = (targets == class_id).nonzero(as_tuple=True)[0]
        # Deterministic shuffle
        perm = torch.randperm(len(class_indices), generator=torch.Generator().manual_seed(42))
        selected_indices = class_indices[perm[:samples_per_class]]
        indices.extend(selected_indices.tolist())
        
    return Subset(dataset, indices)

def load_data(dataset_name='mnist'):
    if dataset_name == 'cifar10':
        dataset_path = './data/CIFAR10'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        full_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
    
    elif dataset_name == 'mnist':
        dataset_path = './data/MNIST'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train = datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
        
    elif dataset_name == 'fmnist':
        dataset_path = './data/FMNIST'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        full_train = datasets.FashionMNIST(root=dataset_path, train=True, download=True, transform=transform)

    # Clean up the raw downloaded archives
    clean_up_archives(dataset_path)

    # Subsample to 5000 per class
    balanced_train = get_balanced_dataset(full_train, samples_per_class=5000)
    
    loader = DataLoader(balanced_train, batch_size=128, shuffle=True, num_workers=2)
    print(f"Loaded {len(balanced_train)} samples for {dataset_name}")
    return loader

# Example usage:
if __name__ == "__main__":
    mnist_loader = load_data('mnist')
    fmnist_loader = load_data('fmnist')
    cifar_loader = load_data('cifar10')