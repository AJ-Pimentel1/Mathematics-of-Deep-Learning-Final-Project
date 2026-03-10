# Mathematics-of-Deep-Learning-Final-Project

## Project Overview

Neural Collapse (NC) refers to an inductive bias characterized by four interrelated geometric phenomena that emerge when a neural network enters the **Terminal Phase of Training (TPT)**. These phenomena — **variability collapse**, **convergence to a simplex equiangular tight frame (ETF)**, **convergence to self-duality**, and **simplification to nearest-class-center behavior** — collectively describe the geometry of the feature space in the final stage of training.

This repository contains experiments examining the **robustness and generalization of Neural Collapse under adversarial and memorization-style settings**. Unlike prior work that primarily evaluates NC robustness using gradient-based perturbations, we apply **DeepFool**, an optimization-based adversarial attack, to estimate the **minimum perturbation required to change a model’s prediction**.

Our experiments track how adversarial robustness evolves as networks continue training well beyond perfect classification accuracy, and we analyze how Neural Collapse behaves when networks are forced to memorize **randomized labels**.

---

# Goals & Hypotheses

### Primary Goal
Measure how the **average minimum adversarial perturbation** (via DeepFool) evolves as Neural Collapse strengthens during extended training into the Terminal Phase of Training.

### Hypothesis
As Neural Collapse solidifies, the **average minimum perturbation required to change predictions will increase**, suggesting that continued training produces larger classification margins.

### Secondary Goal
Test whether Neural Collapse emerges when models **memorize randomized labels**.

### Hypothesis (Memorization Test)
An overparameterized network may achieve zero empirical risk on shuffled labels via memorization, but **full Neural Collapse will not emerge**, since class semantics are destroyed.

---

# Experimental Pipeline

The current workflow consists of four main stages.

## 1. Dataset Generation

Datasets are prepared using:

```bash
python datagen.py
```

This script downloads and prepares the datasets used in the experiments:

- MNIST  
- Fashion-MNIST  
- CIFAR-10  

---

## 2. Base Model Training

Next, we train baseline models to **100% training accuracy**.

```bash
python train.py
```

This trains an overparameterized **NCResNet18** model for each dataset and saves the trained models:

```
results/mnist_100_acc_model.pt
results/fmnist_100_acc_model.pt
results/cifar10_100_acc_model.pt
```

These models serve as the **starting point for Terminal Phase Training (TPT)** experiments.

---

## 3. Neural Collapse + Adversarial Robustness Experiments

After baseline models reach perfect accuracy, we run the Neural Collapse experiments:

```bash
python perturbation.py
```

This script performs the following:

1. Loads the **100% accuracy models**  
2. Measures baseline adversarial robustness using **DeepFool**  
3. Continues training for **300 additional epochs** (Terminal Phase Training)  
4. Computes **Neural Collapse metrics each epoch**  
5. Recomputes adversarial robustness every **25 epochs**  
6. Saves adversarial examples before and after collapse  

### Metrics Recorded

For each epoch we compute:

- **NC1** – Variability Collapse  
- **NC2** – Convergence to Simplex ETF  
- **NC3** – Self-Duality  
- **NC4** – Nearest Class Center behavior  
- **ρ_adv** – Average DeepFool perturbation magnitude  

Results are written to:

```
results/{dataset}_tpt_history.csv
```

### Key Observation

Across MNIST, Fashion-MNIST, and CIFAR-10 we observe:

**ρ_adv increases approximately 8–11× after Neural Collapse**

This suggests that continued training during TPT **substantially increases adversarial margins**.

### Generated Outputs

The experiment also generates:

**Adversarial Example Visualizations**

```
results/{dataset}_Base_100_Acc_adv_figures.png
results/{dataset}_Fully_Collapsed_adv_figures.png
```

Each figure shows:

- Original image  
- Amplified perturbation  
- Adversarial example  

for **one sample from each class**.

---

## 4. Label Shuffling Experiment

To test whether Neural Collapse depends on semantic structure, we perform a **label randomization experiment**.

```bash
python shuffle.py
```

This script:

1. Randomly shuffles dataset labels  
2. Retrains the network on shuffled labels  
3. Tracks Neural Collapse metrics during memorization  

This allows us to compare:

- Neural Collapse with **true semantic labels**  
- Neural Collapse under **pure memorization**

---

# Experiments

## Datasets

- MNIST  
- Fashion-MNIST  
- CIFAR-10  

## Model

All experiments use a modified **ResNet18 architecture (NCResNet18)** designed for Neural Collapse analysis.

## Training Setup

- Optimizer: SGD  
- Momentum: 0.9  
- Weight decay: 5e-4  
- Batch size: 128  
- Additional TPT epochs: **300**  
- Robustness recomputation frequency: **every 25 epochs**

---

# Conda Environment Setup

Create and activate the Conda environment using the provided file.

```bash
conda env create -f environment.yml
conda activate ncollapse
```

If installing manually:

```bash
conda create -n ncollapse python=3.10 -y
conda activate ncollapse

conda install -c conda-forge numpy scipy -y
pip install torch torchvision foolbox tqdm matplotlib
```

---

# Quick Start

Run the full experiment pipeline:

```bash
# 1. Download datasets
python datagen.py

# 2. Train base models to 100% accuracy
python train.py

# 3. Run Neural Collapse + DeepFool experiments
python perturbation.py

# 4. Run shuffled-label memorization experiment
python shuffle.py
```

---

# Reproducibility Notes

- Report random seeds when running experiments.
- DeepFool is iterative and sensitive to attack parameters; ensure **consistent attack settings across checkpoints**.
- GPU acceleration is recommended for the perturbation experiments.

---

# References

Papyan et al., Neural Collapse phenomenon and analysis, 2020.

Moosavi-Dezfooli et al., DeepFool: A simple and accurate method to fool deep neural networks, 2016.

Kothapalli et al., Neural Collapse review and robustness modeling, 2023.

Nguyen et al., Memorization dynamics in Neural Collapse experiments, 2023.
