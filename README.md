# Mathematics-of-Deep-Learning-Final-Project

## Project Overview

Neural Collapse (NC) refers to an inductive bias characterized by four interrelated geometric phenomena that emerge when a neural network enters the Terminal Phase of Training (TPT) \cite{DBLP:journals/corr/abs-2008-08186}. These phenomena — variability collapse, convergence to a simplex equiangular tight frame (ETF), convergence to self-duality, and simplification to nearest-class-center behavior — collectively describe the network's terminal feature space \cite{DBLP:journals/corr/abs-2008-08186}.

This repository contains experiments examining the robustness and generalization of Neural Collapse under adversarial and memorization-style settings. Unlike prior work that primarily evaluates NC robustness using gradient-based perturbations \cite{kothapalli2023neuralcollapsereviewmodelling}, we apply DeepFool — an optimization-based adversarial attack — to measure the minimum perturbation required to flip class predictions as networks progress deeper into TPT \cite{moosavidezfooli2016deepfoolsimpleaccuratemethod}.

## Goals & Hypotheses
- **Primary goal:** Measure how the average minimum adversarial perturbation (via DeepFool) evolves as Neural Collapse strengthens during extended training into TPT.
- **Hypothesis:** As NC manifestations solidify, the average minimum perturbation required to change predictions will increase, indicating a structural benefit to training well into the terminal phase.
- **Secondary goal:** Test NC generalization under label-randomized multiclass data to evaluate whether semantic similarity is necessary for variability compression.
- **Hypothesis (memorization test):** An overparameterized network may achieve zero empirical risk via memorization on shuffled labels, but will fail to reach full Neural Collapse across semantically disjoint classes.

## Methods
- Train overparameterized classification models to and into the Terminal Phase of Training (TPT).
- At checkpoints spanning training, compute NC metrics (variability collapse, ETF alignment, self-duality, nearest-class-center behavior) for feature representations.
- For robustness, run the DeepFool attack per sample to estimate minimum perturbation magnitudes needed to alter class decisions; report per-class and average statistics across training.
- For generalization, train on multiclass datasets with true labels and with randomly shuffled labels to contrast NC emergence under semantic vs. non-semantic label assignments.

## Experiments
- Datasets: use standard multiclass datasets (e.g., CIFAR-10) and controlled subsets for semantic disjointness experiments (e.g., pairs such as truck vs. fish).
- Models: overparameterized convolutional or MLP classifiers appropriate for the dataset scale.
- Metrics: adversarial perturbation norms (DeepFool), NC geometry diagnostics, training/validation accuracy, and memorization checks.

## Conda Environment Setup (cross-platform)
Create and activate a Conda environment named `ncollapse`, then install dependencies. Use the included `environment.yml` to create the environment:

```bash
# create the conda environment from environment.yml
conda env create -f environment.yml

# activate the environment
conda activate ncollapse
```

If you prefer creating the environment manually, example commands:

```bash
# create env with a specific python version
conda create -n ncollapse python=3.10 -y
conda activate ncollapse

# install common packages via conda / pip
conda install -c conda-forge numpy scipy -y
pip install torch torchvision foolbox
```

## Quick Start
1. Create and activate the virtual environment (see above).
2. Prepare the dataset and configuration files for the desired experiment.
3. Run training and evaluation scripts (examples to be added):

```bash
python train.py --config configs/cifar10.yml
python evaluate_deepfool.py --checkpoint checkpoints/ckpt.pth
```

## Reproducibility & Notes
- Report seeds, hardware, and library versions when sharing results.
- DeepFool is iterative and optimization-based; ensure consistent attack parameters across checkpoints for fair comparison.

## References
- Moosavi-Dezfooli et al., DeepFool: A simple and accurate method to fool deep neural networks, 2016. \cite{moosavidezfooli2016deepfoolsimpleaccuratemethod}
- Papyan et al., Neural Collapse phenomenon and analysis, 2020. \cite{DBLP:journals/corr/abs-2008-08186}
- Kothapalli et al., Neural Collapse review and robustness modeling, 2023. \cite{kothapalli2023neuralcollapsereviewmodelling}
- Nguyen et al., Memorization dynamics in two-class NC experiments, 2023. \cite{nguyen2023memorizationdilationmodelingneuralcollapse}

---

If you'd like, I can (a) create an initial `requirements.txt` with suggested packages and pinned versions, (b) add example `train.py` and `evaluate_deepfool.py` stubs, or (c) wire CI/tests to reproduce the central experiments.
