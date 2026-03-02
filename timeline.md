# Neural Collapse: Project Timeline & Milestones

## Phase 1: Core Implementation & Baselines
- [x] **March 2:** Base training script complete.
  - *Details:* Data loaders, ResNet18 architecture, and NC metrics engine (NC1, NC2, NC3, NC4) integrated and functional.
- [ ] **March 5:** Base models trained to 100% accuracy.
  - *Details:* Train 3 distinct random initializations for each of the 3 datasets (CIFAR-10, MNIST, Fashion-MNIST).
- [ ] **March 10:** Adversarial robustness testing complete.
  - *Details:* Calculate minimal DeepFool perturbations at each phase leading up to Neural Collapse to replicate robustness findings.

## Phase 2: Memorization vs. Generalization
- [ ] **March 21:** Dataset pipeline modified to shuffle labels on all datasets.
- [ ] **March 22:** Training complete for the shuffled datasets.
  - *Details:* Pushing the overparameterized model to pure memorization to observe if geometric collapse still occurs identically.

## Phase 3: Writing & Deliverables
- [ ] **March 27:** Base literature review on Neural Collapse complete. All experiment figures generated.
  - *Details:* Finalize plots for intra-class variance, Simplex ETF geometry, and self-duality.
- [ ] **April 1:** All experimental results and metric calculations finalized.
- [ ] **April 4:** Rough draft of the paper assembled with integrated figures.
- [ ] **April 8:** Final report polished and submitted to professor.
- [ ] **April 12:** Presentation slides finalized. Clean up repository.