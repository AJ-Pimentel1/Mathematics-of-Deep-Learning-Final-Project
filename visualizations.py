import os
import pandas as pd
import matplotlib.pyplot as plt

# Set matplotlib style for academic publishing
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'grid.alpha': 0.4
})

DATASETS = ['mnist', 'fmnist', 'cifar10']
DATASET_TITLES = {'mnist': 'MNIST', 'fmnist': 'Fashion-MNIST', 'cifar10': 'CIFAR-10'}

NC_METRICS = {
    'NC1': 'Variability Collapse (NC1)',
    'NC2': 'Convergence to Simplex ETF (NC2)',
    'NC3': 'Convergence to Self-Duality (NC3)',
    'NC4': 'Nearest Class-Center Error (NC4)'
}

def stitch_base_and_tpt(dataset):
    """Loads and concatenates the base and TPT CSVs for a continuous timeline."""
    base_df = pd.read_csv(f'results/{dataset}_base_training_history.csv')
    tpt_df = pd.read_csv(f'results/{dataset}_tpt_history.csv')
    
    zero_error_epoch = base_df['Epoch'].max()
    
    tpt_df['Total_Epoch'] = tpt_df['Epoch'] + zero_error_epoch
    base_df['Total_Epoch'] = base_df['Epoch']
    
    return base_df, tpt_df, zero_error_epoch

def plot_individual_robustness():
    print("Generating Individual Robustness Curves...")
    for dataset in DATASETS:
        try:
            tpt_df = pd.read_csv(f'results/{dataset}_tpt_history.csv')
            tpt_df['Rho_Adv'] = tpt_df['Rho_Adv'].ffill()
            
            plt.figure(figsize=(8, 6))
            plt.plot(tpt_df['Epoch'], tpt_df['Rho_Adv'], color='#2ca02c')
            
            plt.title(f"{DATASET_TITLES[dataset]} - Adversarial Robustness")
            plt.xlabel('TPT Epoch (Post 0-Error)')
            plt.ylabel('$\\hat{\\rho}_{adv}$ (Perturbation Norm)')
            
            plt.tight_layout()
            plt.savefig(f'figures/{dataset}_robustness_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        except FileNotFoundError:
            print(f"  -> Missing robustness data for {dataset}")

def plot_individual_classic_nc():
    print("Generating Individual Base + TPT NC Metric Curves...")
    for dataset in DATASETS:
        try:
            base_df, tpt_df, zero_err_ep = stitch_base_and_tpt(dataset)
            total_epochs = pd.concat([base_df['Total_Epoch'], tpt_df['Total_Epoch']])
            
            for metric, title in NC_METRICS.items():
                metric_total = pd.concat([base_df[metric], tpt_df[metric]])
                
                plt.figure(figsize=(8, 6))
                plt.plot(total_epochs, metric_total, color='#1f77b4')
                
                # The iconic red line for 0 Error
                plt.axvline(x=zero_err_ep, color='#d62728', linestyle='--', linewidth=2)
                plt.text(zero_err_ep + (total_epochs.max() * 0.02), plt.ylim()[1] * 0.8, 
                        '0 Error\n$\\longrightarrow$', color='#d62728', fontsize=14, fontweight='bold')
                
                plt.title(f"{DATASET_TITLES[dataset]} - {title}")
                plt.xlabel('Epoch')
                plt.ylabel(metric)
                
                plt.tight_layout()
                plt.savefig(f'figures/{dataset}_classic_{metric}.png', dpi=300, bbox_inches='tight')
                plt.close()
        except FileNotFoundError:
            print(f"  -> Missing classic NC data for {dataset}")

def plot_individual_shuffled_nc():
    print("Generating Individual Shuffled Label Curves...")
    for dataset in DATASETS:
        try:
            shuff_df = pd.read_csv(f'results/{dataset}_shuffled_training_history.csv')
            
            for metric, title in NC_METRICS.items():
                plt.figure(figsize=(8, 6))
                plt.plot(shuff_df['Epoch'], shuff_df[metric], color='#ff7f0e')
                
                plt.title(f"{DATASET_TITLES[dataset]} (Shuffled Labels) - {title}")
                plt.xlabel('Epoch')
                plt.ylabel(metric)
                
                plt.tight_layout()
                plt.savefig(f'figures/{dataset}_shuffled_{metric}.png', dpi=300, bbox_inches='tight')
                plt.close()
        except FileNotFoundError:
            print(f"  -> Missing shuffled data for {dataset}")

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    plot_individual_robustness()
    plot_individual_classic_nc()
    plot_individual_shuffled_nc()
    print("\nSuccess! All 27 individual visualizations saved to the 'figures' directory.")