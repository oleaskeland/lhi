import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import re


def load_model_and_data(path):
    model = torch.load(os.path.join(path, 'model_best.pkl'), weights_only=False)
    X_test = np.loadtxt(os.path.join(path, 'X_test.txt'))
    y_test = np.loadtxt(os.path.join(path, 'y_test.txt'))
    return model, X_test, y_test


def extract_h_from_path(path):
    match = re.search(r'pendulum_(\d+\.?\d*)', path)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract h from path: {path}")


def predict(model, X_test_tensor, h, steps, HNN = False):

    if HNN:
        return model.predict(X_test_tensor[0], h, steps = steps - 1, keepinitx=True, returnnp=True)
    if hasattr(model, 'predict'):
        return model.predict(X_test_tensor[0], steps - 1, keepinitx=True, returnnp=True)

    else:
        raise ValueError("Model does not support .predict() method.")


def compute_energy(traj):
    # Standard Hamiltonian for simple pendulum: H = p^2/2 - cos(q)
    p, q = traj[:, 0], traj[:, 1]
    return 0.5 * p**2 - np.cos(q)

def compute_global_error(pred, true):
    return np.mean((pred - true)**2, axis=1)

def analyze_models(G_paths, HNN_paths, LHI_paths):
    n = len(G_paths)

    # Track total MSEs
    total_mse = {'G-SympNet': 0.0, 'HNN': 0.0, 'LHI': 0.0}
    total_counts = {'G-SympNet': 0, 'HNN': 0, 'LHI': 0}

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    for i, (g, hnn, lhi) in enumerate(zip(G_paths, HNN_paths, LHI_paths)):
        axes[i, 0].text(-0.25, 0.5, f'Problem_id = {i+1}', transform=axes[i, 0].transAxes,
                        rotation=90, fontsize=12, va='center', ha='center', fontweight='bold')

        for j, (label_base, path) in enumerate(zip(['G-SympNet', 'HNN', 'LHI'], [g, hnn, lhi])):
            h = extract_h_from_path(path)
            model, X_test, y_test = load_model_and_data(path)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            label = f"{label_base}"

            h_steps = len(y_test)
            input = torch.tensor(y_test, dtype=torch.float32)
            if label_base == 'HNN':
                pred = predict(model, input, h, h_steps, HNN=True)
            else:
                pred = predict(model, input, h, h_steps)

            true = y_test
            energy = compute_energy(pred)
            true_energy = compute_energy(true)
            error = compute_global_error(pred, true)

            # Accumulate total squared error
            total_mse[label] += np.sum((pred - true) ** 2)
            total_counts[label] += np.prod(true.shape)

            start_idx = int(0.7 * len(true))
            if h == 0.5:
                start_idx = int(0.95 * len(true))
            if h == 0.1:
                start_idx = int(0.3 * len(true))

            if j == 0:
                axes[i, 0].plot(np.arange(start_idx, len(true)), true[start_idx:, 1],
                                label='Ground Truth', linestyle='--', color="black")
                axes[i, 1].plot(np.arange(start_idx, len(true)), true_energy[start_idx:], label='Ground Truth', linestyle='--', color="black")
            axes[i, 0].plot(np.arange(start_idx, len(true)), pred[start_idx:, 1],
                            label=f"{label} ({params} params)", alpha=0.6)

            axes[i, 1].plot(np.arange(start_idx, len(true)), energy[start_idx:], label=label)
            axes[i, 2].plot(error, label=f"{label} ({params} params)")
            axes[i, 2].set_yscale('log')

            axes[i, 0].set_xlim(start_idx, len(true))
            axes[i, 1].set_xlim(start_idx, len(true))

    for col, title in zip(range(3), ["Pendulum Position", "Total Energy H(t)", "Global Trajectory Error (MSE)"]):
        axes[0, col].set_title(title)

    for ax_row in axes:
        for ax in ax_row:
            ax.legend()

    plt.tight_layout()
    plt.savefig("pendulum_analysis_panel.png", dpi=300)

    # Print average MSE per model
    print("\nAverage MSE per model across all problems:")
    for label in total_mse:
        avg_mse = total_mse[label] / total_counts[label]
        print(f"{label}: {avg_mse:.6e}")




if __name__ == '__main__':
    # MANUALLY specify paths to model output folders here

    base_path = "/Users/oleaskeland/PycharmProjects/LHI/"
    G_paths = [
        base_path + 'outputs/pendulum_1_G', base_path + 'outputs/pendulum_0.1_G', base_path + 'outputs/pendulum_0.5_G'
        # Add more paths if needed
    ]
    HNN_paths = [
        base_path + 'outputs/pendulum_1_HNN', base_path + 'outputs/pendulum_0.1_HNN',base_path + 'outputs/pendulum_0.5_HNN'
        # Add more paths if needed
    ]
    LHI_paths = [
        base_path + 'outputs/pendulum_1_LHI', base_path + 'outputs/pendulum_0.1_LHI', base_path + 'outputs/pendulum_0.5_LHI'
        # Add more paths if needed
    ]

    analyze_models(G_paths, HNN_paths, LHI_paths)
