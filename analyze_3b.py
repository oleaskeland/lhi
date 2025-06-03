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
    match = re.search(r'3B_(\d+\.?\d*)_', path)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract h from path: {path}")

def predict(model, X_test_tensor, h, steps, HNN=False):
    if HNN:
        return model.predict(X_test_tensor[0], h, steps=steps - 1, keepinitx=True, returnnp=True)
    if hasattr(model, 'predict'):
        return model.predict(X_test_tensor[0], steps=steps - 1, keepinitx=True, returnnp=True)
    else:
        raise ValueError("Model does not support .predict() method.")

def compute_energy(traj):
    q = traj[:, 6:]
    p = traj[:, :6]

    p1, p2, p3 = p[:, :2], p[:, 2:4], p[:, 4:6]
    q1, q2, q3 = q[:, :2], q[:, 2:4], q[:, 4:6]

    kinetic = 0.5 * (np.sum(p1**2, axis=1) + np.sum(p2**2, axis=1) + np.sum(p3**2, axis=1))

    r12 = np.linalg.norm(q1 - q2, axis=1)
    r13 = np.linalg.norm(q1 - q3, axis=1)
    r23 = np.linalg.norm(q2 - q3, axis=1)

    potential = -1.0 / r12 - 1.0 / r13 - 1.0 / r23
    return kinetic + potential

def compute_global_error(pred, true):
    return np.mean((pred - true)**2, axis=1)

def analyze_threebody(G_path, HNN_path, LHI_path):
    model_paths = {'G-SympNet': G_path, 'HNN': HNN_path, 'LHI': LHI_path}
    results = {}

    plt.figure(figsize=(16, 10))

    for idx, (label, path) in enumerate(model_paths.items()):
        model, X_test, y_test = load_model_and_data(path)
        h = extract_h_from_path(path)
        input = torch.tensor(y_test, dtype=torch.float32)

        pred = predict(model, input, h, len(y_test), HNN=(label == 'HNN'))

        energy_pred = compute_energy(pred)
        energy_true = compute_energy(y_test)
        error = compute_global_error(pred, y_test)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[label] = {"error": error, "params": params}

        # Plot 1st body (q1) trajectory
        plt.subplot(2, 2, 1)
        plt.plot(pred[:, 6], pred[:, 7], label=label, alpha=0.7)
        if idx == 0:
            plt.plot(y_test[:, 6], y_test[:, 7], linestyle='--', label="Ground Truth", color='black')
        plt.title("Body 1 (x-y) Trajectory")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend()

        # Plot energy
        plt.subplot(2, 2, 2)
        plt.plot(energy_pred, label=label)
        if idx == 0:
            plt.plot(energy_true, label="Ground Truth", linestyle="--", color="black")
        plt.title("Total Energy")
        plt.xlabel("Step"); plt.ylabel("H")
        plt.legend()

        # Plot log MSE over time
        plt.subplot(2, 2, 3)
        plt.plot(error, label=f"{label} ({params} params)")
        plt.yscale("log")
        plt.title("Trajectory MSE")
        plt.xlabel("Step"); plt.ylabel("MSE")
        plt.legend()

    plt.tight_layout()
    plt.savefig("threebody_analysis_panel.png")

    print("\nAverage MSE per model:")
    for label in results:
        avg = np.mean(results[label]["error"])
        print(f"{label}: {avg:.6e}")

if __name__ == '__main__':
    base_path = "/Users/oleaskeland/PycharmProjects/LHI/outputs/"

    analyze_threebody(
        G_path=base_path + "3B_0.5_G",
        HNN_path=base_path + "3B_0.5_HNN",
        LHI_path=base_path + "3B_0.5_LHI"
    )
