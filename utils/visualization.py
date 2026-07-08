# utils/visualization.py
import matplotlib.pyplot as plt

def plot_convergence(train_losses, val_costs, save_path="convergence.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses)
    ax1.set_title("Training Loss")
    ax2.plot(val_costs)
    ax2.set_title("Validation Cost")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_metrics(train_best_costs, train_mean_costs, val_costs, save_path="training_metrics.png"):
    plt.figure(figsize=(10, 6))
    if train_best_costs:
        plt.plot(train_best_costs, label="Train Best Cost", linewidth=2)
    if train_mean_costs:
        plt.plot(train_mean_costs, label="Train Mean Cost", linewidth=2)
    if val_costs:
        plt.plot(val_costs, label="Validation Best Cost", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_route(coords, route, depot, save_path="route.png"):
    plt.figure(figsize=(8,8))
    pts = coords.cpu().numpy()
    plt.scatter(pts[:,0], pts[:,1], c='gray', alpha=0.5)
    plt.scatter(depot[0], depot[1], c='red', marker='s', s=100)
    for i, node in enumerate(route):
        if i < len(route)-1:
            n1, n2 = pts[node], pts[route[i+1]]
            plt.plot([n1[0], n2[0]], [n1[1], n2[1]], 'b-', alpha=0.7)
    plt.title(f"Route with {len(route)} nodes")
    plt.savefig(save_path, dpi=300)
    plt.close()
