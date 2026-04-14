import numpy as np
import matplotlib.pyplot as plt
import json
import os

def generate_evaluation_report(results, experiment_name, output_dir="."):
    """
    Generates a visual report (PNG) and a JSON summary from evaluation results.
    """
    returns = np.array(results["returns"])
    max_tiles = np.array(results["max_tiles"])
    steps = np.array(results["steps"])
    num_episodes = len(returns)

    # 1. Calculate Tile Distribution
    # Standard 2048 milestones
    milestones = [128, 256, 512, 1024, 2048, 4096]
    reach_counts = {m: np.sum(max_tiles >= m) for m in milestones}
    reach_rates = {m: (count / num_episodes) * 100 for m, count in reach_counts.items()}

    # 2. Create Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Ablation Study Report: {experiment_name}\n({num_episodes} evaluation games)", fontsize=22, fontweight='bold')

    # A. Tile Reach Rates (Bar Chart)
    labels = [str(m) for m in milestones]
    rates = [reach_rates[m] for m in milestones]
    
    # Use a nice color gradient for tiles
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(milestones)))
    
    axes[0, 0].bar(labels, rates, color=colors, edgecolor='black')
    axes[0, 0].set_title("Tile Milestone Reach Rate (%)", fontsize=16)
    axes[0, 0].set_ylabel("Percentage of Games", fontsize=14)
    axes[0, 0].set_ylim(0, 105)
    for i, rate in enumerate(rates):
        axes[0, 0].text(i, rate + 1, f"{rate:.1f}%", ha='center', fontweight='bold')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # B. Score Distribution (Histogram)
    axes[0, 1].hist(returns, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(returns), color='red', linestyle='dashed', linewidth=2, label=f"Mean: {np.mean(returns):.0f}")
    axes[0, 1].set_title("Final Score Distribution", fontsize=16)
    axes[0, 1].set_xlabel("Score", fontsize=14)
    axes[0, 1].set_ylabel("Game Count", fontsize=14)
    axes[0, 1].legend()

    # C. Steps Distribution (Survival Time)
    axes[1, 0].hist(steps, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(np.mean(steps), color='blue', linestyle='dashed', linewidth=2, label=f"Mean: {np.mean(steps):.0f}")
    axes[1, 1].axis('off') # We'll use this space for textual summary
    axes[1, 0].set_title("Steps per Game Distribution", fontsize=16)
    axes[1, 0].set_xlabel("Survival Steps", fontsize=14)
    axes[1, 0].set_ylabel("Game Count", fontsize=14)
    axes[1, 0].legend()

    # D. Textual Summary (Using axes[1, 1])
    summary_text = (
        f"--- Performance Summary ---\n\n"
        f"Mean Score: {np.mean(returns):.1f} ± {np.std(returns):.1f}\n"
        f"Max Score Achieved: {np.max(returns):.0f}\n"
        f"Mean Steps: {np.mean(steps):.1f}\n"
        f"Max Steps (Survival): {np.max(steps)}\n\n"
        f"Max Tile Achieved: {np.max(max_tiles)}\n"
        f"2048 Success Rate: {reach_rates.get(2048, 0):.2f}%\n"
        f"1024 Success Rate: {reach_rates.get(1024, 0):.2f}%\n"
        f"512 Success Rate: {reach_rates.get(512, 0):.2f}%\n"
    )
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=14, family='monospace', verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save files
    image_path = os.path.join(output_dir, f"{experiment_name}_report.png")
    json_path = os.path.join(output_dir, f"{experiment_name}_stats.json")
    
    plt.savefig(image_path, dpi=120)
    plt.close()
    
    # JSON export for raw data
    summary_json = {
        "experiment": experiment_name,
        "metrics": {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "max_return": float(np.max(returns)),
            "mean_steps": float(np.mean(steps)),
            "max_steps": int(np.max(steps)),
            "mean_max_tile": float(np.mean(max_tiles)),
            "max_tile_overall": int(np.max(max_tiles))
        },
        "reach_rates": reach_rates
    }
    
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=4)
        
    print(f"Report generated: {image_path}")
    print(f"Stats exported: {json_path}")
