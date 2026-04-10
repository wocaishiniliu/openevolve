"""Plot Pareto curve: MPC cost vs F1 for all methods."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Baselines (7-dataset averages from latest results)
baselines = {
    "All-1B": (0.222, 33.84),
    "All-Qwen1.5B": (0.255, 38.98),
    "All-3B": (0.523, 44.07),
    "MPCache 7B": (0.802, 37.97),
    "All-7B": (1.000, 41.46),
    "All-8B": (1.107, 50.07),
}

# Router Pareto points (4-dataset, from budget test)
router_4ds = {
    0.70: (0.373, 44.32),
    0.80: (0.459, 44.51),
    0.85: (0.584, 48.51),
    0.90: (0.582, 49.16),
    0.95: (0.721, 51.59),
    1.00: (0.845, 52.66),
}

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Plot baselines
for name, (cost, f1) in baselines.items():
    color = 'gray' if 'MPCache' not in name else 'red'
    marker = 's' if 'MPCache' in name else 'o'
    ax.scatter(cost, f1, s=100, marker=marker, color=color, zorder=5)
    offset = (0.02, 0.8)
    if name == "All-1B":
        offset = (0.02, -1.5)
    elif name == "All-Qwen1.5B":
        offset = (0.02, 0.8)
    elif name == "MPCache 7B":
        offset = (-0.15, -2.0)
    ax.annotate(name, (cost, f1), textcoords="offset points",
                xytext=(offset[0]*50, offset[1]*5), fontsize=9)

# Plot router Pareto curve
r_costs = [router_4ds[qt][0] for qt in sorted(router_4ds.keys())]
r_f1s = [router_4ds[qt][1] for qt in sorted(router_4ds.keys())]
ax.plot(r_costs, r_f1s, 'b-o', markersize=8, linewidth=2, label='Router (ours)', zorder=6)
for qt in router_4ds:
    cost, f1 = router_4ds[qt]
    ax.annotate(f"qt={qt}", (cost, f1), textcoords="offset points",
                xytext=(5, -12), fontsize=7, color='blue')

# Budget lines
ax.axvline(x=0.523, color='green', linestyle='--', alpha=0.5, label='Budget-A (3B cost)')
ax.axvline(x=0.802, color='orange', linestyle='--', alpha=0.5, label='Budget-B (MPCache cost)')

ax.set_xlabel('MPC Cost (relative to 7B)', fontsize=12)
ax.set_ylabel('Average F1 Score', fontsize=12)
ax.set_title('Cost-Quality Pareto Curve: Router vs Baselines', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.1, 1.2)
ax.set_ylim(30, 58)

plt.tight_layout()
plt.savefig('/home/yu505948.ucf/work/MPCache_project/MPCache_openevolve/experiments/pareto_curve.png', dpi=150)
print("Saved to experiments/pareto_curve.png")
