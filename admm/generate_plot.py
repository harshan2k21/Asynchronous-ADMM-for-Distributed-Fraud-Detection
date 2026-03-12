import matplotlib.pyplot as plt
import numpy as np

# Exact Total Loss values from your terminal output (Epochs 0, 10, 20, 30, 40, 49 across 5 rounds)
node_1_loss = [
    # Round 1
    0.6931, 0.6461, 0.6118, 0.5867, 0.5684, 0.5561,
    # Round 2
    0.4887, 0.4581, 0.4357, 0.4191, 0.4070, 0.3988,
    # Round 3
    0.3875, 0.3670, 0.3518, 0.3405, 0.3321, 0.3264,
    # Round 4
    0.3179, 0.3037, 0.2930, 0.2850, 0.2789, 0.2748,
    # Round 5
    0.2682, 0.2578, 0.2500, 0.2441, 0.2396, 0.2365
]

node_2_loss = [
    # Round 1
    0.6931, 0.6457, 0.6112, 0.5860, 0.5676, 0.5553,
    # Round 2
    0.6279, 0.4916, 0.3913, 0.3172, 0.2621, 0.2246,
    # Round 3
    0.3219, 0.2848, 0.2568, 0.2357, 0.2196, 0.2085,
    # Round 4
    0.2317, 0.2207, 0.2123, 0.2059, 0.2010, 0.1975,
    # Round 5
    0.1952, 0.1919, 0.1892, 0.1872, 0.1857, 0.1846
]

# Generate the x-axis (communication intervals)
x_axis = np.arange(len(node_1_loss))

# Configure the plot for academic paper styling (high DPI, clear fonts)
plt.figure(figsize=(10, 6), dpi=300)

# Plot the lines
plt.plot(x_axis, node_1_loss, marker='o', linestyle='-', linewidth=2, markersize=4, label='Node 1 (Rows 1-10k)', color='#1f77b4')
plt.plot(x_axis, node_2_loss, marker='s', linestyle='-', linewidth=2, markersize=4, label='Node 2 (Rows 10k-20k)', color='#ff7f0e')

# Add vertical lines to show where the Communication Rounds (Global Aggregations) happened
for i in range(6, len(node_1_loss), 6):
    plt.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5)

# Labels and Title
plt.title('Convergence of Asynchronous ADMM on Distributed Nodes', fontsize=14, fontweight='bold')
plt.xlabel('Training Steps (Local Epochs + Global Aggregation)', fontsize=12)
plt.ylabel('Augmented Lagrangian Loss (Objective Value)', fontsize=12)

# Styling
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()

# Save the figure
filename = 'convergence_plot.png'
plt.savefig(filename)
print(f"Graph successfully saved as {filename}")

# Display it on screen
plt.show()