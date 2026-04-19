import matplotlib.pyplot as plt
import numpy as np

# Data from your terminal output
rounds = ['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5']
sync_times = [4.50, 5.10, 4.80, 6.20, 5.00]
async_times = [2.85, 3.10, 3.05, 3.70, 3.05]

x = np.arange(len(rounds))  # the label locations
width = 0.35  # the width of the bars

# Configure the plot for academic paper styling
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot the bars
rects1 = ax.bar(x - width/2, sync_times, width, label='Synchronous FedAvg (Waits for Straggler)', color='#d62728', edgecolor='black')
rects2 = ax.bar(x + width/2, async_times, width, label='Asynchronous ADMM (Continuous Processing)', color='#2ca02c', edgecolor='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Network Processing Time (Seconds)', fontsize=12, fontweight='bold')
ax.set_xlabel('Global Communication Round', fontsize=12, fontweight='bold')
ax.set_title('Straggler Bottleneck: Synchronous vs Asynchronous Processing Times', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(rounds, fontsize=11)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Function to attach a text label above each bar, displaying its height
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

# Save and show
plt.tight_layout()
filename = 'straggler_simulation_plot.png'
plt.savefig(filename)
print(f"Graph successfully saved as {filename}")

plt.show()