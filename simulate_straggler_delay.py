import numpy as np

# Simulate training times (in seconds) for 5 rounds
# Node 1 is a fast edge device, Node 2 is a "straggler" with a poor connection/hardware
node_1_times = [1.2, 1.1, 1.3, 1.2, 1.1] 
node_2_times = [4.5, 5.1, 4.8, 6.2, 5.0]

sync_round_times = []
async_round_times = []

print("=== STRAGGLER BOTTLENECK SIMULATION (5 ROUNDS) ===")
for i in range(5):
    # Synchronous FedAvg MUST wait for the slowest node
    sync_time = max(node_1_times[i], node_2_times[i])
    sync_round_times.append(sync_time)
    
    # Asynchronous ADMM processes immediately (Average throughput time)
    async_time = np.mean([node_1_times[i], node_2_times[i]])
    async_round_times.append(async_time)
    
    print(f"Round {i+1} | Sync FedAvg Wait: {sync_time:.2f}s | Async ADMM Throughput: {async_time:.2f}s")

total_sync_time = sum(sync_round_times)
total_async_time = sum(async_round_times)

print("-" * 50)
print(f"Total Synchronous Network Time:  {total_sync_time:.2f} seconds")
print(f"Total Asynchronous Network Time: {total_async_time:.2f} seconds")
print(f"Time Saved by Mitigating Straggler: {total_sync_time - total_async_time:.2f} seconds")