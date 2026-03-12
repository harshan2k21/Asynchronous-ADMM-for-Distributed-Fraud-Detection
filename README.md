
---

# Asynchronous ADMM for Distributed Fraud Detection

A privacy-preserving, bottleneck-resistant Federated Learning pipeline built from scratch. This project implements an **Asynchronous Alternating Direction Method of Multipliers (ADMM)** architecture to train a distributed logistic regression model on highly imbalanced click-stream data, actively mitigating the "straggler effect" found in standard federated networks.

## 🚀 Key Features & Engineering Highlights

Unlike standard federated learning implementations (e.g., synchronous FedAvg), this system is engineered for real-world, heterogeneous edge networks:

* **100% Asynchronous Server:** The central Flask parameter server never blocks or waits for slow devices. It accepts HTTP updates continuously, allowing fast nodes to iterate without being throttled by network stragglers.
* **Mathematical Straggler Penalty:** Solves the "stale gradient" problem by calculating a temporal delay ($\tau$) for incoming updates. The server applies a time-decay factor to the Augmented Lagrangian penalty ($\rho$), gracefully merging outdated information without corrupting the global consensus model.
* **Feature Hashing (The Hashing Trick):** Fixes the categorical matrix mismatch problem inherent in decentralized data. Maps isolated, disjointed categorical variables (like unique IP addresses or Device IDs) into a strict, fixed-length 50-dimensional space across all nodes, preventing server-side aggregation crashes without data leakage.
* **Built from Scratch:** The custom Augmented Lagrangian gradient descent, dual-variable tracking, and communication layers are implemented purely in NumPy and Python/Flask, without relying on black-box federated learning frameworks.

## 🏗️ System Architecture

* **Central Parameter Server (`central_server.py`):** A stateless Flask API that maintains the global mathematical state ($z$). It asynchronously aggregates local weights, calculates the time-decayed penalty for stragglers, and updates dual variables ($y_i$).
* **Local Client Nodes (`local_client.py`, `local_client_2.py`):** Independent edge nodes that shard the dataset, apply Feature Hashing, and run a custom Gradient Descent optimizer to minimize Binary Cross-Entropy (BCE) loss alongside the Augmented Lagrangian constraints.

## 📊 Dataset & Performance

This architecture was validated using the **TalkingData AdTracking Fraud Detection** dataset.

* **Data Partitioning:** The dataset was horizontally sharded to simulate strict data isolation between Node 1 and Node 2.
* **Results:** The asynchronous network successfully reached mathematical convergence (loss minimization) across 5 global communication rounds, maintaining predictive parity with baseline ROC-AUC scores in the mid-0.60s on highly imbalanced, isolated test sets.

## 💻 Installation and Usage

### Prerequisites

* Python 3.8+
* Libraries: `numpy`, `pandas`, `scikit-learn`, `flask`, `requests`

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/asynchronous-admm-fraud.git
cd asynchronous-admm-fraud

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Add the dataset:**
Download the `train_sample.csv` from the TalkingData Kaggle competition and place it in the root directory.

### Running the Distributed Network

To simulate the asynchronous network, you will need to open three separate terminal windows to run the server and the two isolated nodes simultaneously.

**Terminal 1 (Start the Central Server):**

```bash
python central_server.py

```

**Terminal 2 (Start Node 1):**

```bash
python local_client.py

```

**Terminal 3 (Start Node 2):**

```bash
python local_client_2.py

```

Watch the server terminal as it asynchronously receives HTTP POST requests, applies the temporal decay penalty to whichever node is slower, and mathematically updates the global consensus model on the fly.

## 🛠️ Future Work

Based on current literature in distributed optimization, future iterations of this architecture could explore:

* **Quantized ADMM (QADMM):** Implementing a compression operator to transmit only the *change* in iterates rather than the absolute arrays, drastically reducing communication bandwidth.
* **Invariant Dropout (FLuID):** Dynamically reducing the sub-model size for known straggler nodes to keep them aligned with the global training pace.

---

*Developed for research and demonstration in privacy-preserving distributed machine learning.*
