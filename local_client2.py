import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import time

def sigmoid(z):
    """Activation function for logistic regression."""
    # Clip values to prevent overflow in np.exp
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))

def load_and_preprocess_data(filepath, num_samples=5000, n_hash_features=50):
    """Loads TalkingData CSV and uses Feature Hashing to handle categorical IDs."""
    print(f"Loading data from {filepath}...")
    
    df = pd.read_csv(filepath, skiprows=range(1, 10001), nrows=num_samples)
    df['hour'] = pd.to_datetime(df['click_time']).dt.hour
    
    feature_cols = ['ip', 'app', 'device', 'os', 'channel', 'hour']
    
    # 1. Convert columns to strings with prefixes so the hasher knows "app_1" is different from "os_1"
    for col in feature_cols:
        df[col] = col + "_" + df[col].astype(str)
        
    # 2. Convert the dataframe into a list of dictionaries (required by FeatureHasher)
    features_dict = df[feature_cols].to_dict(orient='records')
    
    # 3. Apply Feature Hashing to guarantee a fixed feature space for all clients
    hasher = FeatureHasher(n_features=n_hash_features, input_type='dict')
    X_hashed = hasher.transform(features_dict).toarray()
    
    y = df['is_attributed'].values
    
    # 4. Add the Bias Column (Intercept) to prevent the 0.6931 saddle point trap
    X_final = np.c_[np.ones(X_hashed.shape[0]), X_hashed]
    
    # Return the new matrix, labels, and the exact total number of features (hash size + 1 for bias)
    return X_final, y, n_hash_features + 1

class LocalClient:
    def __init__(self, client_id, num_features):
        self.client_id = client_id
        self.local_timestamp = 0
        
        # ML Parameters
        self.num_features = num_features
        self.x = np.zeros(num_features)       # Local weights
        self.y = np.zeros(num_features)       # Dual variable (Lagrange multiplier)
        self.z = np.zeros(num_features)       # Global model from server
        self.rho = 1.0                        # Base penalty
        self.learning_rate = 0.01             # For Gradient Descent

    def train_local(self, X_train, y_train, epochs=50):
        """
        Custom Gradient Descent minimizing the Augmented Lagrangian,
        now with Loss Tracking.
        """
        m = len(y_train)
        print(f"[{self.client_id}] Starting local training for {epochs} epochs...")

        for epoch in range(epochs):
            # 1. Forward pass (Predictions)
            predictions = sigmoid(np.dot(X_train, self.x))
            
            # --- NEW: Calculate Total Loss ---
            # Add a tiny epsilon to np.log to prevent log(0) errors
            epsilon = 1e-15
            bce_loss = -np.mean(y_train * np.log(predictions + epsilon) + (1 - y_train) * np.log(1 - predictions + epsilon))
            
            # ADMM penalty terms: y^T(x - z) + (rho/2)||x - z||^2
            linear_penalty = np.dot(self.y, (self.x - self.z))
            quad_penalty = (self.rho / 2) * np.sum(np.square(self.x - self.z))
            
            total_loss = bce_loss + linear_penalty + quad_penalty
            
            # Print progress every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"[{self.client_id}] Epoch {epoch:02d} | Loss: {total_loss:.4f} (BCE: {bce_loss:.4f})")
            # ---------------------------------
            
            # 2. Compute the base gradient from the data (BCE Gradient)
            gradient_bce = np.dot(X_train.T, (predictions - y_train)) / m
            
            # 3. Add the ADMM Penalty Gradients
            gradient_penalty = self.y + self.rho * (self.x - self.z)
            
            # 4. Total Gradient
            total_gradient = gradient_bce + gradient_penalty
            
            # 5. Update weights
            self.x -= self.learning_rate * total_gradient

        print(f"[{self.client_id}] Training complete. Ready to push.")

    def push_update(self, server_url):
        payload = {
            "client_id": self.client_id,
            "x": self.x.tolist(), 
            "timestamp": self.local_timestamp
        }
        try:
            response = requests.post(f"{server_url}/push_update", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"[{self.client_id}] Server Response: {data['message']}")
                self.z = np.array(data['new_z'])
                self.y = np.array(data['new_y'])
                self.local_timestamp = data['global_timestamp']
            else:
                print(f"[{self.client_id}] Server Error ({response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"[{self.client_id}] Error: Could not connect to the server.")

    def evaluate(self, X_test, y_test):
        """Calculates the ROC-AUC score on unseen test data."""
        # Generate probabilities
        probabilities = sigmoid(np.dot(X_test, self.x))
        
        # Calculate ROC-AUC
        try:
            auc_score = roc_auc_score(y_test, probabilities)
            print(f"\n[{self.client_id}] Final Evaluation | ROC-AUC Score: {auc_score:.4f}")
        except ValueError:
            print(f"\n[{self.client_id}] Error: ROC-AUC requires both positive and negative classes in test set.")        

# --- EXECUTION BLOCK ---
if __name__ == '__main__':
    CSV_PATH = "train_sample.csv" 
    
    try:
        # Load the data
        X_data, y_data, num_features = load_and_preprocess_data(CSV_PATH, num_samples=10000)
        
        # Split into 80% training and 20% testing
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        
        client = LocalClient(client_id="node_1", num_features=num_features)
        SERVER_URL = "http://127.0.0.1:5000"
        
        # --- THE COMMUNICATION LOOP ---
        ROUNDS = 5
        for round_num in range(ROUNDS):
            print(f"\n=== COMMUNICATION ROUND {round_num + 1}/{ROUNDS} ===")
            client.train_local(X_train, y_train, epochs=50)
            client.push_update(SERVER_URL)
            
        print("\n[node_1] All communication rounds completed successfully.")
        
        # --- FINAL EVALUATION ---
# Assuming y_test are your true labels and X_test is your hashed test matrix
        predictions = 1 / (1 + np.exp(-np.dot(X_test, local_x))) # Sigmoid probabilities
        binary_predictions = (predictions >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, predictions)
        pr_auc = average_precision_score(y_test, predictions)
        f1 = f1_score(y_test, binary_predictions)

        print("\n=== FINAL METRICS ===")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"PR-AUC Score:  {pr_auc:.4f}")
        print(f"F1-Score:      {f1:.4f}")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{CSV_PATH}'.")