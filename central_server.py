import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

class CentralServer:
    def __init__(self, num_features, base_rho, alpha):
        self.num_features = num_features
        self.z = np.zeros(num_features)
        self.base_rho = base_rho
        self.alpha = alpha
        self.global_timestamp = 0
        self.client_states = {} # Stores x, y, and rho_hat for each client

    def calculate_decay_penalty(self, local_timestamp):
        tau = max(0, self.global_timestamp - local_timestamp)
        return self.base_rho * np.exp(-self.alpha * tau)

    def receive_update(self, client_id, client_x, local_timestamp):
        # 1. Calculate dynamic penalty for straggler mitigation
        rho_hat = self.calculate_decay_penalty(local_timestamp)
        
        # Initialize client state if it's their first time connecting
        if client_id not in self.client_states:
            self.client_states[client_id] = {
                'x': np.zeros(self.num_features),
                'y': np.zeros(self.num_features),
                'rho_hat': rho_hat
            }
        
        # 2. Update Server's record of Client's weights (x)
        self.client_states[client_id]['x'] = client_x
        self.client_states[client_id]['rho_hat'] = rho_hat
        
        # 3. Update the Dual Variable (y) for this client
        self.client_states[client_id]['y'] += rho_hat * (client_x - self.z)
        
        # 4. Aggregate to find the new Global Consensus Model (z)
        self.update_global_z()
        
        # Increment global clock
        self.global_timestamp += 1
        
        # Return the new global model and the client's updated dual variable
        return self.z, self.client_states[client_id]['y']

    def update_global_z(self):
        """
        Minimizes the Augmented Lagrangian with respect to z across all known clients.
        """
        numerator = np.zeros(self.num_features)
        denominator = 0.0
        
        for state in self.client_states.values():
            numerator += (state['rho_hat'] * state['x']) + state['y']
            denominator += state['rho_hat']
            
        if denominator > 0:
            self.z = numerator / denominator

# Initialize server for 7 features (6 data features + 1 bias column)
admm_server = CentralServer(num_features=51, base_rho=1.0, alpha=0.1)
@app.route('/push_update', methods=['POST'])
def handle_update():
    data = request.get_json()
    client_id = data['client_id']
    client_x = np.array(data['x'])
    local_timestamp = data['timestamp']
    
    # Process the async update and get new global state
    new_z, new_y = admm_server.receive_update(client_id, client_x, local_timestamp)
    
    print(f"[{admm_server.global_timestamp}] Aggregated update from {client_id}. New global z calculated.")
    
    return jsonify({
        "status": "success", 
        "global_timestamp": admm_server.global_timestamp,
        "new_z": new_z.tolist(),
        "new_y": new_y.tolist(),
        "message": "Global consensus updated."
    })

if __name__ == '__main__':
    print("Starting Asynchronous Central Parameter Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)