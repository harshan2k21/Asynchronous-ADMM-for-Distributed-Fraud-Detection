import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

print("Running Local-Only Baseline (Node 1 Data: Rows 1-10,000)...")

# 1. Load ONLY Node 1's data
df = pd.read_csv('train_sample.csv', nrows=10000)

# 2. Preprocessing & Feature Hashing (Matches your ADMM setup)
categorical_features = ['ip', 'app', 'device', 'os', 'channel', 'hour']
df['hour'] = pd.to_datetime(df['click_time']).dt.hour.astype(str)
for col in categorical_features:
    df[col] = col + "_" + df[col].astype(str)

hasher = FeatureHasher(n_features=50, input_type='string')
hashed_features = hasher.transform(df[categorical_features].values).toarray()

X = np.hstack((hashed_features, np.ones((hashed_features.shape[0], 1)))) # Add bias
y = df['is_attributed'].values

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train standard Scikit-Learn Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict_proba(X_test)[:, 1]
binary_predictions = model.predict(X_test)

print("\n=== LOCAL-ONLY BASELINE METRICS ===")
print(f"ROC-AUC: {roc_auc_score(y_test, predictions):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, predictions):.4f}")
print(f"F1-Score:{f1_score(y_test, binary_predictions):.4f}")