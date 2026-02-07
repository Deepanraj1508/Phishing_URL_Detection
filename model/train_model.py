import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# ========================
# 1. Load Dataset
# ========================
DATA_PATH = "dataset/phishing.csv"   # change if your dataset name is different

df = pd.read_csv(DATA_PATH)

# Assuming last column is label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# ========================
# 2. Train-Test Split
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ========================
# 3. Train Model
# ========================
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)


# ========================
# 4. Evaluate
# ========================
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ========================
# 5. Save Model
# ========================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")
