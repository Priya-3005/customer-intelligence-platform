import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load datasets
# -----------------------------
train_df = pd.read_csv("../../data/churn-bigml-80.csv")
test_df = pd.read_csv("../../data/churn-bigml-20.csv")

# -----------------------------
# Drop unused columns
# -----------------------------
drop_cols = ["State", "Area code"]
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

# -----------------------------
# Encode categorical columns
# -----------------------------
label_cols = ["International plan", "Voice mail plan", "Churn"]

for col in label_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# -----------------------------
# Explicit feature list (VERY IMPORTANT)
# -----------------------------
FEATURES = [
    "Account length",
    "International plan",
    "Voice mail plan",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls"
]

X_train = train_df[FEATURES]
y_train = train_df["Churn"]

X_test = test_df[FEATURES]
y_test = test_df["Churn"]

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", accuracy)

# -----------------------------
# Save model
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained & saved successfully")
