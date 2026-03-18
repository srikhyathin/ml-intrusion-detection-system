import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder

print("Loading dataset...")
data = pd.read_csv("data/nsl_kdd.csv", header=None)

X = data.iloc[:, :-2]
y = data.iloc[:, -2]

encoder = LabelEncoder()
for col in X.select_dtypes(include="object").columns:
    X[col] = encoder.fit_transform(X[col])

y = y.apply(lambda x: 0 if x == "normal" else 1)

print("Training RandomForest...")
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X, y)

print("Training IsolationForest...")
iso_model = IsolationForest(contamination=0.1)
iso_model.fit(X)

joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(iso_model, "models/iso_model.pkl")

print("Models saved successfully.")