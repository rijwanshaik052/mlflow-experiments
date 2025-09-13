import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import yfinance as yf

# Fetch BTC-USD data between June and September 2025
df = yf.download("BTC-USD", start="2025-06-01", end="2025-09-01")

# Enable autologging
mlflow.sklearn.autolog()

# Create target: 1 if price goes up next day, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()

# Features & target
X = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[:-1]
y = df['Target'].iloc[:-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train a classifier
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    # Accuracy will be logged automatically
    print(f"Model accuracy: {clf.score(X_test, y_test):.4f}")
