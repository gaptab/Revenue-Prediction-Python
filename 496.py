import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 🚀 Step 1: Generate Dummy Data
np.random.seed(42)
n_samples = 5000

customer_segments = ['High-Value', 'Medium-Value', 'Low-Value']
regions = ['Urban', 'Suburban', 'Rural']
debt_status = ['Paid on Time', 'Late Payment', 'Default']

df = pd.DataFrame({
    'customer_id': np.arange(1, n_samples+1),
    'segment': np.random.choice(customer_segments, n_samples),
    'region': np.random.choice(regions, n_samples),
    'avg_monthly_spend': np.random.uniform(100, 2000, n_samples),
    'payment_delay_days': np.random.randint(0, 120, n_samples),
    'customer_satisfaction': np.random.uniform(1, 10, n_samples),
    'interaction_count': np.random.randint(1, 50, n_samples),
    'issue_resolution_time': np.random.uniform(1, 30, n_samples),  # In days
    'debt_status': np.random.choice(debt_status, n_samples)
})

# 🚀 Step 2: Demand-Response Analysis (Customer Behavior Clustering)
X_cluster = df[['avg_monthly_spend', 'interaction_count', 'issue_resolution_time']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['demand_cluster'] = kmeans.fit_predict(X_cluster)

# 🚀 Step 3: Customer Experience Analysis (Churn Risk Estimation)
df['churn_risk'] = (df['customer_satisfaction'] < 4).astype(int)

# 🚀 Step 4: Debt Collection Prediction Model
df['debt_risk'] = df['debt_status'].apply(lambda x: 1 if x == 'Default' else 0)

X = df[['avg_monthly_spend', 'payment_delay_days', 'customer_satisfaction', 'interaction_count']]
y = df['debt_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Debt Collection Model Performance:\n", classification_report(y_test, y_pred))

# 🚀 Step 5: Save Data and Model
df.to_csv("customer_analytics_data.csv", index=False)
joblib.dump(model, "debt_collection_model.pkl")

print("Data and Model Saved Successfully!")
