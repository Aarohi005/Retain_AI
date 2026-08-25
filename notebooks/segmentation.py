import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_cleaning import load_and_clean   # import your cleaning function

# Step 1: Load cleaned dataset
df = load_and_clean("D:/projects/churn analysis/data/European_Bank.csv")

# Step 2: Select features for clustering
# (choose numerical columns relevant for segmentation)
X = df[['Age', 'Balance', 'CreditScore', 'EstimatedSalary']]

# Step 3: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply Elbow Method
inertia = []
for k in range(1, 11):   # try cluster counts from 1 to 10
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Step 5: Plot the Elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.show()

# Step 6: Fit K-means with chosen k (example: k=4)
optimal_k = 4   # replace with elbow point you observe
km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = km.fit_predict(X_scaled)

print(df[['Age','Balance','CreditScore','EstimatedSalary','Cluster']].head())
