import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('dava.csv')
data = data.drop(data.columns[0], axis=1)

# Görev 1: Özellik Seçimi
features = data.drop('Outcome', axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Görev 2: Optimal Küme Sayısını Belirleme (Elbow Yöntemi)
wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Yöntemi ile Optimal Küme Sayısını Belirleme')
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('WCSS (Küme İçi Kareler Toplamı)')
plt.grid(True, alpha=0.5)
plt.show()

# Görev 3: K-Means Algoritmasını Uygulama
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
data['Cluster'] = clusters

# Görev 4: Sonuçları Görselleştirme ve Yorumlama
colors = ['purple', 'orange', 'green']
cluster_labels = [f'Küme {i}' for i in range(optimal_k)]

plt.figure(figsize=(12, 8))
for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Legal Fees (USD)'], 
                cluster_data['Case Duration (Days)'], 
                s=100, 
                c=colors[i], 
                label=cluster_labels[i], 
                alpha=0.8)

centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)
x_centroid_idx = features.columns.get_loc('Legal Fees (USD)')
y_centroid_idx = features.columns.get_loc('Case Duration (Days)')
plt.scatter(centroids[:, x_centroid_idx], 
            centroids[:, y_centroid_idx], 
            s=300, c='red', marker='*', 
            label='Merkezler (Centroids)')

plt.title('Dava Verilerinin K-Means ile Kümelenmesi')
plt.xlabel('Hukuk Maliyetleri (USD)')
plt.ylabel('Dava Süresi (Gün)')
plt.legend(title='Kümeler')
plt.grid(True, alpha=0.5)
plt.show()

cluster_analysis = data.drop('Outcome', axis=1).groupby('Cluster').mean()
print("\n--- Küme Bazında Ortalama Değerler ---")
print(cluster_analysis)