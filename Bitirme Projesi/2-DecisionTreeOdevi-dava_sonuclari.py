import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv('dava_sonuclari.csv')

## Görevler
### 1.Görev: Veri Ön İşleme
median_evidence = data['Number of Evidence Items'].median()
median_fees = data['Legal Fees (USD)'].median()

condition1 = data['Number of Evidence Items'] > median_evidence
condition2 = data['Legal Fees (USD)'] > median_fees
condition3 = data["Plaintiff's Reputation"] == 3

data['Outcome'] = (condition1 | condition2 | condition3).astype(int)
data = pd.get_dummies(data, columns=['Case Type'], drop_first=True)

### 2.Görev: Veri Setini Ayırma 
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

### 3.Görev: Model Kurulumu
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

### 4.Görev: Modeli Değerlendirme
y_pred = dt_model.predict(X_test)
print("--- Model Değerlendirme Raporu ---")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=['Kaybetmek (0)', 'Kazanmak (1)']))
print(f"Doğruluk (Accuracy): {accuracy_score(y_test, y_pred):.2f}")

### 5.Görev: Sonuçları Görselleştirme
print("\n--- Karar Ağacı Görselleştirmesi ---")
plt.figure(figsize=(20, 12))
plot_tree(dt_model, 
          feature_names=X.columns, 
          class_names=['Kaybetmek', 'Kazanmak'], 
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Dava Sonucunu Tahmin Eden Karar Ağacı", fontsize=20)
plt.show()