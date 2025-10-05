import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')
data.head()

# ## 1.GÖREV : R&D Harcaması ve Kâr Arasındaki İlişki (Scatter Plot): Ar-Ge harcamaları ile kâr arasındaki ilişkiyi gösteren bir dağılım grafiği.
plt.figure(figsize=(10, 6))
plt.scatter(data['R&D Spend'], data['Profit'], alpha=0.7)
plt.title('R&D (Ar-Ge) Harcaması ve Kâr Arasındaki İlişki', fontsize=16)
plt.xlabel('R&D (Ar-Ge) Harcaması', fontsize=12)
plt.ylabel('Kâr', fontsize=12)
plt.grid(True, alpha=0.5)
plt.show()

# ## 2.GÖREV: Yönetim Harcamaları ve Kâr Arasındaki İlişki (Scatter Plot): Yönetim harcamaları ile kâr arasındaki ilişkiyi gösteren bir dağılım grafiği.
plt.figure(figsize=(10, 6))
plt.scatter(data['Administration'], data['Profit'], alpha=0.7)
plt.title('Yönetim Harcamaları ve Kâr Arasındaki İlişki', fontsize=16)
plt.xlabel('Yönetim Harcaması', fontsize=12)
plt.ylabel('Kâr', fontsize=12)
plt.grid(True, alpha=0.5)
plt.show()

# ## 3. GÖREV: Eyaletlere Göre Ortalama Kâr (Bar Chart): Farklı eyaletlerdeki startup'ların ortalama kârlarını karşılaştıran bir çubuk grafik.
plt.figure(figsize=(10, 6))
ort_kar = data.groupby('State')['Profit'].mean().sort_values(ascending=False)
plt.bar(ort_kar.index, ort_kar.values, color=['blue', 'red', 'purple'])
plt.title('Eyaletlere Göre Ortalama Kâr', fontsize=16)
plt.xlabel('Eyalet', fontsize=12)
plt.ylabel('Ortalama Kâr', fontsize=12)
plt.grid(True, alpha=0.5)
plt.show()

# Ortalama değerleri yazdırma
print("\nEyalet\t\t\tOrtalama Kâr")
print("-" * 35)
for state, profit in ort_kar.items():
    print(f"{state:<10}\t\t{profit:,.2f}")

# ## 4. GÖREV: Harcama Türlerinin Karşılaştırması (Boxplot): R&D, yönetim ve pazarlama harcamalarının dağılımını karşılaştıran bir kutu grafiği.
harcama_veri = data[['R&D Spend', 'Administration', 'Marketing Spend']]
plt.figure(figsize=(10, 6))
plt.boxplot(harcama_veri.values, patch_artist=True)
plt.xticks([1, 2, 3], ['R&D Spend', 'Administration', 'Marketing Spend'])
plt.title('Harcama Türlerinin Dağılım Karşılaştırması', fontsize=16)
plt.ylabel('Harcama Miktarı', fontsize=12)
plt.grid(True, alpha=0.5)
plt.show()

# İstatistiksel özet tablosunu oluşturma ve yazdırma
harcama_ozeti = harcama_veri.describe()
tablo_verisi = harcama_ozeti.loc[['mean', '50%', 'std']].T
tablo_verisi.columns = ['Ortalama Değer', 'Orta Değer (Medyan)', 'Dağılım (Std. Sapma)']
tablo_verisi.index = ['R&D Harcaması', 'Yönetim Harcaması', 'Marketing Harcaması']

print("\n\n\n --- Harcama Türlerine Göre İstatistiksel Özet --- \n")
print(tablo_verisi.to_string(float_format="{:,.2f}".format))