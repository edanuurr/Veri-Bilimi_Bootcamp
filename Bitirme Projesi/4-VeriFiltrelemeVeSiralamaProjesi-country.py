import pandas as pd
import numpy as np

data = pd.read_csv('country.csv')

# --- Veri Temizleme Adımları ---
data.columns = data.columns.str.strip()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].str.strip()

columns_to_convert = data.select_dtypes(include=['object']).columns
for col in columns_to_convert:
    if col not in ['Country', 'Region']:
        data[col] = pd.to_numeric(data[col].str.replace(',', '.'), errors='coerce')

numerical_cols = data.select_dtypes(include=np.number).columns
for col in numerical_cols:
    data[col] = data.groupby('Region')[col].transform(lambda x: x.fillna(x.mean()))

data.fillna(data.mean(numeric_only=True), inplace=True)

# --- Görevler ---

# ## 1. Görev : Nüfusa Göre Azalan Sırada Sıralama
print("--- Görev 1: Nüfusa Göre Sıralanmış İlk 10 Ülke ---")
sorted_by_population = data.sort_values(by='Population', ascending=False)
print(sorted_by_population[['Country', 'Population']].head(10))
print("\n" + "="*50 + "\n")

# ## 2. Görev: GDP per capita sütununa göre ülkeleri artan sırada sıralamak(Kişi başına düşen Gayri Safi Yurtiçi Hasıla).
print("--- Görev 2: Kişi Başı GSYİH'ye Göre En Düşük 10 Ülke ---")
sorted_by_gdp = data.sort_values(by='GDP ($ per capita)', ascending=True)
print(sorted_by_gdp[['Country', 'Region', 'GDP ($ per capita)']].head(10))
print("\n" + "="*50 + "\n")

# ## 3. Görev: Population sütunu 10 milyonun üzerinde olan ülkeleri seçmek.
print("--- Görev 3: Nüfusu 10 Milyondan Fazla Olan Ülkeler ---")
population_over_10m = data[data['Population'] > 10000000]
print(f"Nüfusu 10 milyondan fazla olan ülke sayısı: {len(population_over_10m)}")
print(population_over_10m[['Country', 'Population']].head())
print("\n" + "="*50 + "\n")

# ## 4. Görev: Literacy (%) sütununa göre ülkeleri sıralayıp, en yüksek okur-yazarlık oranına sahip ilk 5 ülkeyi seçmek.
print("--- Görev 4: Okur-Yazarlık Oranı En Yüksek İlk 5 Ülke ---")
top_5_literacy = data.sort_values(by='Literacy (%)', ascending=False).head(5)
print(top_5_literacy[['Country', 'Literacy (%)']])
print("\n" + "="*50 + "\n")

# ## 5. Görev:  Kişi Başı GSYİH 10.000'in Üzerinde Olan Ülkeleri Filtreleme: GDP ( per capita) sütunu 10.000'in üzerinde olan ülkeleri seçmek.
print("--- Görev 5: Kişi Başı GSYİH'si 10.000'den Yüksek Olan Ülkeler ---")
gdp_over_10k = data[data['GDP ($ per capita)'] > 10000]
print(f"Kişi başı GSYİH'si 10.000'den yüksek olan ülke sayısı: {len(gdp_over_10k)}")
print(gdp_over_10k[['Country', 'GDP ($ per capita)']].head())
print("\n" + "="*50 + "\n")

# ## Görev 6 : En Yüksek Nüfus Yoğunluğuna Sahip İlk 10 Ülkeyi Seçme
print("--- Görev 6: Nüfus Yoğunluğu En Yüksek İlk 10 Ülke ---")
top_10_density = data.sort_values(by='Pop. Density (per sq. mi.)', ascending=False).head(10)
print(top_10_density[['Country', 'Pop. Density (per sq. mi.)']])
print("\n" + "="*50 + "\n")