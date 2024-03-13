# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:06:50 2024

@author: Domagoj
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 1. Učitavanje podataka
file_path = 'p12.xlsx'
data = pd.read_excel(file_path)

# 2. Osnovne statističke mjere
mean = data['Broj bodova/utakmici'].mean()
mode = data['Broj bodova/utakmici'].mode()[0]
median = data['Broj bodova/utakmici'].median()
five_number_summary = data['Broj bodova/utakmici'].describe()
variance = data['Broj bodova/utakmici'].var()
std_deviation = data['Broj bodova/utakmici'].std()
IQR = data['Broj bodova/utakmici'].quantile(0.75) - data['Broj bodova/utakmici'].quantile(0.25)
range_sample = data['Broj bodova/utakmici'].max() - data['Broj bodova/utakmici'].min()

# 3. Raspodjele frekvencija
bins = np.arange(start=data['Broj bodova/utakmici'].min(), stop=data['Broj bodova/utakmici'].max() + 1, step=2)
frequency_distribution = pd.cut(data['Broj bodova/utakmici'], bins).value_counts().sort_index()
relative_frequency_distribution = frequency_distribution / len(data)
cumulative_frequency_distribution = relative_frequency_distribution.cumsum()

# Stvaranje tablice frekvencija
frequency_table = pd.DataFrame({
    'Intervali': [f'{interval.left}-{interval.right}' for interval in frequency_distribution.index],
    'Frekvencija': frequency_distribution.values,
    'Relativna Frekvencija': relative_frequency_distribution.values,
    'Kumulativna Frekvencija': cumulative_frequency_distribution.values
})

# 4. Grafički prikaz
# Histogram frekvencija
plt.hist(data['Broj bodova/utakmici'], bins=bins, edgecolor='black')
plt.title('Histogram Frekvencija')
plt.xlabel('Broj Bodova/Utakmici')
plt.ylabel('Frekvencija')
plt.show()

# Histogram relativnih frekvencija
plt.hist(data['Broj bodova/utakmici'], bins=bins, weights=np.ones(len(data)) / len(data), edgecolor='black')
plt.title('Histogram Relativnih Frekvencija')
plt.xlabel('Broj Bodova/Utakmici')
plt.ylabel('Relativna Frekvencija')
plt.show()

# Poligon frekvencija
bin_centers = 0.5 * (bins[:-1] + bins[1:])
plt.plot(bin_centers, frequency_distribution, 'o-')
plt.title('Poligon Frekvencija')
plt.xlabel('Broj Bodova/Utakmici')
plt.ylabel('Frekvencija')
plt.show()

# 5. Intervali povjerenja
confidence_levels = [0.90, 0.95, 0.99]
confidence_intervals = {}
for confidence in confidence_levels:
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin_of_error = z_score * std_deviation / np.sqrt(len(data))
    confidence_intervals[confidence] = (mean - margin_of_error, mean + margin_of_error)

# 6. Testiranje hipoteze
t_statistic, p_value = stats.ttest_1samp(data['Broj bodova/utakmici'], 18)  # 18 je primjer pretpostavljene srednje vrijednosti

# Rezultati
print("Aritmetička sredina:", mean)
print("Mod:", mode)
print("Medijan:", median)
print("Karakteristična petorka:", five_number_summary)
print("Varijanca:", variance)
print("Standardna devijacija:", std_deviation)
print("Interkvartilni raspon:", IQR)
print("Raspon uzorka:", range_sample)
print("Tablica Frekvencija:")
print(frequency_table)
print("Intervali povjerenja:", confidence_intervals)
print("Rezultat t-testa: t-statistika =", t_statistic, ", p-vrijednost =", p_value)
