#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:08:19 2026

@author: buseguldalii
"""

"""
Salary Prediction Using Linear, Polynomial, and Support Vector Regression
Comparing different regression models and evaluating SVR performance.
"""

# 1. KÜTÜPHANELER (Tümü en başta toplanır)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# 2. VERİ YÜKLEME VE DÜZENLEME
data = pd.read_csv('maaslar.csv')
data = data.sort_values(by='Egitim Seviyesi')

# Bağımsız (X) ve Bağımlı (y) değişkenleri ayırma
X = data.iloc[:, 1:2].values  # Education Level
y = data.iloc[:, 2:].values   # Salary

# ---------------------------------------------------------
# MODEL 1: LINEAR REGRESSION
# ---------------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X, lin_reg.predict(X), color='blue', label='Linear Prediction')
plt.title('Linear Regression - Education vs Salary')
plt.xlabel('Education Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# ---------------------------------------------------------
# MODEL 2: POLYNOMIAL REGRESSION (Degree = 2)
# ---------------------------------------------------------
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X, lin_reg_poly.predict(X_poly), color='green', label='Polynomial Prediction (Deg=2)')
plt.title('Polynomial Regression - Education vs Salary')
plt.xlabel('Education Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# ---------------------------------------------------------
# MODEL 3: SUPPORT VECTOR REGRESSION (SVR)
# ---------------------------------------------------------
# Veriyi Train (%80) ve Test (%20) olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# SVR mesafe bazlı çalıştığı için Özellik Ölçeklendirme (Feature Scaling) zorunludur
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train_scaled = sc_X.fit_transform(X_train)
y_train_scaled = sc_y.fit_transform(y_train)

# Test verisinde 'fit' (öğrenme) yapılmaz, sadece 'transform' uygulanır
X_test_scaled = sc_X.transform(X_test)

# SVR Modeli Oluşturma ve Eğitme
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train_scaled, y_train_scaled.ravel())

# Test verisi üzerinden tahmin yapma ve tahminleri gerçek ölçeğe geri çevirme
y_pred_scaled = svr_reg.predict(X_test_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# SVR Grafiği (Test Verisi Üzerinden)
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='red', label='Actual Test Data')
# Çizgi çekmek anlamsız grafik yaratacağı için tahminler farklı bir marker ile gösterildi
plt.scatter(X_test, y_pred, color='blue', marker='x', s=100, label='SVR Predictions')
plt.title('Support Vector Regression (Test Set Performance)')
plt.xlabel('Education Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# ---------------------------------------------------------
# SVR MODEL DEĞERLENDİRMESİ (METRİKLER)
# ---------------------------------------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Düzeltilmiş (Adjusted) R-Kare hesaplaması
n = len(y_test)
p = X_test.shape[1]
denominator = n - p - 1

print("\n--- SVR MODEL PERFORMANCE SUMMARY ---")
print(f"R-Squared (R2): {r2:.4f}")

if denominator > 0:
    adj_r2 = 1 - ((1 - r2) * (n - 1) / denominator)
    print(f"Adjusted R-Squared: {adj_r2:.4f}")
else:
    print("Adjusted R-Squared: N/A (Test set is too small)")

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

