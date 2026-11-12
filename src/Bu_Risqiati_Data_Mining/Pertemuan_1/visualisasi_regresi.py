# ruff: file-ignore[print]
"""Memvisualisasikan hasil regresi linear."""

import matplotlib.pyplot as plt

# pyrefly: ignore[missing-import]
from IPython.display import display

from Bu_Risqiati_Data_Mining.Pertemuan_1.regresi import run_regression

# Menggunakan fungsi run_regression yang sudah didefinisikan di notebook
model, df, y_test, predictions, r2 = run_regression()

# Menampilkan tabel interaktif bawaan Colab/Pandas
print("Hasil Prediksi:")
display(df.round(2))

# Grafik harga aktual vs prediksi
plt.figure(figsize=(8, 5))

plt.scatter(
	y_test,
	predictions,
	color="blue",
	label="Prediksi",
)

plt.plot(
	[y_test.min(), y_test.max()],
	[y_test.min(), y_test.max()],
	color="red",
	linestyle="--",
	label="Prediksi Ideal",
)

plt.xlabel("Harga Aktual")
plt.ylabel("Harga Prediksi")
plt.title(f"Harga Aktual vs Prediksi — R² = {r2:.2%}")
plt.legend()
plt.grid(visible=True)
plt.show()
