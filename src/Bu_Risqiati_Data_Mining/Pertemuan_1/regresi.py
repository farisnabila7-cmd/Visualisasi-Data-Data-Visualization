# ruff: file-ignore[print]
# ruff: file-ignore[docstring-missing-returns]
"""Belajar Regresi Kelas Data Mining, Pertemuan 1."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def run_regression() -> tuple[
	LinearRegression,
	pd.DataFrame,
	np.ndarray,
	np.ndarray,
	float,
]:
	"""Melatih model regresi dan mengembalikan hasil evaluasi."""
	# Data Sample
	# fmt: off
	x = np.array([
		[100, 2, 1], [150, 3, 2], [200, 3, 2.5], [120, 2, 1.5], [180, 3, 2.5],
		[250, 4, 3], [130, 2, 1.5], [220, 4, 2.5], [180, 3, 2], [200, 3, 2],
		[170, 3, 2], [160, 3, 1.5], [140, 2, 1.5], [210, 4, 2], [240, 4, 3],
		[190, 3, 2.5], [270, 4, 3], [230, 4, 2.5], [260, 5, 3], [280, 5, 3.5],
	])

	y = np.array([
		300, 400, 500, 350, 450,
		600, 320, 550, 470, 480,
		430, 380, 340, 490, 580,
		460, 620, 570, 640, 680,
	])
	# fmt: on

	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=0.2,
		random_state=42,
	)

	model = LinearRegression()
	model.fit(x_train, y_train)

	predictions = model.predict(x_test)

	df = pd.DataFrame(
		x_test,
		columns=[
			"Luas Tanah",
			"Jumlah Kamar Tidur",
			"Jumlah Kamar Mandi",
		],
	)

	df["Harga Asli"] = y_test
	df["Prediksi Harga"] = predictions
	df["Akurasi Prediksi (%)"] = 100 - (np.abs(predictions - y_test) / y_test * 100)

	r2 = model.score(x_test, y_test)

	return model, df, y_test, predictions, r2


if __name__ == "__main__":
	model, df, y_test, predictions, r2 = run_regression()

	print(f"R² Model: {r2:.2%}")
	print(df.round(2).to_string(index=False))
