# ruff: file-ignore[print]
# ruff: file-ignore[docstring-missing-returns]
"""Contoh dasar Python dan Pandas untuk mengolah data transaksi toko."""

import sys

import pandas as pd

BATAS_DISKON_10 = 5
BATAS_DISKON_5 = 3

FAKTOR_DISKON_10 = 0.90
FAKTOR_DISKON_5 = 0.95
FAKTOR_NORMAL = 1.00


print("Python:", sys.version.split()[0])
print("Pandas:", pd.__version__)


produk = ["Beras", "Minyak", "Gula"]

harga = {
	"Beras": 14000,
	"Minyak": 17000,
	"Gula": 16000,
}


for p in produk:
	print(p, "->", harga[p])


def hitung_total(harga_satuan: float, jumlah: int) -> float:
	"""Menghitung total harga berdasarkan jumlah dan aturan diskon."""
	if jumlah > BATAS_DISKON_10:
		faktor = FAKTOR_DISKON_10
	elif jumlah > BATAS_DISKON_5:
		faktor = FAKTOR_DISKON_5
	else:
		faktor = FAKTOR_NORMAL

	return harga_satuan * jumlah * faktor


print("\nContoh hitung total:")
print(hitung_total(harga["Beras"], 2))


transaksi = [
	{
		"id": 1,
		"tanggal": "2026-09-01",
		"produk": "Beras",
		"kategori": "Pangan",
		"jumlah": 3,
		"harga_satuan": 14000,
	},
	{
		"id": 2,
		"tanggal": "2026-09-01",
		"produk": "Minyak",
		"kategori": "Pangan",
		"jumlah": 2,
		"harga_satuan": 17000,
	},
	{
		"id": 3,
		"tanggal": "2026-09-02",
		"produk": "Sabun",
		"kategori": "Kebersihan",
		"jumlah": 5,
		"harga_satuan": 4500,
	},
	{
		"id": 4,
		"tanggal": "2026-09-02",
		"produk": "Gula",
		"kategori": "Pangan",
		"jumlah": 1,
		"harga_satuan": 16000,
	},
	{
		"id": 5,
		"tanggal": "2026-09-03",
		"produk": "Deterjen",
		"kategori": "Kebersihan",
		"jumlah": 2,
		"harga_satuan": 12000,
	},
	{
		"id": 6,
		"tanggal": "2026-09-03",
		"produk": "Beras",
		"kategori": "Pangan",
		"jumlah": 4,
		"harga_satuan": 14000,
	},
	{
		"id": 7,
		"tanggal": "2026-09-04",
		"produk": "Teh",
		"kategori": "Minuman",
		"jumlah": 6,
		"harga_satuan": 9000,
	},
	{
		"id": 8,
		"tanggal": "2026-09-04",
		"produk": "Kopi",
		"kategori": "Minuman",
		"jumlah": 3,
		"harga_satuan": 11000,
	},
]


df = pd.DataFrame(transaksi)


# Menambahkan kolom total.
df["total"] = df["jumlah"] * df["harga_satuan"]


print("\n=== Data Transaksi ===")
print(df)


print("\n=== 5 Data Pertama ===")
print(df.head())


print("\n=== 5 Data Terakhir ===")
print(df.tail())


print("\n=== Kolom Produk ===")
print(df["produk"])


print("\n=== Produk dan Jumlah ===")
print(df[["produk", "jumlah"]])


print("\n=== Transaksi dengan Jumlah >= 3 ===")
print(df[df["jumlah"] >= BATAS_DISKON_5][["produk", "jumlah"]])


print("\n=== 3 Transaksi Terbesar ===")
print(
	df.sort_values(
		"total",
		ascending=False,
	).head(3),
)


print("\n=== Produk Baris Pertama ===")
print(df.loc[0, "produk"])


print("\n=== Produk Baris Ketiga ===")
print(df.iloc[2]["produk"])


print("\n=== Total Berdasarkan Kategori ===")
print(df.groupby("kategori")["total"].sum())


df.to_csv(
	"transaksi_toko.csv",
	index=False,
)


df2 = pd.read_csv(
	"transaksi_toko.csv",
	dtype={
		"id": "int64",
		"produk": "string",
		"kategori": "string",
		"jumlah": "int64",
		"harga_satuan": "int64",
		"total": "int64",
	},
	parse_dates=["tanggal"],
)


print("\n=== Ukuran Data ===")
print(df2.shape)


print("\n=== Data dari CSV ===")
print(df2.head())
