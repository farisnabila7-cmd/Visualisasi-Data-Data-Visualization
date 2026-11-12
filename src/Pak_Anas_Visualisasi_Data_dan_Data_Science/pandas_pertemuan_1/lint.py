# ruff: file-ignore[print]
"""Praktikum Visualisasi Data dan Data Science: analisis dataset Titanic."""

import numpy as np  # ruff: ignore[unused-import]
import pandas as pd
import seaborn as sns
from IPython.display import display

# ============================================================
# Konstanta
# ============================================================

LABEL_SHAPE = "Bentuk data (baris, kolom):"
LABEL_SURVIVAL = "Proporsi keselamatan:"
LABEL_DUPLICATES = "Jumlah baris duplikat:"
LABEL_AGE_MIN = "Usia minimum:"
LABEL_AGE_UNDER_ONE = "Usia kurang dari 1 tahun:"
LABEL_FARE_ZERO = "Fare sama dengan 0:"
LABEL_CLASS_SURVIVAL = "Proporsi selamat berdasarkan kelas:"
LABEL_WHO_SURVIVAL = "Jumlah dan proporsi selamat berdasarkan kelompok:"
LABEL_CSV_SAVED = "Kamus data berhasil disimpan."

STAT_NOMINAL_BINARY = "Nominal (biner)"
STAT_NOMINAL = "Nominal"
STAT_ORDINAL = "Ordinal"
STAT_RATIO_CONTINUOUS = "Rasio (kontinu)"
STAT_RATIO_DISCRETE = "Rasio (diskrit)"


# ============================================================
# Langkah 1 - Memuat data
# ============================================================

pd.set_option("display.max_columns", None)

df = sns.load_dataset("titanic")

print(LABEL_SHAPE, df.shape)
display(df.head())


# ============================================================
# Langkah 2 - Profil struktur data
# ============================================================

df.info()
display(df.dtypes)


# ============================================================
# Langkah 3 - Statistik dan distribusi nilai
# ============================================================

display(df.describe())
display(df.select_dtypes(exclude="number").describe())

display(df["sex"].value_counts())
display(df["pclass"].value_counts().sort_index())

print(LABEL_SURVIVAL)
display(df["survived"].value_counts(normalize=True).round(3))


# ============================================================
# Langkah 4 - Membuat kamus data
# ============================================================

kamus = pd.DataFrame({
	"variabel": df.columns,
	"tipe_pandas": df.dtypes.astype(str).to_numpy(),
	"n_unik": df.nunique().to_numpy(),
	"n_missing": df.isna().sum().to_numpy(),
})

tipe_statistik = {
	"survived": STAT_NOMINAL_BINARY,
	"pclass": STAT_ORDINAL,
	"sex": STAT_NOMINAL,
	"age": STAT_RATIO_CONTINUOUS,
	"sibsp": STAT_RATIO_DISCRETE,
	"parch": STAT_RATIO_DISCRETE,
	"fare": STAT_RATIO_CONTINUOUS,
	"embarked": STAT_NOMINAL,
	"class": STAT_ORDINAL,
	"who": STAT_NOMINAL,
	"adult_male": STAT_NOMINAL_BINARY,
	"deck": STAT_ORDINAL,
	"embark_town": STAT_NOMINAL,
	"alive": STAT_NOMINAL_BINARY,
	"alone": STAT_NOMINAL_BINARY,
}

deskripsi = {
	"survived": "Status keselamatan: 0 = tidak selamat, 1 = selamat",
	"pclass": "Kelas tiket: 1 = pertama, 2 = kedua, 3 = ketiga",
	"sex": "Jenis kelamin penumpang",
	"age": "Usia penumpang dalam tahun",
	"sibsp": "Jumlah saudara atau pasangan di kapal",
	"parch": "Jumlah orang tua atau anak di kapal",
	"fare": "Harga tiket",
	"embarked": "Kode pelabuhan keberangkatan",
	"class": "Kategori kelas tiket",
	"who": "Kelompok penumpang: man, woman, atau child",
	"adult_male": "Apakah penumpang pria dewasa",
	"deck": "Dek kapal tempat kabin berada",
	"embark_town": "Nama kota pelabuhan keberangkatan",
	"alive": "Status keselamatan dalam bentuk teks",
	"alone": "Apakah penumpang bepergian sendiri",
}

satuan_nilai = {
	"survived": "0 atau 1",
	"pclass": "1, 2, atau 3",
	"sex": "male atau female",
	"age": "tahun",
	"sibsp": "orang",
	"parch": "orang",
	"fare": "satuan harga pada dataset",
	"embarked": "C, Q, atau S",
	"class": "First, Second, atau Third",
	"who": "man, woman, atau child",
	"adult_male": "True atau False",
	"deck": "A sampai G atau nilai kosong",
	"embark_town": "Nama kota pelabuhan",
	"alive": "yes atau no",
	"alone": "True atau False",
}

kamus["tipe_statistik"] = kamus["variabel"].map(tipe_statistik)
kamus["deskripsi"] = kamus["variabel"].map(deskripsi)
kamus["satuan_nilai"] = kamus["variabel"].map(satuan_nilai)

display(kamus)

kamus.to_csv("kamus_data_titanic.csv", index=False)
print(LABEL_CSV_SAVED)


# ============================================================
# Langkah 5 - Pemeriksaan kualitas data
# ============================================================

# 5.1 Kelengkapan
missing = df.isna().sum().to_frame("jumlah_missing")
missing["persen"] = (missing["jumlah_missing"] / len(df) * 100).round(1)

display(missing.sort_values("persen", ascending=False))


# 5.2 Keunikan
jumlah_duplikat = df.duplicated().sum()

print(LABEL_DUPLICATES, jumlah_duplikat)
print(
	"Catatan: jangan langsung menghapus duplikat karena "
	"dataset tidak memiliki ID penumpang.",
)


# 5.3 Validitas
print(LABEL_AGE_MIN, df["age"].min())
print(
	LABEL_AGE_UNDER_ONE,
	(df["age"] < 1).sum(),
)
print(
	LABEL_FARE_ZERO,
	(df["fare"] == 0).sum(),
)


# 5.4 Konsistensi antar kolom
display(pd.crosstab(df["pclass"], df["class"]))
display(pd.crosstab(df["survived"], df["alive"]))
display(pd.crosstab(df["embarked"], df["embark_town"]))


# 5.5 Ringkasan catatan kualitas data
catatan_kualitas = pd.DataFrame([
	{
		"dimensi": "Kelengkapan",
		"temuan": "Kolom deck dan age memiliki nilai kosong.",
		"dampak": "Bergantung pada variabel yang dianalisis.",
		"tindak_lanjut": "Evaluasi penanganan missing value.",
	},
	{
		"dimensi": "Keunikan",
		"temuan": (f"Terdapat {jumlah_duplikat} baris identik."),
		"dampak": "Perlu ditinjau.",
		"tindak_lanjut": ("Periksa konteks sebelum menghapus baris."),
	},
	{
		"dimensi": "Validitas",
		"temuan": "Terdapat tiket dengan fare sama dengan 0.",
		"dampak": "Perlu diperiksa untuk analisis harga tiket.",
		"tindak_lanjut": ("Periksa kemungkinan tiket gratis atau data anomali."),
	},
	{
		"dimensi": "Konsistensi",
		"temuan": ("Beberapa pasangan kolom menyimpan informasi terkait."),
		"dampak": "Ada potensi redundansi data.",
		"tindak_lanjut": ("Evaluasi kolom yang dipertahankan saat persiapan data."),
	},
])

display(catatan_kualitas)


# ============================================================
# Langkah 6 - Pertanyaan analitik
# ============================================================

print(LABEL_CLASS_SURVIVAL)
display(df.groupby("class", observed=True)["survived"].mean().round(3))

print(LABEL_WHO_SURVIVAL)
display(df.groupby("who", observed=True)["survived"].agg(["count", "mean"]).round(3))

# Pertanyaan tambahan:
# Apakah proporsi keselamatan berbeda berdasarkan jenis kelamin?
proporsi_jenis_kelamin = df.groupby("sex", observed=True)["survived"].mean().round(3)

display(proporsi_jenis_kelamin)


# ============================================================
# Pemeriksaan akhir
# ============================================================

print("Jumlah baris dan kolom:", df.shape)
