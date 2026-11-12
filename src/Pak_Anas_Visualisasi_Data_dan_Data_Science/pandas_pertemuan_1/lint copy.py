# ruff: file-ignore[print]
"""Praktikum Visualisasi Data dan Data Science: analisis dataset Titanic."""

import pandas as pd
import seaborn as sns
from IPython.display import display

# ============================================================
# Konstanta
# ============================================================

PERSEN = 100
KUARTIL_ATAS = 0.75

LABEL_SHAPE = "Bentuk data (baris, kolom):"
LABEL_SURVIVAL = "Proporsi keselamatan:"
LABEL_DUPLICATES = "Jumlah baris duplikat:"
LABEL_AGE_MIN = "Usia minimum:"
LABEL_AGE_UNDER_ONE = "Usia kurang dari 1 tahun:"
LABEL_FARE_ZERO = "Fare sama dengan 0:"
LABEL_CLASS_SURVIVAL = "Proporsi selamat berdasarkan kelas:"
LABEL_WHO_SURVIVAL = "Jumlah dan proporsi selamat berdasarkan kelompok:"
LABEL_SEX_SURVIVAL = "Proporsi selamat berdasarkan jenis kelamin:"
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

titanic = sns.load_dataset("titanic")

print(LABEL_SHAPE, titanic.shape)
display(titanic.head())


# ============================================================
# Langkah 2 - Profil struktur data
# ============================================================

titanic.info()
display(titanic.dtypes)


# ============================================================
# Langkah 3 - Statistik dan distribusi nilai
# ============================================================

display(titanic.describe())
display(titanic.select_dtypes(exclude="number").describe())

display(titanic["sex"].value_counts())
display(titanic["pclass"].value_counts().sort_index())

print(LABEL_SURVIVAL)
display(titanic["survived"].value_counts(normalize=True).round(3))


# ============================================================
# Langkah 4 - Membuat kamus data
# ============================================================

kamus = pd.DataFrame({
	"variabel": titanic.columns,
	"tipe_pandas": titanic.dtypes.astype(str).to_numpy(),
	"n_unik": titanic.nunique().to_numpy(),
	"n_missing": titanic.isna().sum().to_numpy(),
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
	"class": "Kategori kelas tiket (duplikat pclass)",
	"who": "Kelompok penumpang: man, woman, atau child",
	"adult_male": "Apakah penumpang pria dewasa",
	"deck": "Dek kapal tempat kabin berada",
	"embark_town": "Nama kota pelabuhan keberangkatan (duplikat embarked)",
	"alive": "Status keselamatan dalam teks (duplikat survived)",
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
missing = titanic.isna().sum().to_frame("jumlah_missing")
missing["persen"] = (missing["jumlah_missing"] / len(titanic) * PERSEN).round(1)

display(missing.sort_values("persen", ascending=False))


# 5.2 Keunikan
jumlah_duplikat = int(titanic.duplicated().sum())

print(LABEL_DUPLICATES, jumlah_duplikat)
print(
	"Catatan: jangan langsung menghapus duplikat karena "
	"dataset tidak memiliki ID penumpang.",
)


# 5.3 Validitas
jumlah_usia_bayi = int((titanic["age"] < 1).sum())
jumlah_fare_nol = int((titanic["fare"] == 0).sum())

print(LABEL_AGE_MIN, titanic["age"].min())
print(LABEL_AGE_UNDER_ONE, jumlah_usia_bayi)
print(LABEL_FARE_ZERO, jumlah_fare_nol)


# 5.4 Konsistensi antar kolom
display(pd.crosstab(titanic["pclass"], titanic["class"]))
display(pd.crosstab(titanic["survived"], titanic["alive"]))
display(pd.crosstab(titanic["embarked"], titanic["embark_town"]))


# 5.5 Ringkasan catatan kualitas data (temuan dihitung dari data)
persen_deck = missing.loc["deck", "persen"]
persen_age = missing.loc["age", "persen"]
jumlah_embarked = int(missing.loc["embarked", "jumlah_missing"])
fare_maks = titanic["fare"].max()
fare_q75 = titanic["fare"].quantile(KUARTIL_ATAS)

catatan_kualitas = pd.DataFrame([
	{
		"dimensi": "Kelengkapan",
		"temuan": (
			f"deck kosong {persen_deck}%, age kosong {persen_age}%, "
			f"embarked dan embark_town masing-masing {jumlah_embarked} baris."
		),
		"dampak": "Tinggi (deck), Sedang (age), Rendah (embarked)",
		"tindak_lanjut": (
			"deck dibuang atau dijadikan penanda tercatat/tidak; "
			"age diimputasi; embarked diisi modus."
		),
	},
	{
		"dimensi": "Keunikan",
		"temuan": f"{jumlah_duplikat} baris identik, tanpa kolom ID penumpang.",
		"dampak": "Sedang",
		"tindak_lanjut": "Periksa konteks sebelum menghapus baris.",
	},
	{
		"dimensi": "Validitas",
		"temuan": (
			f"{jumlah_usia_bayi} usia di bawah 1 tahun (wajar); "
			f"{jumlah_fare_nol} tiket fare = 0 (mencurigakan); "
			f"fare maksimum {fare_maks:.2f} jauh di atas persentil ke-75 "
			f"({fare_q75:.2f})."
		),
		"dampak": "Sedang",
		"tindak_lanjut": (
			"Selidiki fare = 0 (tiket gratis, awak, atau tidak tercatat); "
			"gunakan median untuk fare."
		),
	},
	{
		"dimensi": "Konsistensi",
		"temuan": (
			"crosstab hanya terisi di diagonal: pclass = class, "
			"survived = alive, embarked = embark_town."
		),
		"dampak": "Rendah",
		"tindak_lanjut": "Buang salah satu kolom dari tiap pasangan (Minggu 5).",
	},
])

display(catatan_kualitas)


# ============================================================
# Langkah 6 - Pertanyaan analitik
# ============================================================

print(LABEL_CLASS_SURVIVAL)
display(titanic.groupby("class", observed=True)["survived"].mean().round(3))

print(LABEL_WHO_SURVIVAL)
display(
	titanic.groupby("who", observed=True)["survived"].agg(["count", "mean"]).round(3),
)

# Pertanyaan sendiri:
# Apakah proporsi keselamatan berbeda berdasarkan jenis kelamin?
print(LABEL_SEX_SURVIVAL)
display(titanic.groupby("sex", observed=True)["survived"].mean().round(3))
