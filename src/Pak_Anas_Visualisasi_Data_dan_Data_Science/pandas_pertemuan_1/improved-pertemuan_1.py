# ruff: file-ignore[print]  # ruff: ignore[implicit-namespace-package]
# ruff: file-ignore[docstring-missing-returns]
"""Data cleaning dan analisis transaksi menggunakan Pandas.

Alur: load -> parse tanggal -> bersihkan noise -> hapus duplikat
      -> feature engineering -> deteksi outlier -> analisis -> simpan.
"""

import logging
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

try:
	BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
	BASE_DIR = Path.cwd()

OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

OUTPUT_FILE = OUTPUT_DIR / "data_transaksi_clean.csv"
REJECTED_FILE = OUTPUT_DIR / "data_transaksi_rejected.csv"
LOG_FILE = LOG_DIR / "data_cleaning.log"

SEPARATOR = "=" * 50
SECTION_SEPARATOR = "=" * 40

# --- Aturan noise (nilai mustahil / tidak masuk akal) ---------
# Teks placeholder yang dianggap data kosong saat membaca CSV.
NA_VALUES = ["", "NA", "N/A", "NaN", "null", "None", "-", "?", "Missing", "missing"]

MIN_TANGGAL = pd.Timestamp("2020-01-01")  # tanggal sebelum ini dianggap noise
MIN_HARGA = 1_000  # harga di bawah ini dianggap salah input
MAX_HARGA = 500_000_000  # harga di atas ini dianggap salah input
MAX_JUMLAH = 1_000  # jumlah per baris di atas ini dianggap salah input

# --- Aturan outlier (statistik) -------------------------------
# "drop": baris outlier dibuang dari data clean (disimpan di file rejected)
# "flag": baris outlier dipertahankan, hanya diberi penanda
OUTLIER_STRATEGY = "drop"
OUTLIER_COLUMNS = ["Harga", "Jumlah", "Total_Pendapatan"]
IQR_MULTIPLIER = 3.0  # 1.5 = ketat (outlier ringan ikut), 3.0 = hanya outlier ekstrem
MIN_SAMPLE = 5  # minimal data valid agar deteksi statistik dijalankan

OUTPUT_DIR.mkdir(
	parents=True,
	exist_ok=True,
)

LOG_DIR.mkdir(
	parents=True,
	exist_ok=True,
)

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
	level=logging.INFO,
	handlers=[],
)

logger = logging.getLogger(__name__)
logger.handlers.clear()

file_handler = logging.FileHandler(
	LOG_FILE,
	encoding="utf-8",
)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# ============================================================
# MOCK DATA
# ============================================================

MOCK_DATA = """Invoice_ID,Tanggal_Order,Customer_ID,Kategori,Harga_Str,Jumlah,Kota_Toko
INV-101,2026-01-02,C-881,Elektronik,"Rp 4.500.000",1,Jakarta
INV-102,03/01/2026,C-102,Fashion,"Rp 250.000",3,Bandung
INV-103,2026-01-05,,Elektronik,"Rp 120.000",5,Surabaya
INV-104,2026-01-08,C-102,Rumah Tangga,"Rp 600.000",-2,JAKARTA
INV-105,,C-409,Fashion,"Rp 350.000",2,Yogyakarta
INV-106,2026-01-12,C-881,elektronik,"Rp 4.500.000",1,Medan
INV-107,2026-01-15,C-512,Rumah Tangga,"Rp 150.000",3,bandung
INV-108,2026-01-18,C-204,Fashion,"Rp 500.000",Missing,Surabaya
INV-101,2026-01-02,C-881,Elektronik,"Rp 4.500.000",1,Jakarta
INV-110,2026-01-22,C-301,Kecantikan,"Rp 85.000",10,
"""

# ============================================================
# HELPER
# ============================================================


def split_rejected(
	df: pd.DataFrame,
	mask: pd.Series,
	reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Memisahkan baris yang ditolak (mask True) beserta alasannya."""
	mask = mask.fillna(False).astype(bool)  # ruff: ignore[boolean-positional-value-in-call]

	logger.info(
		"%s: %s baris",
		reason,
		int(mask.sum()),
	)

	rejected = df[mask].assign(Alasan=reason)

	return (
		df[~mask].copy(),
		rejected,
	)


# ============================================================
# LOAD DATA
# ============================================================


def load_data() -> pd.DataFrame:
	"""Memuat data mentah."""
	return pd.read_csv(
		StringIO(MOCK_DATA),
		na_values=NA_VALUES,
		dtype={
			"Invoice_ID": "string",
			"Customer_ID": "string",
			"Kategori": "string",
			"Harga_Str": "string",
			"Kota_Toko": "string",
		},
	)


# ============================================================
# PARSE DATES
# ============================================================


def parse_dates(
	df: pd.DataFrame,
) -> pd.DataFrame:
	"""Standardisasi format tanggal."""
	df = df.copy()

	tanggal = df["Tanggal_Order"].astype("string").str.strip()

	mask_iso = tanggal.str.match(
		r"^\d{4}-\d{2}-\d{2}$",
		na=False,
	)

	mask_dmy = tanggal.str.match(
		r"^\d{2}/\d{2}/\d{4}$",
		na=False,
	)

	parsed_tanggal = pd.Series(
		pd.NaT,
		index=df.index,
		dtype="datetime64[ns]",
	)

	parsed_tanggal.loc[mask_iso] = pd.to_datetime(
		tanggal.loc[mask_iso],
		format="%Y-%m-%d",
		errors="coerce",
	)

	parsed_tanggal.loc[mask_dmy] = pd.to_datetime(
		tanggal.loc[mask_dmy],
		format="%d/%m/%Y",
		errors="coerce",
	)

	df["Tanggal_Order"] = parsed_tanggal

	return df


# ============================================================
# CLEAN FIELDS (NOISE)
# ============================================================


def clean_fields(
	df: pd.DataFrame,
) -> pd.DataFrame:
	"""Membersihkan field numerik dan kategorikal dari noise."""
	df = df.copy()

	# --------------------------------------------------------
	# CLEAN HARGA
	# --------------------------------------------------------

	harga = pd.to_numeric(
		df["Harga_Str"]
		.astype("string")
		.str.replace(
			r"(?i)rp",
			"",
			regex=True,
		)
		.str.replace(
			".",
			"",
			regex=False,
		)
		.str.replace(
			",",
			".",
			regex=False,
		)
		.str.strip(),
		errors="coerce",
	).astype("float64")

	# Harga di luar batas wajar dianggap noise -> dikosongkan.
	harga_invalid = harga.notna() & ~harga.between(MIN_HARGA, MAX_HARGA)

	df["Harga_Invalid"] = harga_invalid
	df["Harga"] = harga.mask(harga_invalid)

	# --------------------------------------------------------
	# CLEAN JUMLAH
	# --------------------------------------------------------

	jumlah = pd.to_numeric(
		df["Jumlah"],
		errors="coerce",
	).astype("float64")

	# Jumlah <= 0, pecahan, atau tidak masuk akal dianggap noise.
	jumlah_invalid = jumlah.notna() & (
		(jumlah <= 0) | (jumlah % 1 != 0) | (jumlah > MAX_JUMLAH)
	)

	df["Jumlah_Invalid"] = jumlah_invalid
	df["Jumlah"] = jumlah.mask(jumlah_invalid).astype("Int64")

	# --------------------------------------------------------
	# CLEAN KOTA
	# --------------------------------------------------------

	df["Kota_Toko"] = (
		df["Kota_Toko"]
		.astype("string")
		.str.strip()
		.str.replace(r"\s+", " ", regex=True)
		.str.title()
		.fillna("Unknown")
	)

	# --------------------------------------------------------
	# CLEAN KATEGORI
	# --------------------------------------------------------

	df["Kategori"] = (
		df["Kategori"]
		.astype("string")
		.str.strip()
		.str.replace(r"\s+", " ", regex=True)
		.str.title()
		.fillna("Unknown")
	)

	# --------------------------------------------------------
	# CLEAN CUSTOMER & INVOICE
	# --------------------------------------------------------

	df["Customer_ID"] = df["Customer_ID"].str.strip().str.upper().fillna("Unknown")

	df["Invoice_ID"] = df["Invoice_ID"].str.strip().str.upper()

	return df


# ============================================================
# HANDLE INVALID ROWS & DUPLICATES
# ============================================================


def handle_missing_and_duplicates(
	df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Membuang baris tidak valid dan duplikat.

	Returns:
		(data bersih, data yang ditolak beserta kolom Alasan)

	"""
	rejected_parts: list[pd.DataFrame] = []

	# Invoice_ID tidak sesuai format
	mask = ~df["Invoice_ID"].str.match(
		r"^INV-\d+$",
		na=False,
	)
	df, rejected = split_rejected(df, mask, "Invoice_ID tidak valid")
	rejected_parts.append(rejected)

	# Tanggal kosong / gagal parse / tidak masuk akal
	mask = (
		df["Tanggal_Order"].isna()
		| (df["Tanggal_Order"] < MIN_TANGGAL)
		| (df["Tanggal_Order"] > pd.Timestamp.today().normalize())
	)
	df, rejected = split_rejected(df, mask, "Tanggal_Order kosong/tidak valid")
	rejected_parts.append(rejected)

	# Duplikat persis
	mask = df.duplicated(keep="first")
	df, rejected = split_rejected(df, mask, "Duplikat persis")
	rejected_parts.append(rejected)

	# Invoice_ID sama tapi isi berbeda (data bentrok) -> simpan yang pertama
	mask = df.duplicated(subset=["Invoice_ID"], keep="first")
	df, rejected = split_rejected(df, mask, "Invoice_ID ganda dengan isi berbeda")
	rejected_parts.append(rejected)

	rejected_all = pd.concat(rejected_parts, ignore_index=False)

	return (
		df,
		rejected_all,
	)


# ============================================================
# FEATURE ENGINEERING
# ============================================================


def create_features(
	df: pd.DataFrame,
) -> pd.DataFrame:
	"""Membuat fitur turunan dari data transaksi."""
	df = df.copy()

	df["Month"] = df["Tanggal_Order"].dt.month

	df["Day_of_Week"] = df["Tanggal_Order"].dt.day_name()

	df["Jumlah_Missing"] = df["Jumlah"].isna()

	df["Total_Pendapatan"] = (df["Harga"] * df["Jumlah"]).astype("float64")

	return df


# ============================================================
# OUTLIER DETECTION
# ============================================================


def iqr_mask(
	series: pd.Series,
) -> pd.Series:
	"""Mendeteksi outlier dengan IQR pada skala log.

	Skala log dipakai karena harga/pendapatan condong ke kanan:
	barang mahal yang wajar tidak langsung dianggap outlier.
	"""
	values = series.astype("float64")

	valid = values.dropna()

	if len(valid) < MIN_SAMPLE:
		return pd.Series(False, index=series.index)  # ruff: ignore[boolean-positional-value-in-call]

	q1, q3 = np.log1p(valid).quantile([0.25, 0.75])

	iqr = q3 - q1

	if iqr == 0:
		return pd.Series(False, index=series.index)  # ruff: ignore[boolean-positional-value-in-call]

	lower = q1 - IQR_MULTIPLIER * iqr
	upper = q3 + IQR_MULTIPLIER * iqr

	logged = np.log1p(values)

	return ((logged < lower) | (logged > upper)).fillna(False)  # ruff: ignore[boolean-positional-value-in-call]


def flag_outliers(
	df: pd.DataFrame,
	column: str,
) -> pd.Series:
	"""Menandai outlier per kategori; fallback ke seluruh data bila sampel kecil."""
	flags = iqr_mask(df[column])

	for _, group in df.groupby("Kategori"):
		if group[column].notna().sum() >= MIN_SAMPLE:
			flags.loc[group.index] = iqr_mask(group[column])

	return flags


def handle_outliers(
	df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Mendeteksi outlier lalu membuang / menandainya sesuai OUTLIER_STRATEGY."""
	df = df.copy()

	for column in OUTLIER_COLUMNS:
		df[f"Outlier_{column}"] = flag_outliers(df, column)

	flag_columns = [f"Outlier_{c}" for c in OUTLIER_COLUMNS]

	df["Is_Outlier"] = df[flag_columns].any(axis=1)

	alasan = df[flag_columns].apply(
		lambda row: (
			"Outlier: "
			+ ", ".join(c.removeprefix("Outlier_") for c in flag_columns if row[c])
		),
		axis=1,
	)

	logger.info(
		"Outlier terdeteksi: %s baris (strategi=%s)",
		int(df["Is_Outlier"].sum()),
		OUTLIER_STRATEGY,
	)

	if OUTLIER_STRATEGY == "drop":
		rejected = df[df["Is_Outlier"]].assign(Alasan=alasan[df["Is_Outlier"]])
		return df[~df["Is_Outlier"]].copy(), rejected

	return df, df.iloc[0:0].assign(Alasan=pd.Series(dtype="string"))


# ============================================================
# PRINT RAW DATA SUMMARY
# ============================================================


def print_raw_summary(
	df: pd.DataFrame,
) -> None:
	"""Menampilkan ringkasan data mentah."""
	print("\n===== HEAD =====")
	print(df.head())

	print("\n===== SHAPE =====")
	print(df.shape)

	print("\n===== DATA TYPES =====")
	print(df.dtypes)

	print("\n===== INFO =====")
	df.info()

	print("\n===== MISSING VALUES =====")
	print(df.isna().sum())

	print("\n===== DUPLICATES =====")
	print(
		"Jumlah exact duplicate:",
		df.duplicated().sum(),
	)


# ============================================================
# PRINT CLEANING SUMMARY
# ============================================================


def print_cleaning_summary(
	df: pd.DataFrame,
) -> None:
	"""Menampilkan hasil parsing dan standardisasi."""
	print("\n===== HARGA =====")
	print(df[["Harga_Str", "Harga"]])

	print("\n===== JUMLAH =====")
	print(df[["Invoice_ID", "Jumlah"]])

	print("\n===== TANGGAL =====")
	print(df[["Invoice_ID", "Tanggal_Order"]])

	print("\n===== KOTA =====")
	print(df["Kota_Toko"].value_counts(dropna=False))

	print("\n===== KATEGORI =====")
	print(df["Kategori"].value_counts(dropna=False))

	print("\n===== MISSING SETELAH CLEANING FIELD =====")
	print(df.isna().sum())


def print_invalid_values(
	df: pd.DataFrame,
) -> None:
	"""Menampilkan nilai noise (harga/jumlah mustahil) yang dikosongkan."""
	print("\n===== NOISE: HARGA TIDAK VALID =====")
	print(df.loc[df["Harga_Invalid"], ["Invoice_ID", "Harga_Str"]])

	print("\n===== NOISE: JUMLAH TIDAK VALID =====")
	print(df.loc[df["Jumlah_Invalid"], ["Invoice_ID", "Kategori"]])


def print_duplicate_summary(
	df: pd.DataFrame,
) -> None:
	"""Menampilkan baris yang terdeteksi sebagai duplicate."""
	duplicate_rows = df[df.duplicated(keep=False)]

	print("\n===== EXACT DUPLICATES =====")
	print(duplicate_rows)

	print("\nInvoice_ID yang duplikat:")
	print(duplicate_rows["Invoice_ID"].tolist())


def print_outlier_summary(
	df_final: pd.DataFrame,
	rejected: pd.DataFrame,
) -> None:
	"""Menampilkan ringkasan outlier (yang dibuang maupun yang hanya ditandai)."""
	print("\n===== OUTLIER =====")
	print(f"Strategi: {OUTLIER_STRATEGY} | IQR x {IQR_MULTIPLIER} (skala log)")

	flagged = pd.concat([
		df_final[df_final["Is_Outlier"]],
		rejected,
	])

	if flagged.empty:
		print("Tidak ada outlier terdeteksi.")
		return

	print(
		flagged[
			[
				"Invoice_ID",
				"Kategori",
				"Harga",
				"Jumlah",
				"Total_Pendapatan",
				"Is_Outlier",
			]
		],
	)


def print_rejected_summary(
	rejected: pd.DataFrame,
) -> None:
	"""Menampilkan seluruh baris yang dibuang beserta alasannya."""
	print("\n===== BARIS DIBUANG =====")

	if rejected.empty:
		print("Tidak ada.")
		return

	print(rejected[["Invoice_ID", "Alasan"]])

	print("\nRingkasan per alasan:")
	print(rejected["Alasan"].value_counts())


def print_time_features(
	df: pd.DataFrame,
) -> None:
	"""Menampilkan fitur waktu."""
	print("\n===== TIME FEATURES =====")
	print(df[["Invoice_ID", "Tanggal_Order", "Month", "Day_of_Week"]])


def print_missing_quantity(
	df: pd.DataFrame,
) -> None:
	"""Menampilkan transaksi dengan quantity missing."""
	print("\n===== MISSING QUANTITY =====")
	print(
		df[df["Jumlah_Missing"]][
			[
				"Invoice_ID",
				"Kategori",
				"Jumlah",
				"Jumlah_Missing",
			]
		],
	)


# ============================================================
# CALCULATE REVENUE
# ============================================================


def calculate_revenue(
	df: pd.DataFrame,
) -> tuple[
	float,
	pd.Series,
]:
	"""Menghitung total pendapatan dan pendapatan per kategori."""
	net_revenue = df["Total_Pendapatan"].sum()

	revenue_by_category = df.groupby("Kategori")["Total_Pendapatan"].sum()

	return (
		net_revenue,
		revenue_by_category,
	)


def print_revenue_summary(
	df: pd.DataFrame,
) -> float:
	"""Menampilkan total pendapatan."""
	net_revenue = df["Total_Pendapatan"].sum()

	print("\n===== TOTAL PENDAPATAN =====")

	print(
		df[
			[
				"Invoice_ID",
				"Harga",
				"Jumlah",
				"Jumlah_Missing",
				"Total_Pendapatan",
			]
		],
	)

	print("\n===== NET REVENUE =====")

	print(f"Net Revenue: Rp {net_revenue:,.0f}")

	return net_revenue


def print_revenue_by_category(
	revenue_by_category: pd.Series,
) -> None:
	"""Menampilkan pendapatan berdasarkan kategori."""
	print("\n===== REVENUE BY CATEGORY =====")

	print(revenue_by_category.sort_values(ascending=False))


def print_quantity_statistics(
	df: pd.DataFrame,
) -> None:
	"""Menampilkan statistik jumlah transaksi."""
	mean_jumlah = df["Jumlah"].mean()

	median_jumlah = df["Jumlah"].median()

	print("\n===== STATISTIK JUMLAH =====")

	print(f"Mean Jumlah: {mean_jumlah:.2f}")

	print(f"Median Jumlah: {median_jumlah:.2f}")


# ============================================================
# SAVE DATA
# ============================================================


def save_clean_data(
	df: pd.DataFrame,
	rejected: pd.DataFrame,
) -> None:
	"""Menyimpan data bersih dan data yang dibuang."""
	df.to_csv(
		OUTPUT_FILE,
		index=False,
		encoding="utf-8",
	)

	rejected.to_csv(
		REJECTED_FILE,
		index=False,
		encoding="utf-8",
	)

	logger.info(
		"Clean dataset disimpan ke %s",
		OUTPUT_FILE,
	)

	logger.info(
		"Rejected dataset (%s baris) disimpan ke %s",
		len(rejected),
		REJECTED_FILE,
	)


# ============================================================
# PRINT FINAL DATA
# ============================================================


def print_final_data(
	df: pd.DataFrame,
) -> None:
	"""Menampilkan hasil akhir data cleaning."""
	print(f"\n{SECTION_SEPARATOR}")
	print("FINAL CLEAN DATA")
	print(SECTION_SEPARATOR)

	print(df.to_string())

	print("\nShape akhir:")
	print(df.shape)

	print("\nKolom:")
	print(df.columns.tolist())

	print("\nData types:")
	print(df.dtypes)

	print("\nMissing values:")
	print(df.isna().sum())

	print(f"\n{SECTION_SEPARATOR}")
	print("FINAL DUPLICATES")
	print(SECTION_SEPARATOR)

	print(
		"Jumlah exact duplicate:",
		df.duplicated().sum(),
	)


# ============================================================
# VISUALIZATION
# ============================================================


def visualize_revenue(
	revenue_by_category: pd.Series,
) -> None:
	"""Membuat grafik pendapatan berdasarkan kategori."""
	print(f"\n{SECTION_SEPARATOR}")
	print("VISUALISASI REVENUE PER KATEGORI")
	print(SECTION_SEPARATOR)

	sns.set_theme(style="whitegrid")

	ax = revenue_by_category.sort_values(ascending=False).plot(
		kind="bar",
		figsize=(10, 5),
	)

	ax.set_title("Revenue per Kategori")

	ax.set_xlabel("Kategori")

	ax.set_ylabel("Pendapatan (Rp)")

	plt.xticks(rotation=0)

	plt.tight_layout()

	plt.show()


def print_completion_message(
	net_revenue: float,
) -> None:
	"""Menampilkan informasi output dan revenue."""
	print(f"\n{SECTION_SEPARATOR}")
	print("DATA CLEANING SELESAI")
	print(SECTION_SEPARATOR)

	print(f"Clean CSV    : {OUTPUT_FILE}")

	print(f"Rejected CSV : {REJECTED_FILE}")

	print(f"Log file     : {LOG_FILE}")

	print(f"Net Revenue  : Rp {net_revenue:,.0f}")


# ============================================================
# MAIN PIPELINE
# ============================================================


def run_pipeline() -> None:
	"""Menjalankan seluruh proses cleaning dan analisis."""
	print(f"{SEPARATOR}\nMEMULAI PROSES PIPELINE DATA CLEANING\n{SEPARATOR}")

	# --------------------------------------------------------
	# LOAD DATA
	# --------------------------------------------------------

	df_raw = load_data()

	logger.info(
		"Shape awal: %s",
		df_raw.shape,
	)

	print_raw_summary(df_raw)

	# --------------------------------------------------------
	# PARSE AND CLEAN NOISE
	# --------------------------------------------------------

	df_parsed = parse_dates(df_raw)

	df_cleaned = clean_fields(df_parsed)

	print_cleaning_summary(df_cleaned)

	print_invalid_values(df_cleaned)

	print_duplicate_summary(df_cleaned)

	# --------------------------------------------------------
	# INVALID ROWS AND DUPLICATES
	# --------------------------------------------------------

	df_final, rejected_rows = handle_missing_and_duplicates(df_cleaned)

	# --------------------------------------------------------
	# FEATURE ENGINEERING
	# --------------------------------------------------------

	df_final = create_features(df_final)

	print_time_features(df_final)

	print_missing_quantity(df_final)

	# --------------------------------------------------------
	# OUTLIER HANDLING
	# --------------------------------------------------------

	df_final, rejected_outliers = handle_outliers(df_final)

	print_outlier_summary(
		df_final,
		rejected_outliers,
	)

	rejected_all = pd.concat(
		[rejected_rows, rejected_outliers],
		ignore_index=False,
	)

	print_rejected_summary(rejected_all)

	# --------------------------------------------------------
	# REVENUE ANALYSIS
	# --------------------------------------------------------

	net_revenue = print_revenue_summary(df_final)

	_, revenue_by_category = calculate_revenue(df_final)

	print_revenue_by_category(revenue_by_category)

	print_quantity_statistics(df_final)

	# --------------------------------------------------------
	# SAVE
	# --------------------------------------------------------

	save_clean_data(df_final, rejected_all)

	print_final_data(df_final)

	# --------------------------------------------------------
	# VISUALIZATION
	# --------------------------------------------------------

	visualize_revenue(revenue_by_category)

	print_completion_message(net_revenue)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
	run_pipeline()
