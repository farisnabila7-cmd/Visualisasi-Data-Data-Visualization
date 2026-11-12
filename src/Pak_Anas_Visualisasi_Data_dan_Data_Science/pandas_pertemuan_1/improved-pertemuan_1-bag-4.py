# ruff: file-ignore[print]  # ruff: ignore[invalid-module-name]

"""Visualisasi distribusi nilai transaksi penjualan positif per invoice."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ============================================================
# CONFIGURATION
# ============================================================

try:
	# BASE_DIR = Path(__file__).resolve().parents[2]  # ruff: ignore[commented-out-code]
	BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
	BASE_DIR = Path.cwd()

OUTPUT_FILE = BASE_DIR / "output" / "data_transaksi_clean.csv"


# ============================================================
# LOAD CLEAN DATA
# ============================================================

df_clean = pd.read_csv(
	OUTPUT_FILE,
	dtype={
		"Invoice_ID": "string",
		"Jumlah": "Float64",
		"Total_Pendapatan": "Float64",
	},
)


# ============================================================
# CALCULATE TRANSACTION VALUE PER INVOICE
# ============================================================

transaksi_per_invoice = (
	df_clean
	.dropna(
		subset=[
			"Invoice_ID",
			"Total_Pendapatan",
		],
	)
	.groupby(
		"Invoice_ID",
		as_index=False,
	)["Total_Pendapatan"]
	.sum()
)


# ============================================================
# FILTER POSITIVE SALES TRANSACTIONS ONLY
# ============================================================

sales_df = transaksi_per_invoice[transaksi_per_invoice["Total_Pendapatan"] > 0].copy()


# ============================================================
# EXTRACT DATA FOR VISUALIZATION
# ============================================================

invoice_list = sales_df["Invoice_ID"].astype(str).tolist()

pendapatan_list = sales_df["Total_Pendapatan"].astype(float).tolist()


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("DISTRIBUSI NILAI TRANSAKSI PENJUALAN (SALES ONLY)")

print("=" * 60)

for inv, rev in zip(
	invoice_list,
	pendapatan_list,
	strict=False,
):
	print(f"{inv:<12} {rev:>12,.1f}")

print("\nStatistik nilai transaksi penjualan:")

print(sales_df["Total_Pendapatan"].describe())


# ============================================================
# VISUALIZATION
# ============================================================

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))


# Membuat histogram distribusi transaksi penjualan
sns.histplot(
	x=pendapatan_list,
	bins=5,
	kde=True,
	ax=ax,
	color="#2ecc71",
)


# ============================================================
# X-AXIS FORMAT
# ============================================================

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"Rp {int(x):,}"))


# ============================================================
# CHART LABELS
# ============================================================

plt.title(
	"Distribusi Nilai Total Transaksi Penjualan per Invoice",
	fontsize=14,
	pad=15,
	fontweight="bold",
)

plt.xlabel(
	"Total Nilai Penjualan (Rp)",
	fontsize=12,
)

plt.ylabel(
	"Frekuensi",
	fontsize=12,
)


# ============================================================
# DISPLAY
# ============================================================

plt.tight_layout()
plt.show()
