# ruff: ignore[invalid-module-name]
"""Visualisasi tren pendapatan bulanan dari data transaksi yang telah dibersihkan."""

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
		"Month": "Int64",
		"Total_Pendapatan": "Float64",
	},
)


# ============================================================
# GROUP REVENUE BY MONTH
# ============================================================

pendapatan_bulanan = (
	df_clean.groupby("Month", dropna=True)["Total_Pendapatan"].sum().reset_index()
)


# ============================================================
# MONTH NAME MAPPING
# ============================================================

nama_bulan = {
	1: "Januari",
	2: "Februari",
	3: "Maret",
	4: "April",
	5: "Mei",
	6: "Juni",
	7: "Juli",
	8: "Agustus",
	9: "September",
	10: "Oktober",
	11: "November",
	12: "Desember",
}

pendapatan_bulanan["Nama_Bulan"] = pendapatan_bulanan["Month"].map(nama_bulan)

pendapatan_bulanan = pendapatan_bulanan.sort_values("Month")


# ============================================================
# VISUALIZATION
# ============================================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 5))

sns.lineplot(
	data=pendapatan_bulanan,
	x="Nama_Bulan",
	y="Total_Pendapatan",
	marker="o",
	linewidth=2.5,
)

plt.title(
	"Tren Pendapatan Bulanan",
	fontsize=14,
	pad=15,
)

plt.xlabel(
	"Bulan",
	fontsize=12,
)

plt.ylabel(
	"Net Revenue (Rp)",
	fontsize=12,
)

plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"Rp {int(x):,}"))

plt.tight_layout()
plt.show()
