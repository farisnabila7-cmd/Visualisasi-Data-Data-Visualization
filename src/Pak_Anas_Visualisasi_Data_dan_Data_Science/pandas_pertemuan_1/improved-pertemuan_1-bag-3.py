# ruff: ignore[invalid-module-name]
"""Visualisasi gross revenue dan refund berdasarkan kategori."""

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
		"Kategori": "string",
		"Jumlah": "Float64",
		"Total_Pendapatan": "Float64",
	},
)


# ============================================================
# CALCULATE GROSS AND REFUND
# ============================================================

gross = df_clean[df_clean["Jumlah"] > 0].groupby("Kategori")["Total_Pendapatan"].sum()

refund = (
	df_clean[df_clean["Jumlah"] < 0].groupby("Kategori")["Total_Pendapatan"].sum().abs()
)


# ============================================================
# CREATE SUMMARY DATAFRAME
# ============================================================

summary_df = pd.DataFrame({
	"Gross Revenue (Penjualan)": gross,
	"Refund (Pengembalian)": refund,
}).fillna(0)

summary_df = summary_df.sort_values(
	"Gross Revenue (Penjualan)",
	ascending=False,
)


# ============================================================
# VISUALIZATION
# ============================================================

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 6))


# Membuat grouped bar chart
summary_df.plot(
	kind="bar",
	ax=ax,
	color=[
		"#2ecc71",
		"#e74c3c",
	],
	width=0.6,
)


# ============================================================
# BAR ANNOTATION
# ============================================================

for p in ax.patches:
	val = getattr(
		p,
		"get_height",
		lambda: 0.0,
	)()

	if val > 0:
		width = getattr(
			p,
			"get_width",
			lambda: 0.2,
		)()

		x_pos = getattr(
			p,
			"get_x",
			lambda: 0.0,
		)()

		ax.annotate(
			f"Rp {val:,.0f}",
			(
				x_pos + width / 2.0,
				val,
			),
			ha="center",
			va="center",
			xytext=(0, 8),
			textcoords="offset points",
			fontsize=9,
			fontweight="bold",
		)


# ============================================================
# CHART LABELS
# ============================================================

plt.title(
	"Gross Revenue vs Refund per Kategori",
	fontsize=14,
	pad=20,
	fontweight="bold",
)

plt.xlabel(
	"Kategori Produk",
	fontsize=12,
	labelpad=10,
)

plt.ylabel(
	"Nilai Transaksi (Rp)",
	fontsize=12,
	labelpad=10,
)

plt.xticks(
	rotation=0,
	fontsize=11,
)


# ============================================================
# Y-AXIS FORMAT
# ============================================================

ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, _: f"Rp {int(x):,}"))


# ============================================================
# LEGEND
# ============================================================

plt.legend(
	loc="upper right",
	frameon=True,
	shadow=True,
	fontsize=10,
)


# ============================================================
# DISPLAY
# ============================================================

plt.tight_layout()
plt.show()
