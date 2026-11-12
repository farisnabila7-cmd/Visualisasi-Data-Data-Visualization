"""Dashboard analisis dan visualisasi data penumpang Titanic."""

import pathlib
from typing import IO, TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
	from streamlit.delta_generator import DeltaGenerator
else:
	DeltaGenerator = object

# ==================== KONFIGURASI ====================

st.set_page_config(
	page_title="Dashboard Analisis Penumpang Titanic",
	layout="wide",
)

# Palet Okabe-Ito (aman bagi buta warna)
WARNA = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7"]

NAMA_FILE = "titanic_bersih.csv"

JALUR_COLAB = pathlib.Path(
	"/content/drive/MyDrive/Colab Notebooks/Visualisasi Data/titanic_bersih.csv",
)

# Konstanta untuk menghindari duplikasi string
STATUS_LABEL = "Status Selamat"
JUMLAH_LABEL = "Jumlah Penumpang"

# ==================== FUNGSI PENDUKUNG ====================


def tampilkan_grafik(
	container: DeltaGenerator,
	fig: go.Figure,
) -> None:
	"""Menampilkan grafik Plotly dengan kompatibilitas Streamlit."""
	try:
		container.plotly_chart(fig, width="stretch")
	except TypeError:
		container.plotly_chart(fig, use_container_width=True)


@st.cache_data
def load_data(sumber: str | pathlib.Path | IO[bytes]) -> pd.DataFrame:
	"""Memuat dataset Titanic dan menambahkan kolom pendukung visualisasi."""  # ruff: ignore[docstring-missing-returns]
	df = pd.read_csv(sumber)

	df["jumlah_penumpang"] = 1
	df[STATUS_LABEL] = df["survived"].map({0: "Meninggal", 1: "Selamat"})

	return df


# ==================== PEMUATAN DATA ====================

folder_app = pathlib.Path(__file__).resolve().parent

kandidat = [
	folder_app / NAMA_FILE,
	pathlib.Path(NAMA_FILE),
	JALUR_COLAB,
]

sumber_data: str | pathlib.Path | IO[bytes] | None = next(
	(path for path in kandidat if path.exists()),
	None,
)

if sumber_data is None:
	st.info(f"Berkas {NAMA_FILE} tidak ditemukan. Unggah berkas CSV di bawah ini.")

	unggahan = st.file_uploader(
		"Unggah titanic_bersih.csv",
		type="csv",
	)

	if unggahan is None:
		st.stop()

	sumber_data = unggahan

d = load_data(sumber_data)

# ==================== JUDUL DASHBOARD ====================

st.title("Dashboard Analitis Karakteristik Penumpang Titanic")

st.caption(
	"Sumber Data: Cleaned Titanic Dataset | Implementasi Praktikum Visualisasi Data",
)

# ==================== SIDEBAR FILTER ====================

st.sidebar.header("Konfigurasi Filter")

semua_pelabuhan = sorted(d["embarked"].dropna().unique())

pelabuhan = st.sidebar.multiselect(
	"Pelabuhan Keberangkatan",
	semua_pelabuhan,
	default=semua_pelabuhan,
)

semua_gender = sorted(d["sex"].dropna().unique())

gender = st.sidebar.multiselect(
	"Jenis Kelamin (Sex)",
	semua_gender,
	default=semua_gender,
)

# Filter rentang tarif tiket
tarif_min = float(d["fare"].min())
tarif_max = float(d["fare"].max())

rentang_tarif = st.sidebar.slider(
	"Rentang Tarif Tiket ($)",
	min_value=tarif_min,
	max_value=tarif_max,
	value=(tarif_min, tarif_max),
)

# ==================== FILTER DATA ====================

x = d[
	d["embarked"].isin(pelabuhan)
	& d["sex"].isin(gender)
	& d["fare"].between(
		rentang_tarif[0],
		rentang_tarif[1],
	)
]

if x.empty:
	st.warning(
		"Peringatan: Kombinasi filter menghasilkan data kosong. "
		"Silakan sesuaikan kembali filter Anda.",
	)
	st.stop()

# ==================== BARIS KPI ====================

k1, k2, k3, k4 = st.columns(4)

total_penumpang = len(x)
proporsi_selamat = (x["survived"].sum() / total_penumpang) * 100
rata_usia = x["age"].mean()
median_tarif = x["fare"].median()

k1.metric(
	"Total Penumpang",
	f"{total_penumpang:,} Jiwa",
)

k2.metric(
	"Tingkat Kelangsungan Hidup",
	f"{proporsi_selamat:.1f} %",
)

k3.metric(
	"Rata-rata Usia",
	f"{rata_usia:.1f} Tahun",
)

k4.metric(
	"Median Tarif Tiket ($)",
	f"{median_tarif:,.2f}",
)

# ==================== BARIS VISUALISASI 1 ====================

b1, b2 = st.columns(2)

# Histogram kelas tiket berdasarkan status keselamatan
fig_pclass = px.histogram(
	x,
	x="pclass",
	color=STATUS_LABEL,
	barmode="group",
	color_discrete_sequence=[WARNA[3], WARNA[0]],
	title="Peluang Selamat Lebih Tinggi pada Penumpang Kelas Atas",
	labels={
		"pclass": "Kelas Tiket",
		"count": "Jumlah Penumpang",
		STATUS_LABEL: "Status",
	},
)

tampilkan_grafik(b1, fig_pclass)

# Histogram distribusi usia penumpang
fig_age = px.histogram(
	x,
	x="age",
	nbins=20,
	color_discrete_sequence=[WARNA[1]],
	title="Struktur Demografi Didominasi Usia Produktif (20-35 Tahun)",
	labels={
		"age": "Rentang Usia (Tahun)",
		"count": "Frekuensi",
	},
)

tampilkan_grafik(b2, fig_age)

# Sunburst berdasarkan gender, kelas, dan status keselamatan
fig_sunburst = px.sunburst(
	x,
	path=["sex", "pclass", STATUS_LABEL],
	values="jumlah_penumpang",
	color_discrete_sequence=WARNA,
	title=("Hierarki Kelangsungan Hidup Berdasarkan Interaksi Gender & Kelas Kamar"),
)

tampilkan_grafik(st, fig_sunburst)

# ==================== BARIS VISUALISASI 2 ====================

c1, c2 = st.columns(2)

# Scatter plot tarif tiket terhadap usia
fig_scatter = px.scatter(
	x,
	x="age",
	y="fare",
	color=STATUS_LABEL,
	color_discrete_map={
		"Selamat": WARNA[0],
		"Meninggal": WARNA[3],
	},
	opacity=0.7,
	title="Tarif Tiket Tinggi Cenderung Dimiliki Penumpang yang Selamat",
	labels={
		"age": "Usia (Tahun)",
		"fare": "Tarif Tiket ($)",
		STATUS_LABEL: "Status",
	},
	hover_data=["pclass", "sex"],
)

tampilkan_grafik(c1, fig_scatter)

# Diagram pie proporsi penumpang berdasarkan pelabuhan
asal = x["embarked"].value_counts().reset_index()

asal.columns = [
	"Pelabuhan",
	JUMLAH_LABEL,
]

fig_pie = px.pie(
	asal,
	names="Pelabuhan",
	values=JUMLAH_LABEL,
	color_discrete_sequence=WARNA,
	hole=0.4,
	title="Mayoritas Penumpang Berangkat dari Satu Pelabuhan Utama",
)

tampilkan_grafik(c2, fig_pie)

# ==================== RINCIAN DATA ====================

with st.expander("Klik untuk Menampilkan Lembar Rincian Data Transaksional"):
	st.dataframe(x)
