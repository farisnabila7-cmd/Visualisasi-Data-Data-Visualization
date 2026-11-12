# ruff: file-ignore[print]
# ruff: file-ignore[docstring-missing-returns]
# ruff: file-ignore[mixed-case-variable-in-global-scope]
"""Game tebak angka berbasis terminal."""

import secrets
import time
from typing import TypedDict

GLOBAL_HISTORY: list[PlayerStats] = []

MAX_ATTEMPTS = 5
BASE_SCORE = 100
secrets_MIN = 1
secrets_MAX = 100


class PlayerStats(TypedDict):
	"""Struktur statistik pemain."""

	player: str
	score: int
	attempts: list[int]


def inisialisasi_game(
	player_name: str,
	stats: PlayerStats | None = None,
) -> PlayerStats:
	"""Siapkan data awal pemain."""
	if stats is None:
		return {
			"player": player_name,
			"score": 0,
			"attempts": [],
		}

	stats["player"] = player_name
	return stats


def hitung_skor_bonus(
	attempts: list[int],
	base_score: int = BASE_SCORE,
) -> int:
	"""Hitung bonus berdasarkan jumlah percobaan."""
	faktor = len(attempts)
	sisa = MAX_ATTEMPTS - faktor

	if sisa <= 0:
		return 0

	return base_score // sisa


def dapatkan_petunjuk(angka_rahasia: int, tebakan: int) -> str:
	"""Berikan petunjuk berdasarkan tebakan pemain."""
	if tebakan < angka_rahasia:
		return "Terlalu kecil!"

	if tebakan > angka_rahasia:
		return "Terlalu besar!"

	return "Tepat sekali!"


def validasi_input(teks: str) -> bool:
	"""Validasi bahwa input hanya berisi angka."""
	return teks.isdigit()


def muat_pesan_selamat() -> str:
	"""Ambil pesan kemenangan secara acak."""
	pesan = ["Hebat!", "Luar biasa!", "Kamu beruntung!"]
	return secrets.choice(pesan)


def jalankan_sesi(nama: str) -> None:
	"""Jalankan satu sesi permainan."""
	print(f"\nSelamat datang {nama}! Tebak angka dari 1 sampai 100.")

	player_stats = inisialisasi_game(nama)
	angka_rahasia = secrets.randint(secrets_MIN, secrets_MAX)
	game_over = False

	while not game_over:
		tebakan_user = input("Masukkan tebakan Anda: ")

		if not validasi_input(tebakan_user):
			print("Masukkan angka yang valid!")
			continue

		tebakan = int(tebakan_user)
		attempts = player_stats["attempts"]
		attempts.append(tebakan)

		if tebakan == angka_rahasia:
			print(muat_pesan_selamat())

			bonus = hitung_skor_bonus(attempts)
			player_stats["score"] += BASE_SCORE + bonus

			print(f"Total skor Anda: {player_stats['score']}")
			game_over = True
		else:
			petunjuk = dapatkan_petunjuk(angka_rahasia, tebakan)
			print(petunjuk)

		if len(attempts) >= MAX_ATTEMPTS and not game_over:
			print(f"Kesempatan habis! Angka rahasia adalah: {angka_rahasia}")
			break

	GLOBAL_HISTORY.append(player_stats)


def tampilkan_papan_skor() -> None:
	"""Tampilkan riwayat permainan."""
	print("\n=== PAPAN SKOR GLOBAL ===")

	for sesi in GLOBAL_HISTORY:
		print(f"Pemain: {sesi['player']} | Mencoba: {len(sesi['attempts'])} kali")


def main() -> None:
	"""Jalankan aplikasi."""
	print("Sistem Game Tebak Angka Pro v1.0")

	para_pemain = ["Andi", "Budi", "Cici"]

	for pemain in para_pemain:
		jalankan_sesi(pemain)
		time.sleep(0.5)

	tampilkan_papan_skor()


if __name__ == "__main__":
	main()
