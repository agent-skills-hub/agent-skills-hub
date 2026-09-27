# TRANING-GAGAL-03 — Jumat Si Umi (tugas-04)

**Verdict owner (27 Sep 2026): GAGAL — "masih cacat".** Perbandingan
langsung dari owner: **tugas-03 (Waguri) lebih berhasil.**

## Yang diserahkan

Konsep narrative timeline "Jumat Si Umi" (Pagi papan peringkat /
Siang slider LN⇄Anime / Sore 2 babak 3D / Malam transformasi).
Teknis terverifikasi (zero error, slider jalan, anti-crop bekerja).
Tetap GAGAL di rasa.

## Yang diketahui

- Detail cacat spesifik: pending penjelasan owner.
- Pembanding: tugas-03 Waguri dinilai lebih berhasil — hook wajah
  full-bleed + palet hangat + 3D pendukung terbukti formula yang
  lebih aman daripada narasi timeline + palet dingin.

## Alasan rinci (verdict owner 27 Sep 2026 + self-audit)

**Dari owner:**
1. **Gambar kurang** — tidak berkesan / tidak jelas.
2. **Text kurang** — copy tipis, tidak ada kedalaman.
3. **Animasi kurang** — tidak ada yang baru.
4. **Konsep sama** — terasa seperti tugas-03 di-reskin.
5. **Burik** — gambar pecah/low-res di layar desktop.

**Temuan self-audit (disuruh cari sendiri):**
6. Skeleton 1:1 dengan tugas-03 (hero-grid, story-words, chapters,
   trans-stage, closing, marquee, footer) — restruktur narasi tidak
   mengubah rasa karena tulangnya identik.
7. Source images memang kecil (stand 700px, LN 439px) — di-stretch ke
   layar 1280px+ pasti burik; blur-fill background malah menegaskan
   kesan blur.
8. Multiply blend menggelapkan kulit Umi di hero — terlihat kotor,
   bukan moody.
9. Teks outline "No.2" (transparent + stroke) murahan dan susah dibaca.
10. 3D = sistem yang sama persis, cuma ganti warna (inti, ring,
   kelopak, kamera, interaksi — semua sama). Recolor bukan konsep baru.
11. Slider membandingkan bust vs full-body — skala timpang, canggung.
12. Tidak ada interaksi yang benar-benar baru (drag, cursor-light,
   slider = pola lama dengan baju baru).

## Pelajaran

Struktur baru tidak otomatis lebih baik dari struktur terbukti.
Eksperimen konsep wajib membawa minimal satu elemen yang sudah
terbukti lulus (wajah dominan / palet hangat), bukan mengganti
semuanya sekaligus. Tambahan dari arsip ini: reskin + ganti warna
3D tidak dihitung sebagai konsep baru; gambar kecil jangan
di-stretch ke layar besar; multiply blend di atas bg gelap
mengotori warna kulit.
