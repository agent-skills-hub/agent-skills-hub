# TRANING-GAGAL — Waguri Showcase (tugas-01)

**Verdict owner: GAGAL.** Diarsipkan sebagai patokan "belum cukup" —
jangan dijadikan contoh berhasil.

## Yang diserahkan

`index.html` + `assets/` (6 gambar asli dari internet). Gambar, fakta
berlabel, dan console bersih **bukan masalahnya** — yang gagal adalah
rasa motion-nya.

## Kenapa gagal

1. **Scroll animation kurang.** Isinya cuma reveal fade/slide generik +
   satu strip horizontal. Tidak ada pinned scrubbed chapter (scroll =
   kamera), tidak ada parallax multi-layer, tidak ada sequence yang
   dikendalikan progress scroll seperti referensi sinematik
   (Apple / undream / the-web-people).
2. **3D kurang.** Cuma tilt kartu saat hover. Tidak ada scene,
   kedalaman spasial, atau elemen yang merespons scroll secara 3D.
   Dibanding `example.txt` (expeditione.fun): datar.
3. **Pacing & build-up datar.** Semua konten tampil begitu section
   masuk viewport — tidak ada ketegangan yang dibangun lalu dilepas,
   tidak ada transisi antar-section yang sinematik, tidak ada satu
   momen "wah" yang diorkestrasi.
4. **Text motion minim.** Cuma judul hero per-huruf + satu paragraf
   light-up. Tidak ada text reveal dramatis yang di-scrub penuh
   sepanjang scroll.

## Pelajaran (masuk standar)

"Premium" di track ini berarti: scroll mengendalikan kamera/sequence
(pin + scrub + parallax), ada minimal satu momen 3D/spasial yang
nyata, dan pacing dibangun (tidak semua konten diobral di depan).
Definisi ini melengkapi `INSTRUKSI.md` — bukan menggantikannya.
