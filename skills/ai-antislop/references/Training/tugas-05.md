# Tugas-05 — Bukit Wisteria (Waguri diorama split)

**Brief: terapkan bank INSPIRASI, bukan reskin.** Arsitektur baru:
split editorial + diorama 3D (Expeditione), satu warna disiplin
(Undream), satu pesan per layar (Apple), satu momen 3D fully
interaktif (Bruno). Topik: Waguri (formula pemenang) + fakta
terverifikasi yang sama. Gambar: 6 stills lokal (sudah verified).

**Keputusan agen (stated upfront):**
- Layout split: panel editorial opaque kiri, diorama kanan
  (grup di x=+2.6). Mobile: stack.
- Diorama prosedural "bukit wisteria": gundukan + pohon wisteria
  (foliage instanced) + lentera batu KLIKABLE (dusk/dawn toggle) +
  jalan batu + kelopak + fireflies.
- Interaksi: drag = orbit inspeksi (stay, damped), klik lentera =
  toggle suasana, scroll = orbit sinematik + dolly.
- Satu warna disiplin: wisteria + cream di atas plum hangat.
- Anotasi playful (catatan miring + panah) untuk obat kaku.

**Status:** LULUS (27 Sep 2026, verdict owner: "bagus").
**Need improvement (tercatat, belum dikerjakan):**
1. Terlalu membosankan di beberapa bagian — section statis perlu
   variasi motion & micro-interaction.
2. Kurang tombol/kontrol — tambah UI affordance interaktif
   (contoh: navigasi galeri, toggle auto-orbit, kontrol kelopak).
   Aturan: tiap section idealnya punya minimal 1 hal yang bisa
   disentuh user.
