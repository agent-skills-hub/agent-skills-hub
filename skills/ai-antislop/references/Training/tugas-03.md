# Tugas-03 — HANA REBLOOM (improve dari 2 kegagalan)

**Brief: perbaiki 7 downgrade tugas-02.** Prinsip (dari
`traning-gagal-02/README.md`): wajah & konten dulu, 3D sebagai
pendukung — bukan sebaliknya.

**Keputusan agen (stated upfront):**
1. Hero = wajah Waguri full-bleed (ep1.jpg + Ken Burns + shade),
   seperti tugas-01 yang hook-nya terbukti. 3D tidak tampil di hero.
2. Section readable opaque kembali (cerita/transformasi/footer);
   3D hanya tampil di momen khusus: section artefak + penutup +
   sela galeri (section transparan). [Koreksi: body tetap transparan,
   bukan opaque — yang opaque section-nya.]
3. Core diperkecil (0.95) + diredupkan; kelopak bertekstur lembut
   (CanvasTexture radial, additive) — bunuh kesan confetti.
4. Palet hangat dipertahankan; substansi (fakta, sumber, caption)
   tidak dipangkas.
5. Vendor Three.js tetap (real 3D tetap syarat) — sekarang justified
   sebagai pendukung, bukan pembawa acara.

**Status:** LULUS BERSYARAT (27 Sep 2026, verdict owner: "berhasil
but need improvement"). Lulus pertama di track ini.
**Need improvement (tercatat, belum dikerjakan):**
1. Konfirmasi palet hangat + hero settle dari owner (baru di-push,
   belum dinilai).
2. Misteri "foto judat kedut" + cara buka (file vs localhost) —
   jawaban owner pending; guard file:// sudah dipasang.
3. Reduced-motion: masih code-verified, belum browser-test.
**Improvement 27 Sep (3D bermakna):** still profil→depan back-to-back
di ruang 3D, grup berputar 0→180° di-drive scroll + daun romantis
(tekstur lembut, fall lambat, hue keemasan). Jujur: bukan model
karakter 3D (tidak feasible tanpa merusak rupa) — turn sinematik
dari imagery asli. Verified: turn terbaca di screenshot, zero error.
**Revert 27 Sep (verdict owner: turn jelek, 3D lama lebih bagus):**
turn-group + daun dicabut, kembali ke inti + kelopak lembut.
Copy chapter dikembalikan. Verified: screenshot Babak II (inti,
ring, kelopak lembut), zero error.
