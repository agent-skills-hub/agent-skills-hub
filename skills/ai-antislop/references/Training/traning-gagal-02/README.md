# TRANING-GAGAL-02 — HANA Three.js Experience (tugas-02)

**Verdict owner: GAGAL — malah downgrade dari tugas-01.** Diarsipkan
sebagai patokan kegagalan kedua: teknis naik, art direction turun.

## Yang diserahkan

Site Three.js/WebGL real (bukan CSS): inti icosahedron physical,
600 kelopak InstancedMesh, starfield, 4 lampu, kamera keyframe
di-drive scroll, 3 interaksi orisinal. Semua terverifikasi jalan,
console zero error. Tetap GAGAL — masalahnya bukan teknis.

## Kenapa gagal (7 downgrade vs tugas-01)

1. **Hook emosional hilang.** Hero = blob ungu abstrak, bukan wajah
   Waguri. Fan showcase yang 1 detik pertama tidak menunjukkan
   siapa tokohnya = bunuh diri.
2. **Objek 3D generik.** Icosahedron ungu bisa jadi apa saja — tidak
   terhubung ke bunga, wisteria, atau Waguri. Konsep HANA cuma hidup
   di teks.
3. **Kelopak = confetti.** Plane kotak tanpa tekstur, warna flat.
   Dari dekat terlihat murahan.
4. **Readability dikorbankan.** Teks bertarung dengan partikel di
   belakangnya. `text-shadow` itu perban, bukan solusi.
5. **Konten menipis.** Substansi riset (tabel spek, verdict, pola
   stealth→reveal) diganti chapter lore berisi vibes.
6. **Bobot naik, payoff turun.** +670KB vendor + loop GPU terus-menerus
   untuk hasil yang kalah berkesan dari still anime 150KB.
7. **Palet bergeser dingin.** Ungu neon gelap = rasa "template WebGL",
   bukan rasa show-nya (cream/blush hangat).

## Pelajaran (tambahan untuk INSTRUKSI)

Teknologi tidak menyelamatkan art direction. Urutan yang benar:
**wajah & konten dulu (hook + substansi), 3D sebagai pendukung** —
bukan sebaliknya. Real WebGL tanpa identitas = demo generik.
Aturan ini sekarang tertulis di `references/domains.md` (v1.9+ follow-up).
