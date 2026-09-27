# INSTRUKSI TRAINING — Standar Web Premium

Sumber kurasi: `example.txt` (milik owner — **file itu jangan diubah**;
kurasi milik owner, standar di file ini yang boleh berkembang).

Status verifikasi per referensi ditulis apa adanya di bawah. Yang
berlabel "kurasi owner" artinya: aku tahu dari kurasi + pola
umumnya, bukan dari inspeksi langsung (situs WebGL berat tidak bisa
di-fetch sebagai teks).

## Standar umum (berlaku untuk semua deliverable web)

1. **Kaya seperti referensi, lentur gerakannya.** 3D boleh impresif
   (kelas expeditione.fun) tapi tidak kaku dan tidak berat di device:
   utamakan `transform`/`opacity` (GPU), hindari layout thrash.
2. Motion minimal wajib: entrance staggered, scroll reveal, hover
   feedback, satu cue 3D/depth, progress indicator.
3. `prefers-reduced-motion`: motion non-esensial mati total.
4. Verifikasi dengan nonton langsung (scroll penuh + hover semua kartu)
   + console zero error. Count-up/animasi harus mendarat di angka
   final yang pas, tanpa drift dan tanpa layout shift.

## Per kategori (dari example.txt)

### Contoh Web 3D — kurasi owner
- https://expeditione.fun/
- Yang dipelajari: scene berkedalaman, kamera bergerak halus, objek 3D
  sebagai centerpiece yang responsif terhadap mouse/scroll — bukan
  diorama statis. Tetap ringan: kaya kesan, murah di device.

### Scroll Animation — kurasi owner
- https://demo-03-track.hirotos.com/
- https://webflow.com/made-in-webflow/website/3d-scroll-animation-in-webflow
- https://webflow.com/made-in-webflow/3d-scroll
- Yang dipelajari: section ter-pin saat scroll, konten berubah seiring
  scroll (scrub), transisi antar chapter yang sinematik.

### Motion / UI Animation — terverifikasi
- https://stealthis.dev/showcase/ — wall berisi 199 live preview efek:
  bank ide micro-interaction & pola komponen.
- Standar: tiap elemen interaktif punya feedback visual yang terasa
  instan (<200ms), tidak ada tombol/kartu yang "mati" saat di-hover.

### GSAP Advance — terverifikasi
- https://github.com/wsmr/UI-HTML-GSAP-scrolltrigger_showcase
- 10 teknik wajib kuasai: fade-in on scroll, horizontal pin (ala Apple),
  parallax multi-layer, pin & rotate 360°, text reveal kata-per-kata,
  scale on scroll, stagger, image wipe, counter, progress indicator.
- Aturan main: `scrub` untuk animasi terikat scroll; `transform`
  bukan `left/top`; `will-change` hemat; target 60fps. Semua teknik
  self-contained — bisa dicopas polanya langsung ke project.

### Web Cinematic / Premium Elite — kurasi owner
- https://www.undreamstudio.com/projects/skybag-experience/
- https://www.the-web-people.com/projects/cinematic-scroll
- https://www.apple.com/
- https://www.samsung.com/id/
- Yang dipelajari: product showcase ala Apple — scroll = kamera, satu
  section satu pesan, tipografi raksasa, pacing lambat-penuh-keyakinan.
  Standar "premium": whitespace lega, tidak ada elemen yang berteriak
  bersamaan, transisi tidak terburu-buru.

## Cara training berjalan

1. Owner menaruh tugas di folder ini sebagai file `tugas-*.md`.
   Bank pola premium ada di `INSPIRASI.md` (wajib dibaca sebelum
   desain — diorama vs fullscreen-3D, satu warna tema, satu pesan
   per layar).
2. Agen membaca tugas + referensi terkait di atas, lalu membangun.
3. Verifikasi wajib: serve lokal → scroll penuh → screenshot →
   console zero error → matikan server. Bukti dilampirkan, bukan
   diklaim.
4. Owner menilai dari rasa (sedekat apa dengan referensi?) + checklist
   standar umum di atas.
5. Pola yang lolos maupun gagal dicatat ke `../pattern-log.md`
   (Law 11) — training yang tidak dicatat = training yang menguap.
