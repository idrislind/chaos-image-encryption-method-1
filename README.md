# Method 1 — Asymmetric + 2D Hénon Chaos Image Encryption

Prototype penelitian untuk **Enkripsi Citra Asimetris Berbasis Chaos**.

## Milestone 4 — Robustness & Key Sensitivity

Milestone 4 mempertahankan core cipher **diffusion v3** dari Milestone 3.6 dan menambahkan pengujian robustness/security yang dibutuhkan untuk eksperimen:

- RSA-3072 + OAEP-SHA256
- X25519/ECDH + HKDF-SHA256
- 2D Hénon chaotic map
- Pixel permutation
- Two-round nonlinear diffusion v3
- Exact reversible decryption pada jalur tanpa gangguan
- Histogram, entropy, correlation, PSNR, SSIM, NPCR, UACI
- RSA vs ECC benchmark + batch runner
- **Salt-and-pepper ciphertext noise attack**
- **Ciphertext block erasure / cropping-like data-loss attack**
- **JPEG recompression attack pada ciphertext**
- **Recovery MSE / PSNR / SSIM / pixel-error-rate**
- **256-bit session-key sensitivity test (1-bit key flip)**
- CSV, JSON, PNG output dan plots

> Cipher chaos pada project ini adalah metode penelitian/eksperimental. Statistik seperti entropy, NPCR, UACI, correlation, dan key sensitivity tidak merupakan bukti formal keamanan kriptografi seperti definisi IND-CPA/IND-CCA.

## 1. Setup

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

Expected Milestone 4:

```text
25 passed
```

## 2. Robustness attack test

Jalankan pada satu image:

```powershell
python .\main.py robustness `
  --image tes.png `
  --crypto rsa `
  --out-dir results\robustness_rsa
```

Default attack levels:

```text
Salt-pepper noise : 0.1%, 0.5%, 1% ciphertext bytes
Block erasure     : 1%, 5%, 10%, 25% area
JPEG quality      : 90, 70, 50
```

Untuk ECC:

```powershell
python .\main.py robustness `
  --image tes.png `
  --crypto ecc `
  --out-dir results\robustness_ecc
```

Custom levels:

```powershell
python .\main.py robustness `
  --image tes.png `
  --crypto rsa `
  --noise-levels 0.001 0.005 0.01 `
  --block-levels 0.01 0.05 0.10 0.25 `
  --jpeg-qualities 95 90 70 50 `
  --seed 42 `
  --out-dir results\robustness_custom
```

Output:

```text
results/robustness_rsa/
├── robustness.csv
├── summary.json
├── keys/
├── images/
│   ├── original.png
│   ├── cipher.png
│   ├── baseline_recovered.png
│   ├── *.attacked_cipher.png
│   └── *.recovered.png
└── plots/
    ├── recovered_psnr.png
    ├── recovered_ssim.png
    ├── pixel_error_rate.png
    └── cipher_change_percent.png
```

`robustness.csv` mencatat untuk setiap attack scenario:

```text
crypto
attack
parameter / parameter_value
cipher_npcr_percent
cipher_uaci_percent
exact_match
mse_recovered
psnr_recovered_db
ssim_recovered
pixel_error_rate_percent
```

### Catatan tentang cropping

Decryptor membutuhkan ciphertext dengan dimensi yang sama seperti image asli. Oleh karena itu Milestone 4 memakai **block erasure** untuk merepresentasikan cropping/data loss: sebuah region ciphertext dihilangkan (diisi 0), tetapi dimensi image dipertahankan. Ini mengisolasi efek hilangnya region tanpa membuat input decryptor invalid secara struktural.

### Catatan tentang JPEG

Ciphertext normal harus disimpan dalam format lossless seperti PNG. JPEG attack di sini **sengaja** melakukan recompression lossy pada ciphertext untuk mengukur bagaimana error kompresi menyebar saat decryption.

## 3. Key sensitivity

Command ini mengisolasi core cipher dari RSA/ECC. Pada setiap trial, session key 256-bit diubah **tepat 1 bit**, sedangkan plaintext dan nonce dipertahankan sama.

```powershell
python .\main.py key-sensitivity `
  --image tes.png `
  --trials 20 `
  --out-dir results\key_sensitivity
```

Output:

```text
results/key_sensitivity/
├── trials.csv
├── summary.json
├── images/
│   ├── original.png
│   ├── cipher_original_key.png
│   ├── cipher_one_bit_key_change.png
│   └── wrong_key_recovered.png
└── plots/
    ├── npcr_by_trial.png
    ├── uaci_by_trial.png
    ├── wrong_key_ssim.png
    └── wrong_key_pixel_error.png
```

Yang diperiksa:

```text
cipher NPCR / UACI
wrong-key MSE / PSNR / SSIM
wrong-key pixel error rate
wrong-key exact match count
```

Expected behavior secara kualitatif:

- perubahan 1 bit key menghasilkan ciphertext yang sangat berbeda;
- wrong key tidak mengembalikan image asli;
- NPCR/UACI harus dievaluasi sebagai distribusi multi-trial, bukan satu angka tunggal.

## 4. Differential stability v3

Validasi multi-key tetap tersedia:

```powershell
python .\main.py differential-sweep `
  --image tes.png `
  --trials 30 `
  --out-dir results\diffusion_sweep
```

Milestone 3.6 dipertahankan karena hasil pilot menunjukkan diffusion v3 menghilangkan outlier ekstrem yang ditemukan pada v2.

## 5. RSA vs ECC benchmark

```powershell
python .\main.py benchmark `
  --image tes.png `
  --out-dir results\benchmark_test
```

Batch:

```powershell
python .\main.py batch-benchmark `
  --input-dir data\sample `
  --out-dir results\pilot `
  --crypto all `
  --limit 20
```

Format input yang didukung batch:

```text
.png .jpg .jpeg .bmp .tif .tiff .pgm .ppm .webp .avif
```

AVIF bergantung pada dukungan Pillow di environment.

## 6. Contoh sanity result internal Milestone 4

Sanity test internal menggunakan image synthetic, bukan hasil dataset penelitian:

```text
Key sensitivity, 10 trial:
Cipher NPCR mean/std : ~99.61 / 0.03 %
Cipher UACI mean/std : ~33.59 / 0.17 %
Wrong-key SSIM mean  : ~0.007
Wrong-key pixel error: ~99.61 %
Wrong-key exact      : 0/10
```

Pada robustness smoke test, noise kecil menghasilkan degradasi bertahap, sementara JPEG recompression terhadap ciphertext menyebabkan recovery sangat rusak. Angka smoke test ini hanya untuk memverifikasi implementasi, bukan untuk dilaporkan sebagai hasil penelitian.

## 7. Interpretasi robustness

Pengujian ini **tidak mengasumsikan cipher harus kebal terhadap JPEG/cropping/noise**. Tujuannya adalah mengukur:

1. seberapa besar ciphertext berubah akibat gangguan;
2. bagaimana perubahan tersebut menyebar pada image hasil decryption;
3. PSNR/SSIM/MSE/pixel-error-rate dari recovered image terhadap original.

Untuk diffusion kuat, error propagation yang besar setelah ciphertext rusak dapat merupakan konsekuensi desain, bukan otomatis kegagalan algoritma.

## 8. Next — Milestone 5

Setelah hasil Phase 4 tervalidasi pada beberapa sample image:

- dataset runner UCID / BOSSbase / BOWS-2;
- aggregate robustness per dataset;
- final tables/plots;
- final experiment report;
- packaging source code + reproducibility instructions.
