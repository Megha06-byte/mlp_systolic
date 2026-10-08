Training: 15 epochs x 1437 samples = 21555 per-sample SGD steps

| Platform | Time (ms) | ns / sample | Test acc | Bit-exact |
|---|---|---|---|---|
| CPU: NumPy, batch 1 (Python loop) | 729.16 | 33828 | 96.67% | n/a (float) |
| CPU: NumPy, mini-batch 32 (vectorised) | 42.60 | 1976 | 96.11% | n/a (float) |
| CPU: C -O3 -march=native, batch 1 | 23.27 | 1079 | 96.67% | n/a (float) |
| FPGA engine P=1,Q=1,R=1 @ 100 MHz (est.) | 20.909 | 970 | 96.94% | yes |
| FPGA engine P=1,Q=1,R=1 @ 200 MHz (est.) | 10.455 | 485 | 96.94% | yes |
| FPGA engine P=8,Q=4,R=5 @ 100 MHz (est.) | 4.527 | 210 | 96.94% | yes |
| FPGA engine P=8,Q=4,R=5 @ 200 MHz (est.) | 2.263 | 105 | 96.94% | yes |
| FPGA engine P=16,Q=8,R=10 @ 100 MHz (est.) | 3.018 | 140 | 96.94% | yes |
| FPGA engine P=16,Q=8,R=10 @ 200 MHz (est.) | 1.509 | 70 | 96.94% | yes |
| FPGA engine P=64,Q=16,R=10 @ 100 MHz (est.) | 2.156 | 100 | 96.94% | yes |
| FPGA engine P=64,Q=16,R=10 @ 200 MHz (est.) | 1.078 | 50 | 96.94% | yes |

**Speedup of the engine (training loop only / end-to-end incl. data load + weight readback)**

| Engine config | vs NumPy, batch 1 (Python loop) @100 MHz | vs NumPy, batch 1 (Python loop) @200 MHz | vs NumPy, mini-batch 32 (vectorised) @100 MHz | vs NumPy, mini-batch 32 (vectorised) @200 MHz | vs C -O3 -march=native, batch 1 @100 MHz | vs C -O3 -march=native, batch 1 @200 MHz |
|---|---|---|---|---|---|---|
| P=1,Q=1,R=1 | 34.9x / 34.4x | 69.7x / 68.9x | 2.0x / 2.0x | 4.1x / 4.0x | 1.1x / 1.1x | 2.2x / 2.2x |
| P=8,Q=4,R=5 | 161.1x / 152.1x | 322.2x / 304.1x | 9.4x / 8.9x | 18.8x / 17.8x | 5.1x / 4.9x | 10.3x / 9.7x |
| P=16,Q=8,R=10 | 241.6x / 221.9x | 483.2x / 443.8x | 14.1x / 13.0x | 28.2x / 25.9x | 7.7x / 7.1x | 15.4x / 14.2x |
| P=64,Q=16,R=10 | 338.3x / 300.8x | 676.5x / 601.6x | 19.8x / 17.6x | 39.5x / 35.1x | 10.8x / 9.6x | 21.6x / 19.2x |

CPU: Intel(R) Xeon(R) Processor @ 2.80GHz (single thread). Integer reference accuracy: 96.94%.
