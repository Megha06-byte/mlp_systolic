#!/usr/bin/env python3
"""
bench.py : train the same MLP on CPU baselines and on the hardware trainer,
check the hardware is bit-exact, and report speedups.

All runs use the same algorithm: 64-16-10 MLP, MSE loss, per-sample SGD,
lr 2^-7, same initial weights, same sample order, 15 epochs.

Hardware time = engine cycles / f_clk. Fmax has NOT been measured (no Vivado
run yet), so results are given at 100 MHz (conservative, any recent FPGA) and
200 MHz (UltraScale+ target). End-to-end time also counts loading the dataset,
order and weights and reading the weights back, at 1 word per cycle.
"""
import os, sys, time, json, subprocess, tempfile
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hwref as R
from fpga_trainer import FpgaTrainer

EPOCHS = 15
CLKS_MHZ = (100, 200)
HW_CFGS = [(1, 1, 1), (8, 4, 5), (16, 8, 10), (64, 16, 10)]   # (P, Q, R) lanes
LR_F = 2.0 ** -R.LR

# ------------------------------------------------------------------ data
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
Xq, Xqt = R.quantize_x(Xtr), R.quantize_x(Xte)
Xf, Xft = Xq / 128.0, Xqt / 128.0                    # same input scale as hardware
p0 = R.init_weights()
order = R.make_order(len(Xq), EPOCHS)
NSTEPS = len(order)
f0 = dict(W1=p0["Wm1"] / 2**20, b1=p0["b1"] / 2**19,
          W2=p0["Wm2"] / 2**20, b2=p0["b2"] / 2**19)

def acc_float(W1, b1, W2, b2):
    h = np.maximum(Xft @ W1.T + b1, 0)
    return float(((h @ W2.T + b2).argmax(1) == yte).mean())

rows = []
print(f"dataset: {len(Xq)} train / {len(Xqt)} test, {EPOCHS} epochs = {NSTEPS} SGD steps\n")

# ------------------------------------------------------------------ 1. integer reference
t = time.perf_counter()
ref = R.train(p0, Xq, ytr, order)
t_ref = time.perf_counter() - t
acc_ref = R.accuracy(ref, Xqt, yte)
print(f"integer reference (spec, NumPy): {acc_ref*100:.2f}% test accuracy")

# ------------------------------------------------------------------ 2. CPU baselines
def numpy_b1():
    W1, b1, W2, b2 = (f0[k].copy() for k in ("W1", "b1", "W2", "b2"))
    t = time.perf_counter()
    for n in order:
        x = Xf[n]; a = W1 @ x + b1; h = np.maximum(a, 0)
        e2 = W2 @ h + b2; e2[ytr[n]] -= 1.0
        e1 = (W2.T @ e2) * (a > 0)
        W2 -= LR_F * np.outer(e2, h); b2 -= LR_F * e2
        W1 -= LR_F * np.outer(e1, x); b1 -= LR_F * e1
    return time.perf_counter() - t, acc_float(W1, b1, W2, b2)

def numpy_mb(B=32):
    W1, b1, W2, b2 = (f0[k].copy() for k in ("W1", "b1", "W2", "b2"))
    Y = np.eye(10)[ytr]
    t = time.perf_counter()
    for s in range(0, NSTEPS, B):
        idx = order[s:s + B]; xb = Xf[idx]
        a = xb @ W1.T + b1; h = np.maximum(a, 0)
        e2 = h @ W2.T + b2 - Y[idx]
        e1 = (e2 @ W2) * (a > 0)
        W2 -= LR_F * e2.T @ h; b2 -= LR_F * e2.sum(0)
        W1 -= LR_F * e1.T @ xb; b1 -= LR_F * e1.sum(0)
    return time.perf_counter() - t, acc_float(W1, b1, W2, b2)

def c_o3(reps=20):
    d = tempfile.mkdtemp(prefix="cpu_sgd_")
    exe = os.path.join(d, "cpu_sgd")
    subprocess.run(["gcc", "-O3", "-march=native", "-o", exe,
                    os.path.join(HERE, "cpu_sgd.c")], check=True)
    Xf.astype(np.float32).tofile(os.path.join(d, "x.f32"))
    ytr.astype(np.int32).tofile(os.path.join(d, "y.i32"))
    order.astype(np.int32).tofile(os.path.join(d, "order.i32"))
    for k, fn in (("W1", "w1"), ("b1", "b1"), ("W2", "w2"), ("b2", "b2")):
        f0[k].astype(np.float32).tofile(os.path.join(d, fn + ".f32"))
    out = subprocess.run([exe, d, str(len(Xq)), str(NSTEPS), str(reps)],
                         check=True, capture_output=True, text=True).stdout
    rd = lambda fn, sh: np.fromfile(os.path.join(d, fn), np.float32).reshape(sh)
    acc = acc_float(rd("w1_c.f32", (16, 64)), rd("b1_c.f32", (16,)),
                    rd("w2_c.f32", (10, 16)), rd("b2_c.f32", (10,)))
    return float(out.strip()), acc

cpu = {}
for name, fn, reps in (("NumPy, batch 1 (Python loop)", numpy_b1, 3),
                       ("NumPy, mini-batch 32 (vectorised)", numpy_mb, 5),
                       ("C -O3 -march=native, batch 1", c_o3, 1)):
    best, acc = 1e30, None
    for _ in range(reps):
        dt, acc = fn()
        best = min(best, dt)
    cpu[name] = dict(time=best, acc=acc)
    print(f"{name:36s}: {best*1e3:9.2f} ms  ({best/NSTEPS*1e9:8.1f} ns/sample)  acc {acc*100:.2f}%")

# ------------------------------------------------------------------ 3. hardware
hw = {}
load_words = 2 * len(Xq) + NSTEPS + (1024 + 16 + 160 + 10)   # samples+labels+order+weights
read_words = 1024 + 16 + 160 + 10
for P, Qx, Rx in HW_CFGS:
    tr = FpgaTrainer(lanes=P, l2_lanes=Qx, bp_lanes=Rx)
    t = time.perf_counter()
    out, cyc = tr.train(p0, Xq, ytr, order)
    sim_s = time.perf_counter() - t
    tr.close()
    exact = all(np.array_equal(out[k], ref[k]) for k in ref)
    acc = R.accuracy(out, Xqt, yte)
    key = f"P={P},Q={Qx},R={Rx}"
    hw[key] = dict(cycles=cyc, cyc_per_step=cyc / NSTEPS, exact=exact, acc=acc,
                   e2e_cycles=cyc + load_words + read_words, sim_wall=sim_s)
    print(f"HW {key:16s}: {cyc:9d} cycles ({cyc/NSTEPS:5.2f}/step)  bit-exact={exact}  "
          f"acc {acc*100:.2f}%  [sim wall {sim_s:.0f}s]")

# ------------------------------------------------------------------ 4. table
lines = []
lines.append(f"Training: {EPOCHS} epochs x {len(Xq)} samples = {NSTEPS} per-sample SGD steps\n")
lines.append("| Platform | Time (ms) | ns / sample | Test acc | Bit-exact |")
lines.append("|---|---|---|---|---|")
for k, v in cpu.items():
    lines.append(f"| CPU: {k} | {v['time']*1e3:.2f} | {v['time']/NSTEPS*1e9:.0f} | "
                 f"{v['acc']*100:.2f}% | n/a (float) |")
for k, v in hw.items():
    for f in CLKS_MHZ:
        t = v["cycles"] / (f * 1e6)
        lines.append(f"| FPGA engine {k} @ {f} MHz (est.) | {t*1e3:.3f} | "
                     f"{t/NSTEPS*1e9:.0f} | {v['acc']*100:.2f}% | {'yes' if v['exact'] else 'NO'} |")

lines.append("\n**Speedup of the engine (training loop only / end-to-end incl. data load + weight readback)**\n")
hdr = "| Engine config | " + " | ".join(f"vs {k} @{f} MHz" for k in cpu for f in CLKS_MHZ) + " |"
lines.append(hdr)
lines.append("|---|" + "---|" * (len(cpu) * len(CLKS_MHZ)))
for k, v in hw.items():
    cells = []
    for name, c in cpu.items():
        for f in CLKS_MHZ:
            t_eng = v["cycles"] / (f * 1e6)
            t_e2e = v["e2e_cycles"] / (f * 1e6)
            cells.append(f"{c['time']/t_eng:.1f}x / {c['time']/t_e2e:.1f}x")
    lines.append(f"| {k} | " + " | ".join(cells) + " |")

lines.append(f"\nCPU: {subprocess.run(['sh','-c','lscpu | grep \"Model name\" | cut -d: -f2'], capture_output=True, text=True).stdout.strip()}"
             f" (single thread). Integer reference accuracy: {acc_ref*100:.2f}%.")
report = "\n".join(lines)
print("\n" + report)
with open(os.path.join(HERE, "results.md"), "w") as f:
    f.write(report + "\n")
with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(dict(cpu=cpu, hw=hw, nsteps=NSTEPS, epochs=EPOCHS,
                   acc_ref=acc_ref, clks_mhz=CLKS_MHZ), f, indent=1)
