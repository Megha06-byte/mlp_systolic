#!/usr/bin/env python3
"""
test_bitexact.py : quick regression for the on-chip training engine.

Trains for a few hundred SGD steps on the RTL (via FpgaTrainer) and on the
Python integer spec (hwref.py) from the same initial weights and sample order,
then checks every trained weight and bias is identical.

usage: python3 train/test_bitexact.py [steps]
"""
import os, sys
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hwref as R
from fpga_trainer import FpgaTrainer

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
CFGS = [(1, 1, 1), (8, 4, 5), (16, 8, 10)]          # (P, Q, R) lanes

X, y = load_digits(return_X_y=True)
Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
Xq, p0 = R.quantize_x(Xtr), R.init_weights()
order = R.make_order(len(Xq), 1)[:STEPS]
ref = R.train(p0, Xq, ytr, order)

ok = True
for P, Q, Rr in CFGS:
    hw = FpgaTrainer(lanes=P, l2_lanes=Q, bp_lanes=Rr)
    out, cyc = hw.train(p0, Xq, ytr, order)
    hw.close()
    bad = {k: int((out[k] != ref[k]).sum()) for k in ref}
    good = not any(bad.values())
    ok &= good
    print(f"P={P:2d} Q={Q:2d} R={Rr:2d}: {cyc:7d} cycles ({cyc/STEPS:5.1f}/step)  "
          f"{'PASS' if good else 'FAIL ' + str(bad)}")
sys.exit(0 if ok else 1)
