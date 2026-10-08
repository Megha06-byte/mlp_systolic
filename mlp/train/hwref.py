"""
hwref.py : bit-exact integer specification of the on-chip trainer.

Every operation here is mirrored one-to-one by rtl/train/mlp_train_engine.v.
If the RTL and this file disagree by a single LSB, the testbench fails.

Network : 64 -> 16 (ReLU, saturating at 255) -> 10, MSE loss, per-sample SGD
Number formats (Qm.n = n fractional bits):
  x        uint8  0..127   Q.7   (pixel/16 scaled)
  h        uint8  0..255   Q.7   hidden activation
  Wm1,Wm2  int32           Q.20  master weights (updated)
  wf1,wf2  int16           Q.12  forward weights = sat16(Wm >> 8)
  b1,b2    int32           Q.19  biases (same scale as accumulators)
  acc      int32           Q.19
  e2,e1    int16           Q.12  output / hidden error
SGD learning rate = 2^-LR, rounding = round-half-up via (v + 2^(s-1)) >> s.
"""
import numpy as np

N0, N1, N2 = 64, 16, 10
WSH = 8                      # master Q.20 -> forward Q.12
LR = 7                       # learning rate 2^-7

def rsr(v, s):
    """round-half-up arithmetic right shift (matches RTL)"""
    if s <= 0:
        return v << (-s)
    return (v + (1 << (s - 1))) >> s

def sat(v, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(v, lo, hi)

def fwd_w(Wm):
    return sat(Wm >> WSH, 16)

def quantize_x(raw_pixels):                     # digits pixels 0..16
    return np.clip(np.round(raw_pixels / 16.0 * 127), 0, 127).astype(np.int64)

def init_weights(seed=0):
    rng = np.random.default_rng(seed)
    # variance 1/fan_in (He-style init is too large for an MSE head and kills
    # most hidden ReLUs); small positive hidden bias keeps units alive
    W1 = rng.normal(0, np.sqrt(1 / N0), (N1, N0))
    W2 = rng.normal(0, np.sqrt(1 / N1), (N2, N1))
    Wm1 = np.round(W1 * (1 << 20)).astype(np.int64)
    Wm2 = np.round(W2 * (1 << 20)).astype(np.int64)
    b1 = np.full(N1, round(0.1 * (1 << 19)), np.int64)
    return dict(Wm1=Wm1, b1=b1, Wm2=Wm2, b2=np.zeros(N2, np.int64))

def forward(p, x):
    wf1, wf2 = fwd_w(p["Wm1"]), fwd_w(p["Wm2"])
    acc1 = p["b1"] + wf1 @ x
    h = np.clip(acc1 >> 12, 0, 255)
    mask = (acc1 > 0) & ((acc1 >> 12) < 255)    # ReLU' (0 when clipped)
    acc2 = p["b2"] + wf2 @ h
    return acc1, h, mask, acc2, wf2

def sgd_step(p, x, y):
    """one sample: forward, MSE error, backprop, update (in place)"""
    _, h, mask, acc2, wf2 = forward(p, x)
    t = np.zeros(N2, np.int64); t[y] = 1 << 19          # one-hot target, Q.19
    e2 = sat((acc2 - t) >> 7, 16)                       # Q.12
    s1 = wf2.T @ e2                                     # Q.24, uses pre-update W2
    e1 = np.where(mask, sat(s1 >> 12, 16), 0)           # Q.12
    p["Wm2"] -= rsr(np.outer(e2, h), LR - 1)            # Q.19 grad -> Q.20 * 2^-LR
    p["Wm1"] -= rsr(np.outer(e1, x), LR - 1)
    p["b2"]  -= rsr(e2, LR - 7)                         # Q.12 -> Q.19 * 2^-LR
    p["b1"]  -= rsr(e1, LR - 7)

def train(p, Xq, y, order):
    p = {k: v.copy() for k, v in p.items()}
    for n in order:
        sgd_step(p, Xq[n], int(y[n]))
    for k in ("Wm1", "Wm2", "b1", "b2"):                # RTL registers are 32 bit
        assert np.abs(p[k]).max() < (1 << 31), f"{k} overflow"
    return p

def accuracy(p, Xq, y):
    pred = [int(np.argmax(forward(p, x)[3])) for x in Xq]
    return float(np.mean(np.array(pred) == y))

def make_order(n_samples, epochs, seed=1):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(n_samples) for _ in range(epochs)])
