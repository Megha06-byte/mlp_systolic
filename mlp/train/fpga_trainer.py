"""
fpga_trainer.py : Python plug-in for the on-chip SGD engine.

    from fpga_trainer import FpgaTrainer
    hw = FpgaTrainer(lanes=8)
    params, cycles = hw.train(init_params, Xq, y, order)

The training loop in Python stays the same; only the inner SGD loop moves
to the hardware. Backends:
  "sim"   cycle-accurate RTL simulation (Verilator if installed, else Icarus
          Verilog), used here since there's no board attached. Wall-clock
          time of the simulation is NOT hardware time; hardware time is
          cycles / f_clk.
  board   (not implemented) same interface over PCIe/AXI (e.g. PYNQ
          MMIO + DMA): write the same buffers, start, poll done, read back.
"""
import os, subprocess, tempfile, shutil
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RTL = os.path.join(ROOT, "rtl", "train", "mlp_train_engine.v")
TB = os.path.join(ROOT, "tb", "tb_train.v")


def _h32(v):
    return f"{int(v) & 0xffffffff:08x}"


def _s32(s):
    v = int(s, 16)
    return v - (1 << 32) if v & (1 << 31) else v


class FpgaTrainer:
    def __init__(self, lanes=8, l2_lanes=4, bp_lanes=5, lr=7, backend="sim",
                 simulator=None, workdir=None):
        assert 64 % lanes == 0 and 16 % l2_lanes == 0 and 10 % bp_lanes == 0
        self.lanes, self.l2_lanes, self.bp_lanes = lanes, l2_lanes, bp_lanes
        self.lr, self.backend = lr, backend
        self.simulator = simulator or ("verilator" if shutil.which("verilator") else "iverilog")
        self.workdir = workdir or tempfile.mkdtemp(prefix="fpga_train_")
        self._cmd = None

    # ---------------------------------------------------------------- build
    def _build(self):
        if self._cmd is not None:
            return self._cmd
        if self.simulator == "verilator":
            obj = os.path.join(self.workdir, f"obj_p{self.lanes}_q{self.l2_lanes}_r{self.bp_lanes}")
            subprocess.run(["verilator", "--binary", "--timing", "-O3", "-Wno-fatal",
                            "-Wno-lint", "-Wno-style", "--unroll-count", "4096",
                            "--unroll-stmts", "100000", "--top-module", "tb_train",
                            f"-GP={self.lanes}", f"-GQ={self.l2_lanes}", f"-GR={self.bp_lanes}",
                            f"-GLR={self.lr}", "-Mdir", obj,
                            "-j", "8", TB, RTL],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._cmd = [os.path.join(obj, "Vtb_train")]
        else:
            vvp = os.path.join(self.workdir, f"train_p{self.lanes}.vvp")
            subprocess.run(["iverilog", "-g2005", "-o", vvp,
                            f"-Ptb_train.P={self.lanes}", f"-Ptb_train.Q={self.l2_lanes}",
                            f"-Ptb_train.R={self.bp_lanes}", f"-Ptb_train.LR={self.lr}",
                            RTL, TB], check=True)
            self._cmd = ["vvp", "-n", vvp]
        return self._cmd

    # ---------------------------------------------------------------- train
    def train(self, params, Xq, y, order):
        """Run len(order) SGD steps on the hardware. Returns (params, cycles)."""
        if self.backend != "sim":
            raise NotImplementedError("board backend: same buffers over MMIO/DMA")
        assert len(Xq) <= 2048 and len(order) <= 32768
        cmd, d = self._build(), self.workdir

        with open(os.path.join(d, "samples.hex"), "w") as f:
            for x in Xq:                         # byte k -> bits [8k+7:8k]
                f.write("".join(f"{int(v):02x}" for v in x[::-1]) + "\n")
        np.savetxt(os.path.join(d, "labels.hex"), np.asarray(y), fmt="%08x")
        np.savetxt(os.path.join(d, "order.hex"), np.asarray(order), fmt="%08x")
        for name in ("Wm1", "b1", "Wm2", "b2"):
            fn = {"Wm1": "w1_in.hex", "b1": "b1_in.hex",
                  "Wm2": "w2_in.hex", "b2": "b2_in.hex"}[name]
            with open(os.path.join(d, fn), "w") as f:
                f.write("\n".join(_h32(v) for v in params[name].ravel()) + "\n")

        subprocess.run(cmd + [f"+NSAMP={len(Xq)}", f"+NSTEPS={len(order)}"],
                       cwd=d, check=True, stdout=subprocess.DEVNULL)

        def rd(fn, shape):
            with open(os.path.join(d, fn)) as f:
                return np.array([_s32(l) for l in f.read().split()], np.int64).reshape(shape)

        out = dict(Wm1=rd("w1_out.hex", (16, 64)), b1=rd("b1_out.hex", (16,)),
                   Wm2=rd("w2_out.hex", (10, 16)), b2=rd("b2_out.hex", (10,)))
        cycles = int(open(os.path.join(d, "cycles.txt")).read())
        return out, cycles

    def close(self):
        shutil.rmtree(self.workdir, ignore_errors=True)
