#!/usr/bin/env bash
# Train/prune/quantise the model, regenerate hardcoded weights, run the testbench.
# Needs: python3 + numpy + scikit-learn, Icarus Verilog >= 11.
set -euo pipefail
cd "$(dirname "$0")"
python3 model/gen.py
cd sim
iverilog -g2005 -I../rtl -I. -o tb.vvp ../rtl/*.v ../tb/tb_mlp_top.v
vvp -n tb.vvp | grep -v '\$finish'
