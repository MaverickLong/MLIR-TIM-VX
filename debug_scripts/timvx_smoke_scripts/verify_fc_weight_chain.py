#!/usr/bin/env python3
"""verify_fc_weight_reorder.py — bisect the FC weight transformation
chain by comparing bytes at three points:

  (A) source MLIR .tosa.const weight (NHWC weight, shape [Oc=1000,1,1,Ic=512])
  (B) .timvx.mlir post-canonicalize timvx.const weight (shape [K=512, M=1000])
  (C) externalized .bin file (raw bytes in TIM-VX inner-first order)

Transformations expected:
  A -> (transpose [2,1,3,0] folded by canonicalize) -> shape [W=1,H=1,Ic=512,Oc=1000]
       which `wAttr.reshape` then drops to [K=512, M=1000] without moving bytes
  B -> (reorderMlirToTvx for [512,1000]) -> bytes in TIM-VX inner-first

Each step is implemented in Python and we check byte-for-byte agreement.
A mismatch localizes the bug.
"""
import re, struct, pathlib
from pathlib import Path

ROOT = Path("/home/radxa/ufs/home/radxa/MLIR-TIM-VX")
SRC_MLIR = ROOT / "resnet18_weights_v1.mlir"
TVX_MLIR = ROOT / "example/lower_out/resnet18_weights_v1/resnet18_weights_v1.timvx.mlir"
BIN = ROOT / "example/lower_out/resnet18_weights_v1/_timvx_const_83.bin"

K, M = 512, 1000

def load_dense_hex(text, pattern_re):
    """Find a `dense<"0xHEX..."> : tensor<...>` matching pattern_re; return
    the parsed bytes. Returns first match."""
    m = re.search(pattern_re, text, re.S)
    if not m: return None
    s = m.group(1).strip().strip('"')
    if s.startswith("0x") or s.startswith("0X"): s = s[2:]
    return bytes.fromhex(s)

# --- (B): post-canonicalize FC weight in .timvx.mlir, shape [512, 1000] ui8
tvx_text = TVX_MLIR.read_text()
b_bytes = load_dense_hex(tvx_text,
    r'"timvx\.const"\(\)\s*<\{[^>]*?values\s*=\s*dense<("0x[0-9A-Fa-f]+")[^>]*?>\s*:\s*tensor<512x1000xui8>')
print(f"(B) post-canonicalize ui8 weight in .timvx.mlir: {len(b_bytes)} bytes (expected {K*M})")
assert len(b_bytes) == K * M

# --- (C): externalized .bin
c_bytes = BIN.read_bytes()
print(f"(C) externalized .bin: {len(c_bytes)} bytes")
assert len(c_bytes) == K * M

# (B) is in MLIR row-major for shape [K=512, M=1000]:
#     byte at offset (k*M + m) is weight for class m, input feature k.
# (C) is in TIM-VX inner-first for shape {K=512, M=1000}:
#     byte at offset (k + m*K) is weight for class m, input feature k.
# Reorder B -> C using the same logic as reorderMlirToTvx.
expected_C = bytearray(K * M)
for k in range(K):
    for m in range(M):
        expected_C[k + m * K] = b_bytes[k * M + m]

if bytes(expected_C) == c_bytes:
    print("(B -> C) reorderMlirToTvx step: byte-EXACT match ✓ "
          "(EmitC weight reorder is correct)")
else:
    diff = sum(1 for x, y in zip(expected_C, c_bytes) if x != y)
    print(f"(B -> C) reorderMlirToTvx step: {diff}/{K*M} byte differences ✗")

# --- (A): source TOSA weight in resnet18_weights_v1.mlir.
# It's the const that becomes the conv1x1 weight; shape is [Oc=1000, 1, 1, Ic=512]
# in the TFLite-export format. After tosa-quant-anchor it's
# tensor<1000x1x1x512x!quant.uniform<i8:f32, S>>; pre-anchor it's
# tensor<1000x1x1x512xi8>.
src_text = SRC_MLIR.read_text()
a_bytes = load_dense_hex(src_text,
    r'"tosa\.const"\(\)\s*<\{values\s*=\s*dense<("0x[0-9A-Fa-f]+")[^>]*?>\s*:\s*tensor<1000x1x1x512xi8>')
if a_bytes is None:
    print("(A) source TOSA weight not found at shape <1000x1x1x512xi8>")
else:
    print(f"(A) source i8 weight in resnet18_weights_v1.mlir: {len(a_bytes)} bytes (expected {K*M})")
    assert len(a_bytes) == K * M
    # Storage is signed i8. Values are weight[oc, 0, 0, ic] = byte at
    # offset (oc*512 + ic) in MLIR row-major.
    # After --timvx-promote-i8-to-u8: bytes XOR 0x80 (signed -> unsigned).
    # After --canonicalize TransposeOfConstFold for kPermNHWCToWHCN [2,1,3,0]:
    #     out shape [1, 1, 512, 1000] = [W, H, I, O]
    #     byte at MLIR offset (w, h, ic, oc) = w*1*512*1000 + h*512*1000 + ic*1000 + oc
    #                                        = ic*1000 + oc  (W=H=0)
    #     and that = source byte at (oc, 0, 0, ic) = oc*512 + ic
    # Reshape to [512, 1000] preserves bytes.
    # So after the full source -> B chain:
    #   byte_B[k*M + m] (in row-major shape [K=512, M=1000])
    #     == byte_A[m*K + k]  (with bytes XOR 0x80 for u8 promotion)
    expected_B = bytearray(K * M)
    for k in range(K):       # ic in source
        for m in range(M):   # oc in source
            expected_B[k * M + m] = a_bytes[m * K + k] ^ 0x80

    if bytes(expected_B) == b_bytes:
        print("(A -> B) transpose-fold + reshape + promote: byte-EXACT match ✓ "
              "(transpose canonicalizer is correct)")
    else:
        diff = sum(1 for x, y in zip(expected_B, b_bytes) if x != y)
        print(f"(A -> B) transpose-fold + reshape + promote: "
              f"{diff}/{K*M} byte differences ✗")
        # Show first 10 mismatches with context
        n_shown = 0
        for i in range(K * M):
            if expected_B[i] != b_bytes[i]:
                k, m = divmod(i, M)
                print(f"    offset {i} (k={k}, m={m}): "
                      f"expected 0x{expected_B[i]:02x}, "
                      f"got 0x{b_bytes[i]:02x}  "
                      f"(source byte at oc={m}, ic={k}: 0x{a_bytes[m*K+k]:02x})")
                n_shown += 1
                if n_shown >= 10: break

print("\n--- spot check: weight rows for class 51 (NPU argmax) and 284 (REF argmax) ---")
def row_in_C(m):
    return c_bytes[m*K : (m+1)*K]
def row_from_A(m):
    if a_bytes is None: return None
    return bytes((a_bytes[m*K + k] ^ 0x80) for k in range(K))

for m in [51, 284, 285]:
    cR = row_in_C(m)
    aR = row_from_A(m)
    print(f"  class {m:>4}: .bin row[0:8]   = {cR[:8].hex()}")
    if aR is not None:
        print(f"  class {m:>4}: source row[0:8] = {aR[:8].hex()}  "
              f"({'MATCH' if cR == aR else 'DIFFER'})")
