#!/usr/bin/env python3
"""recover_perm.py — given matching multisets, recover the permutation
mapping NPU output index -> REF output index.

Strategy: bin both vectors by quantized value (output_scale * (k - zp));
within each bin, NPU has some classes and REF has the same classes
permuted. Either the permutation is identity-on-each-bin (i.e. NPU is
correct, modulo small per-class quant noise) or it has a global pattern
across bins.
"""
import json, struct, pathlib
from collections import defaultdict, Counter

# Load both
ref_json = json.load(open("/home/radxa/ufs/home/radxa/MLIR-TIM-VX/output_log.json"))
ref = [0.0] * 1000
for name, (cls_str, val_str) in ref_json.items():
    ref[int(cls_str)] = float(val_str)
npu = list(struct.unpack("<1000f", pathlib.Path("/tmp/resnet_npu_out.bin").read_bytes()))

# Multiset comparison
print("=== multiset comparison (sorted desc) ===")
ref_sorted = sorted(ref, reverse=True)
npu_sorted = sorted(npu, reverse=True)
mat = sum(1 for a, b in zip(ref_sorted, npu_sorted) if abs(a - b) < 1e-3)
print(f"sorted-value match within 1e-3: {mat}/1000")
mat2 = sum(1 for a, b in zip(ref_sorted, npu_sorted) if abs(a - b) < 0.16)
print(f"sorted-value match within 0.16 (1 quant level): {mat2}/1000")

# Bin by quantized value (rounded to step 0.1416)
def bin_by_value(v):
    return [(i, round(x / 0.1416) * 0.1416) for i, x in enumerate(v)]

ref_by_val = defaultdict(list)
npu_by_val = defaultdict(list)
for i, q in bin_by_value(ref):
    ref_by_val[q].append(i)
for i, q in bin_by_value(npu):
    npu_by_val[q].append(i)

# How many distinct quantized values in each
print(f"\n#distinct quant values: REF={len(ref_by_val)}, NPU={len(npu_by_val)}")

# Bin-size mismatch: classes that have value V in REF but not in NPU
print("\n=== bin counts (top-10 by value) ===")
ref_keys = sorted(ref_by_val.keys(), reverse=True)
print(f"{'value':>9} {'REF#':>5} {'NPU#':>5}")
for v in ref_keys[:15]:
    print(f"{v:>9.4f} {len(ref_by_val[v]):>5d} {len(npu_by_val.get(v, [])):>5d}")

# Try to recover permutation: for each NPU class i, find a REF class j
# such that ref[j] == npu[i] (within 1 quant level). If unique, map i->j.
print("\n=== greedy permutation recovery (NPU[i] ~ REF[π(i)]) ===")
# Round to step 0.1416 for matching
def q(x): return round(x / 0.1416) * 0.1416

ref_pool = defaultdict(list)
for j in range(1000):
    ref_pool[q(ref[j])].append(j)

mapping = {}
unmatched_npu = []
for i in range(1000):
    bin_ = ref_pool[q(npu[i])]
    if not bin_:
        unmatched_npu.append(i)
        continue
    j = bin_.pop(0)  # take first available
    mapping[i] = j

print(f"  matched: {len(mapping)}/1000  unmatched: {len(unmatched_npu)}")
identities = sum(1 for i, j in mapping.items() if i == j)
print(f"  identity (i==j): {identities}/{len(mapping)}")
shifts = Counter(j - i for i, j in mapping.items())
print(f"  top shift offsets (j - i):")
for d, c in shifts.most_common(15):
    print(f"    delta={d:>+5d}: {c} pairs")

# Specific classes the user cares about
print("\n=== specific cells ===")
for c in [51, 284, 285, 223, 281, 282, 283, 1, 5, 0, 457, 818, 111]:
    name = next((n for n, kv in ref_json.items() if int(kv[0]) == c), "?")
    if c in mapping:
        j = mapping[c]
        jname = next((n for n, kv in ref_json.items() if int(kv[0]) == j), "?")
        print(f"  NPU[{c:>3}]={npu[c]:>7.4f} -> REF[{j:>3}]={ref[j]:>7.4f}"
              f"   src={name}  -->  match={jname}")
    else:
        print(f"  NPU[{c:>3}]={npu[c]:>7.4f}  (no match)")

# Bit-reverse / structural permutation test
print("\n=== structural permutation tests ===")
# Test: is mapping consistent with bit-reversal of 10-bit indices?
def bitrev10(n):
    r = 0
    for _ in range(10):
        r = (r << 1) | (n & 1); n >>= 1
    return r
br_match = sum(1 for i, j in mapping.items() if bitrev10(i) == j)
print(f"  bit-reverse10 matches: {br_match}/{len(mapping)}")

# Test: stride-N permutation (i = a*S + b -> j = b*S' + a)
for S in [4, 8, 16, 32, 64]:
    block = 1000 // S
    cnt = sum(1 for i, j in mapping.items()
              if (i % S) * (1000 // S) + (i // S) == j)
    print(f"  stride-{S} (i=aS+b -> j=b*K+a, K={1000//S}): {cnt} matches")
