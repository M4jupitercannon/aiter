"""Stage1 async-copy vs sync-copy vs CK baseline benchmark.

Reads cases from dsv3_fp4_tuned_moe.csv (first 16 non-fallback rows),
benchmarks FlyDSL stage1 with sync and async copy, and compares to CK times.

Usage:
    python bench_stage1_async.py
    python bench_stage1_async.py --num-iters 50
"""

import argparse
import csv
import os
import sys

import torch
import aiter
from aiter import dtypes, QuantType
from aiter.test_common import run_perftest, checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

torch.set_default_device("cuda")

Q_TYPE = QuantType.per_1x32
Q_DTYPE_A = dtypes.fp4x2
TORCH_QUANT = aiter.get_torch_quant(Q_TYPE)

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "aiter", "configs", "model_configs", "dsv3_fp4_tuned_moe.csv",
)


def load_cases(csv_path):
    cases = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("_tag", "").strip()
            if tag == "flydsl_fallback":
                continue
            cases.append(dict(
                token=int(row["token"]),
                model_dim=int(row["model_dim"]),
                inter_dim=int(row["inter_dim"]),
                expert=int(row["expert"]),
                topk=int(row["topk"]),
                block_m=int(row["block_m"]),
                ck_us1=float(row["us1"]),
            ))
    return cases


def setup_data(token, model_dim, inter_dim, E, topk, block_m,
               dtype=torch.bfloat16):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = TORCH_QUANT(w1, quant_dtype=Q_DTYPE_A)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    a1_qt, a1_scale = TORCH_QUANT(inp, quant_dtype=Q_DTYPE_A)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = \
        moe_sorting(topk_ids, topk_weights, E, model_dim, dtype, block_m)

    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w1_scale_shuf = e8m0_shuffle(w1_scale)

    a1_scale_sort = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token, block_size=block_m,
    )

    return dict(
        a1_qt=a1_qt, w1_qt_shuf=w1_qt_shuf,
        w1_scale_shuf=w1_scale_shuf, a1_scale_sort=a1_scale_sort,
        sorted_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
    )


def call_stage1(d, topk, block_m, use_async):
    return flydsl_moe_stage1(
        a=d["a1_qt"], w1=d["w1_qt_shuf"],
        sorted_token_ids=d["sorted_ids"],
        sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"],
        topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w1_scale=d["w1_scale_shuf"], a1_scale=d["a1_scale_sort"],
        use_async_copy=use_async,
    )


def fn_stage1(a1_qt, w1_qt_shuf, sorted_ids, sorted_expert_ids,
              num_valid_ids, w1_scale_shuf, a1_scale_sort,
              topk, block_m, use_async):
    return flydsl_moe_stage1(
        a=a1_qt, w1=w1_qt_shuf,
        sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w1_scale=w1_scale_shuf, a1_scale=a1_scale_sort,
        use_async_copy=use_async,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--atol", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=0.005)
    args = parser.parse_args()

    cases = load_cases(args.csv)
    ni, nw = args.num_iters, args.num_warmup
    results = []

    print(f"\nBenchmarking {len(cases)} cases  (iters={ni}, warmup={nw})")
    print("=" * 100)

    for c in cases:
        token = c["token"]

        # if token != 1024:
        #     continue
        # if c["block_m"] == 32:
        #     continue

        # block_m = c["block_m"] if c["block_m"] <= 64 else 64
        block_m = c["block_m"]
        # block_m = 128
        ck_us = c["ck_us1"]

        torch.cuda.empty_cache()
        print(f"\n  token={token:>5d}  block_m={block_m:>3d}  (CK={ck_us:.2f} us)")

        d = setup_data(token, c["model_dim"], c["inter_dim"],
                       c["expert"], c["topk"], block_m)

        common = (
            d["a1_qt"], d["w1_qt_shuf"],
            d["sorted_ids"], d["sorted_expert_ids"], d["num_valid_ids"],
            d["w1_scale_shuf"], d["a1_scale_sort"],
            c["topk"], block_m,
        )

        ref_out = call_stage1(d, c["topk"], block_m, use_async=False)
        async_out = call_stage1(d, c["topk"], block_m, use_async=True)
        torch.cuda.synchronize()

        err_ratio = checkAllclose(
            ref_out, async_out,
            rtol=args.rtol, atol=args.atol,
            msg=f"    [t={token},bm={block_m}] ",
        )
        prec_ok = "PASS" if err_ratio == 0 else ("WARN" if err_ratio <= 0.05 else "FAIL")

        # print("    sync ...", end="", flush=True)
        # _, us_sync = run_perftest(
        #     fn_stage1, *common, False,
        #     num_iters=ni, num_warmup=nw, num_rotate_args=1,
        # )
        # print(f"  {us_sync:.2f} us")

        print("    async...", end="", flush=True)
        _, us_async = run_perftest(
            fn_stage1, *common, True,
            num_iters=ni, num_warmup=nw, num_rotate_args=1,
        )
        print(f"  {us_async:.2f} us")

        results.append(dict(
            token=token, block_m=block_m,
            # ck_us=ck_us, sync_us=us_sync, async_us=us_async,
            ck_us=ck_us, sync_us=0, async_us=us_async,
            err_ratio=err_ratio, prec_ok=prec_ok,
        ))

    print(f"\n{'=' * 100}")
    print("SUMMARY: Stage1 Performance Comparison")
    print(f"{'=' * 100}")
    hdr = (f"  {'token':>6s}  {'bm':>3s}  {'prec':>4s}  "
           f"{'CK_us':>9s}  {'sync_us':>9s}  {'async_us':>9s}  "
           f"{'sync/CK':>8s}  {'async/CK':>9s}  {'async/sync':>11s}")
    print(hdr)
    print(f"  {'-'*6}  {'-'*3}  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*11}")
    for r in results:
        sync_vs_ck = r["sync_us"] / r["ck_us"] if r["ck_us"] > 0 else 0
        async_vs_ck = r["async_us"] / r["ck_us"] if r["ck_us"] > 0 else 0
        async_vs_sync = r["async_us"] / r["sync_us"] if r["sync_us"] > 0 else 0
        print(f"  {r['token']:>6d}  {r['block_m']:>3d}  {r['prec_ok']:>4s}  "
              f"{r['ck_us']:>9.2f}  {r['sync_us']:>9.2f}  {r['async_us']:>9.2f}  "
              f"{sync_vs_ck:>8.3f}  {async_vs_ck:>9.3f}  {async_vs_sync:>11.3f}")


if __name__ == "__main__":
    main()
