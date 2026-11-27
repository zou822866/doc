"""Black-box CoCoA baseline computed offline from existing RSC results.

CoCoA (Vashurin et al., 2025, "Uncertainty Quantification for LLMs through
Minimum Bayes Risk: Bridging Confidence and Consistency") scores a fixed
output y* as

    U_CoCoA(y*|x) = u(y*|x) * (1/N) * sum_j (1 - s(y*, y_j)),

where u is a confidence-based uncertainty, y_j are sampled answers and s is a
similarity in [0, 1]. Original CoCoA uses token probabilities for u, which
GPT-5.4 does not expose, so this script uses the black-box adaptation:

    * y*      = the fixed response y shared by all methods
    * y_j     = the N independent answers already drawn by RSC
    * s       = exact match (letter for GPQA/LongBench, normalized numeric
                value for TableBench), so the consistency term = 1 - z_proxy
    * u       = 1 - c (verbalized), with c from Direct-VC or from SC-Avg

No API calls are made. Every quantity is read from the per-case result JSON.

Variants reported
    CoCoA-VC        u = 1 - c_DirectVC,  confidence = 1 - U  (label-free)
    CoCoA-Avg       u = 1 - c_SCAvg,     confidence = 1 - U  (label-free)
    CoCoA-VC-log    u = -log c_DirectVC  (unbounded score; ranking only)
    CoCoA-Avg-log   u = -log c_SCAvg     (unbounded score; ranking only)
    *+iso           the label-free score mapped to a probability by isotonic
                    regression fitted with K-fold cross-fitting. Uses
                    correctness labels, so it is a reference, not a
                    label-free competitor.
Reference rows (recomputed with the same metric code)
    Direct-VC, SC-Avg, z_proxy-only, RSC

Note: 1 - U = c + z - c*z ("noisy-OR") is always >= max(c, z), so the
label-free CoCoA confidence is pushed upward; CoCoA is designed for ranking.

Usage
    python cocoa_baseline.py \
        --dataset GPQA       results/rsc_gpqa.json       results/gpqa_baseline1.json \
        --dataset LongBench  results/rsc_longbench.json  results/longbench_baseline1.json \
        --dataset TableBench results/rsc_tablebench.json results/tablebench_baseline1.json \
        --out_dir results/cocoa --bootstrap 10000

The baseline1 (Direct-VC) file is optional when the RSC rows already carry
the Direct-VC confidence (see DIRECT_VC_KEYS).
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

EPS = 1e-12

# Where to look for each per-case field, in order of preference.
DIRECT_VC_KEYS: Sequence[Sequence[str]] = (
    ("baseline1", "confidence"),
    ("baseline1_confidence",),
    ("direct_vc_confidence",),
    ("answer_step", "confidence"),
)
SC_AVG_KEYS: Sequence[Sequence[str]] = (
    ("baseline2_source_confidence",),
    ("baseline2", "confidence"),
)
RSC_KEYS: Sequence[Sequence[str]] = (("final_confidence",),)
Z_KEYS: Sequence[Sequence[str]] = (("z_proxy",),)
LABEL_KEYS: Sequence[Sequence[str]] = (("response_correct",),)


# --------------------------------------------------------------------- I/O
def load_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        rows = json.loads(text)
    else:  # JSONL
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [r for r in rows if r.get("status", "ok") != "error"]


def dig(row: Dict[str, Any], keys: Sequence[Sequence[str]]) -> Optional[Any]:
    for path in keys:
        value: Any = row
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def case_key(row: Dict[str, Any]) -> str:
    qid = row.get("qid", row.get("record_id"))
    if qid is not None:
        return str(qid)
    return str(row.get("question", "")).strip()


def clamp01(x: Any) -> float:
    return min(1.0, max(0.0, float(x)))


def z_from_samples(row: Dict[str, Any], n: int) -> Optional[float]:
    """Recompute z_proxy from the first n independent answers."""
    outputs = row.get("consistency_sampled_outputs") or []
    flags = [o.get("matches_baseline2_response") for o in outputs[:n]]
    flags = [bool(f) for f in flags if f is not None]
    if len(flags) < n:
        return None
    return sum(flags) / n


def build_cases(
    rsc_path: Path,
    vc_path: Optional[Path],
    n_consistency: Optional[int],
) -> List[Dict[str, Any]]:
    rsc_rows = load_rows(rsc_path)
    vc_by_key: Dict[str, float] = {}
    if vc_path is not None:
        for r in load_rows(vc_path):
            c = dig(r, DIRECT_VC_KEYS)
            if c is not None:
                vc_by_key[case_key(r)] = clamp01(c)

    cases, missing_vc = [], 0
    for r in rsc_rows:
        key = case_key(r)
        label = dig(r, LABEL_KEYS)
        z = dig(r, Z_KEYS)
        c_avg = dig(r, SC_AVG_KEYS)
        c_rsc = dig(r, RSC_KEYS)
        if label is None or z is None or c_avg is None:
            raise ValueError(f"{rsc_path}: case {key} lacks label/z_proxy/SC-Avg")
        if n_consistency is not None:
            z_sub = z_from_samples(r, n_consistency)
            if z_sub is None:
                raise ValueError(
                    f"{rsc_path}: case {key} has fewer than "
                    f"{n_consistency} consistency samples"
                )
            z = z_sub
        c_vc = vc_by_key.get(key)
        if c_vc is None:
            c_vc = dig(r, DIRECT_VC_KEYS)
        if c_vc is None:
            missing_vc += 1
        cases.append({
            "qid": key,
            "correct": int(bool(label)),
            "z_proxy": clamp01(z),
            "c_direct_vc": None if c_vc is None else clamp01(c_vc),
            "c_sc_avg": clamp01(c_avg),
            "c_rsc": None if c_rsc is None else clamp01(c_rsc),
        })
    if missing_vc:
        print(f"  [warn] {missing_vc}/{len(cases)} cases have no Direct-VC "
              "confidence; CoCoA-VC rows are skipped for this dataset.")
    return cases


# ----------------------------------------------------------------- CoCoA
def cocoa_score(u: float, z: float) -> float:
    """U = u * mean_j(1 - s(y*, y_j)); with exact match this is u * (1 - z)."""
    return u * (1.0 - z)


def add_cocoa(cases: List[Dict[str, Any]]) -> None:
    for c in cases:
        z = c["z_proxy"]
        for tag, conf in (("vc", c["c_direct_vc"]), ("avg", c["c_sc_avg"])):
            if conf is None:
                c[f"U_cocoa_{tag}"] = c[f"conf_cocoa_{tag}"] = None
                c[f"U_cocoa_{tag}_log"] = None
                continue
            u_lin = 1.0 - conf
            U = cocoa_score(u_lin, z)
            c[f"U_cocoa_{tag}"] = U
            c[f"conf_cocoa_{tag}"] = 1.0 - U  # = conf + z - conf*z
            u_log = -math.log(max(conf, EPS))
            c[f"U_cocoa_{tag}_log"] = cocoa_score(u_log, z)


# --------------------------------------------------------------- metrics
def ece(labels: List[int], conf: List[float], n_bins: int = 10) -> float:
    """Equal-width bins, same convention as rejection_sampling.compute_ece."""
    n, total = len(labels), 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(conf)
               if lo <= c < hi or (b == n_bins - 1 and c == hi)]
        if idx:
            acc = sum(labels[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - avg)
    return total


def brier(labels: List[int], conf: List[float]) -> float:
    return sum((c - y) ** 2 for y, c in zip(labels, conf)) / len(labels)


def nll(labels: List[int], conf: List[float]) -> float:
    """Binary NLL, clipped to [1e-12, 1-1e-12] as in compute_log_loss."""
    out = 0.0
    for y, c in zip(labels, conf):
        p = min(1 - EPS, max(EPS, c))
        out -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return out / len(labels)


def auroc(labels: List[int], score: List[float]) -> float:
    """Mann-Whitney AUROC, correct = positive, ties get half credit."""
    order = sorted(range(len(score)), key=lambda i: score[i])
    ranks = [0.0] * len(score)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def aurc(labels: List[int], score: List[float]) -> float:
    """Area under the risk-coverage curve (higher score = kept first).

    Risk at coverage k/n is the error rate among the k most confident cases.
    Within a group of tied scores, errors are counted at their expected
    position under a random order, so the result does not depend on how ties
    happen to be sorted.
    """
    n = len(labels)
    groups: Dict[float, List[int]] = {}
    for s, y in zip(score, labels):
        groups.setdefault(s, []).append(y)
    errors_before, seen, total = 0.0, 0, 0.0
    for s in sorted(groups, reverse=True):
        ys = groups[s]
        g, g_err = len(ys), len(ys) - sum(ys)
        for t in range(1, g + 1):
            total += (errors_before + t * g_err / g) / (seen + t)
        errors_before += g_err
        seen += g
    return total / n


def all_metrics(labels: List[int], conf: Optional[List[float]],
                rank_score: List[float]) -> Dict[str, float]:
    out = {"AUROC": auroc(labels, rank_score), "AURC": aurc(labels, rank_score)}
    if conf is not None:
        out.update({"ECE": ece(labels, conf), "Brier": brier(labels, conf),
                    "NLL": nll(labels, conf)})
    return out


# ------------------------------------------------- isotonic (cross-fitted)
def pav_fit(x: List[float], y: List[int]) -> Tuple[List[float], List[float]]:
    """Pool-adjacent-violators; returns block upper x-edges and fitted values."""
    pairs = sorted(zip(x, y))
    blocks: List[List[float]] = []  # [sum_y, count, max_x]
    for xi, yi in pairs:
        blocks.append([float(yi), 1.0, xi])
        while len(blocks) > 1 and (
            blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]
        ):
            s, c, mx = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
            blocks[-1][2] = mx
    return [b[2] for b in blocks], [b[0] / b[1] for b in blocks]


def pav_predict(edges: List[float], vals: List[float], x: float) -> float:
    for e, v in zip(edges, vals):
        if x <= e:
            return v
    return vals[-1]


def cross_fit_isotonic(score: List[float], labels: List[int],
                       k: int, seed: int) -> List[float]:
    idx = list(range(len(score)))
    random.Random(seed).shuffle(idx)
    folds = [idx[f::k] for f in range(k)]
    out = [0.0] * len(score)
    for f in range(k):
        test = set(folds[f])
        train = [i for i in idx if i not in test]
        edges, vals = pav_fit([score[i] for i in train],
                              [labels[i] for i in train])
        for i in folds[f]:
            out[i] = pav_predict(edges, vals, score[i])
    return out


# --------------------------------------------------------------- methods
def method_table(cases: List[Dict[str, Any]], iso_folds: int,
                 seed: int) -> Dict[str, Dict[str, Any]]:
    """Return {method: {"conf": list|None, "rank": list, "uses_labels": bool}}."""
    labels = [c["correct"] for c in cases]
    has_vc = all(c["c_direct_vc"] is not None for c in cases)
    has_rsc = all(c["c_rsc"] is not None for c in cases)
    col = lambda k: [c[k] for c in cases]  # noqa: E731
    neg = lambda xs: [-x for x in xs]       # noqa: E731

    m: Dict[str, Dict[str, Any]] = {}
    if has_vc:
        m["Direct-VC"] = {"conf": col("c_direct_vc"), "rank": col("c_direct_vc")}
    m["SC-Avg"] = {"conf": col("c_sc_avg"), "rank": col("c_sc_avg")}
    m["z_proxy-only"] = {"conf": col("z_proxy"), "rank": col("z_proxy")}
    if has_rsc:
        m["RSC"] = {"conf": col("c_rsc"), "rank": col("c_rsc")}
    for tag, name in (("vc", "CoCoA-VC"), ("avg", "CoCoA-Avg")):
        if tag == "vc" and not has_vc:
            continue
        m[name] = {"conf": col(f"conf_cocoa_{tag}"),
                   "rank": neg(col(f"U_cocoa_{tag}"))}
        m[f"{name}-log"] = {"conf": None,
                            "rank": neg(col(f"U_cocoa_{tag}_log"))}
        iso = cross_fit_isotonic(neg(col(f"U_cocoa_{tag}")), labels,
                                 iso_folds, seed)
        m[f"{name}+iso"] = {"conf": iso, "rank": neg(col(f"U_cocoa_{tag}")),
                            "uses_labels": True}
    for v in m.values():
        v.setdefault("uses_labels", False)
    return m


def paired_bootstrap(cases: List[Dict[str, Any]], a: Dict[str, Any],
                     b: Dict[str, Any], B: int, seed: int
                     ) -> Dict[str, Tuple[float, float]]:
    """95% CI of metric(a) - metric(b), resampling cases with replacement."""
    rng = random.Random(seed)
    labels = [c["correct"] for c in cases]
    n = len(cases)
    diffs: Dict[str, List[float]] = {}
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        y = [labels[i] for i in idx]
        if 0 < sum(y) < n:
            ma = all_metrics(y, None if a["conf"] is None else [a["conf"][i] for i in idx],
                             [a["rank"][i] for i in idx])
            mb = all_metrics(y, None if b["conf"] is None else [b["conf"][i] for i in idx],
                             [b["rank"][i] for i in idx])
            for k in ma:
                if k in mb:
                    diffs.setdefault(k, []).append(ma[k] - mb[k])
    ci = {}
    for k, d in diffs.items():
        d.sort()
        ci[k] = (d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1])
    return ci


# ------------------------------------------------------------------ main
METRICS = ("ECE", "Brier", "NLL", "AUROC", "AURC")


def fmt(x: Optional[float]) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", nargs="+", required=True,
                   metavar=("NAME RSC_FILE", "BASELINE1_FILE"),
                   help="NAME RSC_FILE [BASELINE1_FILE]; repeat per dataset.")
    p.add_argument("--out_dir", default="results/cocoa")
    p.add_argument("--n_consistency", type=int, default=None,
                   help="Use only the first n independent answers for z_proxy "
                        "(default: the stored z_proxy, i.e. all N).")
    p.add_argument("--iso_folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=0,
                   help="Paired-bootstrap resamples for CoCoA vs RSC (0 = off).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    md_lines: List[str] = []

    for spec in args.dataset:
        if len(spec) not in (2, 3):
            p.error("--dataset takes NAME RSC_FILE [BASELINE1_FILE]")
        name, rsc_file = spec[0], Path(spec[1])
        vc_file = Path(spec[2]) if len(spec) == 3 else None
        print(f"\n== {name}: {rsc_file}")
        cases = build_cases(rsc_file, vc_file, args.n_consistency)
        add_cocoa(cases)
        labels = [c["correct"] for c in cases]
        print(f"  n = {len(cases)}, accuracy = {sum(labels) / len(labels):.3f}, "
              f"consistently-wrong (z=1 & wrong) = "
              f"{sum(1 for c in cases if c['z_proxy'] == 1 and not c['correct'])}")

        # per-case table
        case_path = out_dir / f"cocoa_cases_{name}.csv"
        with case_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
            w.writeheader()
            w.writerows(cases)

        methods = method_table(cases, args.iso_folds, args.seed)
        md_lines += [f"\n### {name} (n={len(cases)})\n",
                     "| Method | " + " | ".join(METRICS) + " | uses labels |",
                     "|---" * (len(METRICS) + 2) + "|"]
        for mname, spec_m in methods.items():
            res = all_metrics(labels, spec_m["conf"], spec_m["rank"])
            row = {"dataset": name, "method": mname, "n": len(cases),
                   "uses_labels": spec_m["uses_labels"],
                   **{k: res.get(k) for k in METRICS}}
            summary_rows.append(row)
            md_lines.append(f"| {mname} | " + " | ".join(fmt(res.get(k)) for k in METRICS)
                            + f" | {'yes' if spec_m['uses_labels'] else 'no'} |")
            print(f"  {mname:<16}" + "".join(f"{k}={fmt(res.get(k))}  " for k in METRICS))

        if args.bootstrap and "RSC" in methods:
            print(f"  paired bootstrap ({args.bootstrap}x), CoCoA - RSC, 95% CI:")
            for mname in ("CoCoA-VC", "CoCoA-Avg"):
                if mname in methods:
                    ci = paired_bootstrap(cases, methods[mname], methods["RSC"],
                                          args.bootstrap, args.seed)
                    print(f"    {mname:<10}" + "  ".join(
                        f"{k}=[{lo:+.3f}, {hi:+.3f}]" for k, (lo, hi) in ci.items()))
                    for k, (lo, hi) in ci.items():
                        summary_rows.append({"dataset": name,
                                             "method": f"{mname} - RSC (95% CI)",
                                             "n": len(cases), "uses_labels": False,
                                             k: f"[{lo:+.3f}, {hi:+.3f}]"})

    with (out_dir / "cocoa_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "n", "uses_labels", *METRICS])
        w.writeheader()
        w.writerows(summary_rows)
    (out_dir / "cocoa_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_dir / 'cocoa_summary.csv'}, {out_dir / 'cocoa_summary.md'}, "
          f"and per-case cocoa_cases_<dataset>.csv")


if __name__ == "__main__":
    main()
