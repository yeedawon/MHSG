"""rationale_eval.py — arm별 생성 근거의 품질을 teacher 원본과 대조한다. GPU 불필요.

왜 필요한가: 생성 가중치를 낮추면 채점은 좋아졌다(역U자, PAPER_TRACK.md §1-d).
남은 질문은 **"그러면 근거는 나빠지는가"**다. 나빠지지 않으면 저가중 설정이 모든
면에서 우월하고, 나빠지면 진짜 상충 곡선이 된다.

⚠️ 여기서 재는 것은 **teacher 근거와의 일치도**이지 근거의 절대적 품질이 아니다.
   모델은 teacher 근거를 정답으로 학습했으므로 "학습한 과제를 홀드아웃에서 얼마나
   재현하는가"로는 타당하지만, "사람이 보기에 좋은가"는 다른 질문이다.
   논문에서는 이 한계를 반드시 명시하고, LLM judge나 사람 평가로 보완할 것.

지표:
  · char-3gram F1   — 어절 경계에 덜 민감해 한국어에 무난하다. 빠르다.
  · ROUGE-L F1      — 문자 LCS 기반. 순서를 반영한다(--fast로 생략 가능).
  · distinct-3gram  — 반복 퇴화 탐지. 낮으면 같은 말을 되풀이한다는 뜻.
  · 빈 근거 비율    — 생성 실패율.

유의성은 항목별 점수를 구한 뒤 **페어드 부트스트랩**으로 본다.

    python3 rationale_eval.py --ref-file rationales_all.jsonl \\
        9b_par 9b_par_g4 9b_par_g2 9b_par_genonly --ref 9b_par_genonly

의존성 없음.
"""
import argparse
import json
import os
import random

TRAITS = ("content", "organization", "expression")
ROOT = os.environ.get("AWES_ROOT", os.path.dirname(os.path.abspath(__file__)))
RUNS = os.environ.get("AWES_RUNS", os.path.join(ROOT, "runs"))
N_FOLDS = int(os.environ.get("AWES_NFOLDS", "4"))


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _norm(s):
    """공백 정규화. 줄바꿈·중복 공백이 지표를 흔들지 않게 한다."""
    return " ".join((s or "").split())


def ngrams(s, n=3):
    s = s.replace(" ", "")
    return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]


def ngram_f1(a, b, n=3):
    """문자 n-gram 다중집합 F1."""
    A, B = ngrams(a, n), ngrams(b, n)
    if not A or not B:
        return 0.0
    ca, cb = {}, {}
    for g in A:
        ca[g] = ca.get(g, 0) + 1
    for g in B:
        cb[g] = cb.get(g, 0) + 1
    inter = sum(min(v, cb.get(g, 0)) for g, v in ca.items())
    if inter == 0:
        return 0.0
    p, r = inter / len(A), inter / len(B)
    return 2 * p * r / (p + r)


def lcs_len(a, b):
    """문자 LCS 길이. 행 두 개만 유지해 메모리를 아낀다."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(a, b):
    a, b = a.replace(" ", ""), b.replace(" ", "")
    if not a or not b:
        return 0.0
    l = lcs_len(a, b)
    if l == 0:
        return 0.0
    p, r = l / len(a), l / len(b)
    return 2 * p * r / (p + r)


def distinct_ratio(s, n=3):
    g = ngrams(s, n)
    return len(set(g)) / len(g) if g else 0.0


def load_pred_rationales(run):
    """id → {trait: rationale}. 4-fold 전체."""
    out = {}
    for k in range(N_FOLDS):
        p = os.path.join(RUNS, run, f"fold{k}", "predictions.jsonl")
        if not os.path.exists(p):
            print(f"  [{run}] fold{k} 없음 — 건너뜀")
            continue
        for r in load_jsonl(p):
            out[r["id"]] = {t: _norm((r.get(t) or {}).get("rationale", "")) for t in TRAITS}
    return out


def load_reference(path):
    """teacher 근거 원본. rationale:{trait} / rationale:{trait_rationale} 둘 다 받는다."""
    ref = {}
    for r in load_jsonl(path):
        d = r.get("rationale") or {}
        ref[r["id"]] = {t: _norm(d.get(t) or d.get(f"{t}_rationale") or "") for t in TRAITS}
    return ref


def per_item_scores(pred, ref, ids, fast=False):
    """항목×영역 단위 점수 벡터. 부트스트랩은 이 스칼라들을 리샘플한다."""
    f1, rl, dist, empty, plen = [], [], [], [], []
    for i in ids:
        for t in TRAITS:
            p, g = pred[i][t], ref[i][t]
            empty.append(1.0 if not p else 0.0)
            plen.append(float(len(p)))
            f1.append(ngram_f1(p, g))
            rl.append(0.0 if fast else rouge_l(p, g))
            dist.append(distinct_ratio(p))
    return {"f1": f1, "rouge_l": rl, "distinct": dist, "empty": empty, "len": plen}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def paired_bootstrap(a, b, boot=2000, seed=42):
    """같은 항목 인덱스를 두 arm에 동시 적용 — 글 난이도의 공통 변동이 상쇄된다."""
    n = len(a)
    obs = mean(a) - mean(b)
    rng = random.Random(seed)
    diffs = []
    for _ in range(boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(mean([a[i] for i in idx]) - mean([b[i] for i in idx]))
    diffs.sort()
    lo, hi = diffs[int(0.025 * boot)], diffs[min(int(0.975 * boot), boot - 1)]
    p = sum(1 for d in diffs if d > 0) / boot
    return obs, lo, hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref-file", required=True, help="teacher 근거 원본 jsonl")
    ap.add_argument("--ref", default=None, help="비교 기준 arm (기본: 첫 번째)")
    ap.add_argument("--metric", default="f1", choices=("f1", "rouge_l", "distinct"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--fast", action="store_true", help="ROUGE-L 생략(LCS가 느릴 때)")
    args = ap.parse_args()

    # --fast는 ROUGE-L을 계산하지 않고 0으로 채운다. 그 상태로 검정을 돌리면
    # 0끼리 비교해 Δ=0, CI=[0,0], p<0.001 같은 무의미한 표가 나온다(2026-08-18 실측).
    if args.fast and args.metric == "rouge_l":
        raise SystemExit("--fast는 ROUGE-L을 건너뛴다 — --metric rouge_l과 함께 쓸 수 없다.")

    ref = load_reference(args.ref_file)
    preds = {nm: load_pred_rationales(nm) for nm in args.runs}

    ids = sorted(set.intersection(*(set(p) for p in preds.values())) & set(ref))
    print(f"공통 항목 n={len(ids)} (arm {len(preds)}개) · 영역 3개 → 비교 단위 {len(ids)*3}")
    if len(ids) < 100:
        raise SystemExit("공통 id가 너무 적다 — fold 구성을 확인할 것")

    S = {nm: per_item_scores(preds[nm], ref, ids, args.fast) for nm in args.runs}

    print(f"\n{'arm':16s} {'3gramF1↑':>9s} {'ROUGE-L↑':>9s} {'distinct↑':>10s} "
          f"{'빈근거':>7s} {'길이':>7s}")
    print("-" * 66)
    for nm in args.runs:
        s = S[nm]
        rl = "  (생략)" if args.fast else f"{mean(s['rouge_l']):9.4f}"
        print(f"{nm:16s} {mean(s['f1']):9.4f} {rl:>9s} {mean(s['distinct']):10.4f} "
              f"{mean(s['empty'])*100:6.1f}% {mean(s['len']):7.1f}")

    base = args.ref or args.runs[0]
    print(f"\n페어드 부트스트랩 {args.boot}회 · 기준={base} · 지표={args.metric}")
    print(f"{'arm':16s} {'Δ vs ref':>10s} {'95% CI':>21s} {'P(Δ>0)':>8s} {'양측p':>7s}")
    print("-" * 70)
    for nm in args.runs:
        if nm == base:
            print(f"{nm:16s} {'(기준)':>10s}")
            continue
        xa, xb = S[nm][args.metric], S[base][args.metric]
        if xa == xb:      # 두 벡터가 완전히 같다 = 지표가 계산되지 않았다는 뜻
            print(f"{nm:16s} {'—':>10s}  (두 arm의 {args.metric} 값이 동일 — 지표 미계산?)")
            continue
        obs, lo, hi, p = paired_bootstrap(xa, xb, args.boot)
        pv = 2.0 * min(p, 1.0 - p)
        pv_s = f"<{2.0/args.boot:.3f}" if pv == 0 else f"{pv:.3f}"
        mark = "" if lo <= 0 <= hi else "  ✅유의"
        print(f"{nm:16s} {obs:+10.4f} [{lo:+.4f}, {hi:+.4f}] {p:8.3f} {pv_s:>7s}{mark}")

    print("\n⚠️ 이 지표는 teacher 근거와의 일치도다. 절대적 품질이 아니며,")
    print("   논문에서는 한계를 명시하고 LLM judge나 사람 평가로 보완할 것.")


if __name__ == "__main__":
    main()
