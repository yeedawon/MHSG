"""paper_eval.py — 논문 트랙 평가. arm 간 비교를 AES 표준 지표로 낸다.

대회의 블렌드 score는 쓰지 않는다(외부 독자에게 의미가 없다). AES 문헌 비교를 위해
**QWK**를 주 지표로 두고, 연속 gold를 살리는 Spearman·RMSE를 함께 보고한다.

  · QWK   — 예측·정답을 모두 정수화(사사오입)해 산출. AES 표준
  · ρ, RMSE — 연속 gold 기준. 정수화로 잃는 양을 같이 보이기 위해 정수판도 병기
  · 유의성 — 페어드 부트스트랩. 같은 에세이 집합을 리샘플해 두 arm에 동시 적용하므로
             글 난이도의 공통 변동이 상쇄된다

판정은 **4-fold OOF 전체**로 한다. 단일 fold 판정은 이 프로젝트에서 과거 3회 뒤집혔다.

    python3 paper_eval.py par par_nogen
    python3 paper_eval.py par seq_rs seq_sr --ref par --boot 2000

의존성 없음(순수 파이썬). GPU 불필요.
"""
import argparse
import json
import math
import os
import random

TRAITS = ("content", "organization", "expression")
ROOT = os.environ.get("AWES_ROOT", os.path.dirname(os.path.abspath(__file__)))
RUNS = os.environ.get("AWES_RUNS", os.path.join(ROOT, "runs"))
N_FOLDS = int(os.environ.get("AWES_NFOLDS", "4"))
SCALE = (1, 5)


# --------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------
def to_int(x):
    """사사오입. 파이썬 round()는 뱅커스라 round(2.5)=2로 어긋난다."""
    return max(SCALE[0], min(SCALE[1], int(math.floor(x + 0.5))))


def rmse(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def _rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return 0.0 if den == 0 else num / den


def qwk(a, b, lo=SCALE[0], hi=SCALE[1]):
    """Quadratic Weighted Kappa. a,b는 정수 등급 시퀀스."""
    k = hi - lo + 1
    n = len(a)
    O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[x - lo][y - lo] += 1
    ha = [0] * k
    hb = [0] * k
    for x in a:
        ha[x - lo] += 1
    for y in b:
        hb[y - lo] += 1
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            E = ha[i] * hb[j] / n
            num += w * O[i][j]
            den += w * E
    return 1.0 - num / den if den else 0.0


# --------------------------------------------------------------------------
# 로딩
# --------------------------------------------------------------------------
def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_gold():
    """id → {trait: float}. holdout_fold*.jsonl 합집합 = OOF 평가 대상 전체."""
    d = os.path.join(ROOT, "AWES", "data")
    if not os.path.isdir(d):
        d = ROOT
    g = {}
    # glob 금지 — 경로에 '[...]'가 들어가면 문자 클래스로 해석돼 조용히 0건이 된다.
    for fn in sorted(x for x in os.listdir(d)
                     if x.startswith("holdout_fold") and x.endswith(".jsonl")):
        for r in load_jsonl(os.path.join(d, fn)):
            g[r["id"]] = {t: float(r["score"][t]) for t in TRAITS}
    if not g:
        raise SystemExit(f"holdout_fold*.jsonl 없음 — AWES_ROOT 확인 (현재 {ROOT})")
    return g


def load_oof(run, folds=None):
    """OOF 예측. 폴드가 하나라도 없으면 어느 것이 없는지 알리고 None.

    folds를 주면 그 폴드만 읽는다 — 학습이 도는 중에 중간 점검할 때 쓴다.
    ⚠️ 부분 폴드 수치는 4-fold 결과와 표본이 달라 **같은 표에 놓으면 안 된다.**
    """
    if not os.path.isdir(RUNS):
        raise SystemExit(f"runs 폴더 없음: {RUNS} (AWES_RUNS로 지정)")
    merged = {}
    for k in (range(N_FOLDS) if folds is None else folds):
        p = os.path.join(RUNS, run, f"fold{k}", "predictions.jsonl")
        if not os.path.exists(p):
            have = sorted(x for x in os.listdir(os.path.join(RUNS, run))
                          if x.startswith("fold")) \
                if os.path.isdir(os.path.join(RUNS, run)) else "run 폴더 없음"
            print(f"  [load_oof] {run}: fold{k} 없음 — 보유 {have}")
            return None
        for r in load_jsonl(p):
            merged[r["id"]] = {t: float(r[t]["score"]) for t in TRAITS}
    return merged


# --------------------------------------------------------------------------
# 평가
# --------------------------------------------------------------------------
def vectors(ids, gold, pred):
    """3영역 평균 기준 (연속, 정수) 벡터와 영역별 정수 등급."""
    gc = [sum(gold[i][t] for t in TRAITS) / 3.0 for i in ids]
    pc = [sum(pred[i][t] for t in TRAITS) / 3.0 for i in ids]
    pi = [sum(to_int(pred[i][t]) for t in TRAITS) / 3.0 for i in ids]
    per = {t: ([to_int(gold[i][t]) for i in ids], [to_int(pred[i][t]) for i in ids])
           for t in TRAITS}
    return gc, pc, pi, per


def report(name, ids, gold, pred):
    gc, pc, pi, per = vectors(ids, gold, pred)
    q = sum(qwk(*per[t]) for t in TRAITS) / len(TRAITS)
    row = {
        "run": name, "n": len(ids),
        "qwk": q,
        "rho": spearman(gc, pc), "rho_int": spearman(gc, pi),
        "rmse": rmse(gc, pc), "rmse_int": rmse(gc, pi),
        "per_trait_qwk": {t: qwk(*per[t]) for t in TRAITS},
    }
    return row


def paired_bootstrap(ids, gold, a, b, metric="qwk", boot=2000, seed=42):
    """metric(a) − metric(b). 클수록 좋은 방향으로 통일(RMSE는 부호 반전)."""
    def m(sub):
        ra, rb = report("a", sub, gold, a), report("b", sub, gold, b)
        if metric == "rmse":
            return -ra["rmse"], -rb["rmse"]
        return ra[metric], rb[metric]

    oa, ob = m(ids)
    obs = oa - ob
    rng = random.Random(seed)
    n = len(ids)
    diffs = []
    for _ in range(boot):
        sub = [ids[rng.randrange(n)] for _ in range(n)]
        xa, xb = m(sub)
        diffs.append(xa - xb)
    diffs.sort()
    lo = diffs[int(0.025 * boot)]
    hi = diffs[min(int(0.975 * boot), boot - 1)]
    return obs, lo, hi, sum(1 for d in diffs if d > 0) / boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="비교할 arm (runs/<name>)")
    ap.add_argument("--ref", default=None, help="기준 arm (기본: 첫 번째)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--metric", default="qwk", choices=("qwk", "rho", "rho_int", "rmse"))
    ap.add_argument("--folds", default=None,
                    help="쓸 fold (예: 0 또는 0,1). 학습 중 중간 점검용 — "
                         "부분 폴드 수치는 4-fold 결과와 비교 불가")
    args = ap.parse_args()
    folds = None
    if args.folds:
        folds = tuple(int(x) for x in args.folds.replace(" ", "").split(",") if x != "")
        print(f"⚠️ fold {folds}만 사용 — 표본이 달라 4-fold 결과와 같은 표에 놓지 말 것")

    gold = load_gold()
    preds = {}
    for nm in args.runs:
        p = load_oof(nm, folds)
        if p is None:
            raise SystemExit(f"{nm}: OOF 미완성 — 비교 불가")
        preds[nm] = p

    ids = sorted(set.intersection(*(set(p) for p in preds.values())) & set(gold))
    print(f"공통 평가 대상 n={len(ids)} (arm {len(preds)}개)")
    if len(ids) < 100:
        raise SystemExit("공통 id가 너무 적다 — fold 분할이 같은지 확인할 것")
    if folds is None and len(ids) < 1900:
        print(f"⚠️ 4-fold인데 {len(ids)}건뿐 — 일부 폴드의 예측이 비었을 수 있다")

    rows = [report(nm, ids, gold, preds[nm]) for nm in args.runs]
    print(f"\n{'arm':16s} {'QWK↑':>7s} {'ρ↑':>7s} {'ρint↑':>7s} "
          f"{'RMSE↓':>7s} {'RMSEint↓':>9s} | " +
          " ".join(f"{t[:4]:>6s}" for t in TRAITS) + "  (영역별 QWK)")
    print("-" * 96)
    for r in rows:
        print(f"{r['run']:16s} {r['qwk']:7.4f} {r['rho']:7.4f} {r['rho_int']:7.4f} "
              f"{r['rmse']:7.4f} {r['rmse_int']:9.4f} | " +
              " ".join(f"{r['per_trait_qwk'][t]:6.3f}" for t in TRAITS))

    ref = args.ref or args.runs[0]
    print(f"\n페어드 부트스트랩 {args.boot}회 · 기준={ref} · 지표={args.metric}")
    print(f"{'arm':16s} {'Δ vs ref':>10s} {'95% CI':>21s} {'P(Δ>0)':>8s}")
    print("-" * 60)
    for nm in args.runs:
        if nm == ref:
            print(f"{nm:16s} {'(기준)':>10s}")
            continue
        obs, lo, hi, p = paired_bootstrap(ids, gold, preds[nm], preds[ref],
                                          args.metric, args.boot)
        mark = "" if lo <= 0 <= hi else "  ✅유의"
        print(f"{nm:16s} {obs:+10.4f} [{lo:+.4f}, {hi:+.4f}] {p:8.3f}{mark}")

    print("\n95% CI가 0을 포함하면 그 차이는 이 데이터로 뒷받침되지 않는다.")


if __name__ == "__main__":
    main()
