"""rationale_consistency.py — 근거의 **평가 방향**이 실제 점수를 따라가는지 잰다.

왜 필요한가: rationale_eval.py의 유사도(n-gram·ROUGE)는 어휘 겹침만 본다.
평가 방향이 정반대여도 teacher의 상투구를 많이 쓰면 점수가 올라간다.
실제로 gold가 낮은 글에 대해 teacher는 부정 평가를 쓰는데, 어떤 arm은 긍정
평가를 쓰면서도 유사도가 더 높게 나오는 사례를 관측했다(2026-08-18).

여기서는 근거의 **극성**(긍정/부정)을 어휘로 추정하고, 그것이 gold 점수와
얼마나 상관하는지 본다. 상관이 높을수록 "점수에 부합하는 근거"다.

⚠️ 한계: 사전 기반 극성이라 반어·이중부정·완곡 표현을 놓친다. 절대 수준이 아니라
   **arm 간 상대 비교**로만 읽을 것. teacher 자신의 상관이 사실상 상한이다.

    python3 rationale_consistency.py --ref-file rationales_all.jsonl \\
        9b_par 9b_par_g2 9b_par_g4 9b_par_genonly
"""
import argparse, json, os, math

TRAITS = ("content", "organization", "expression")
KOR = {"content": "내용", "organization": "구성", "expression": "표현"}
ROOT = os.environ.get("AWES_ROOT", os.path.dirname(os.path.abspath(__file__)))
RUNS = os.environ.get("AWES_RUNS", os.path.join(ROOT, "runs"))
N_FOLDS = int(os.environ.get("AWES_NFOLDS", "4"))

# 국어 채점 근거에서 반복적으로 쓰이는 평가 어휘. 어간만 잡아 활용형을 포괄한다.
NEG = ["미흡", "부족", "아쉽", "단편적", "매끄럽지", "부자연", "불명확", "모호",
       "산만", "빈약", "취약", "오류", "어색", "반복", "장황", "비약", "근거 없",
       "설득력이 떨어", "떨어진다", "부재", "결여", "혼란", "불충분", "약하"]
# ⚠️ 짧은 어간은 부정 문맥에 끼어든다("구조적 **완결**성이 미흡하다"에서 '완결'이
#    긍정으로 잡힌다). 그래서 긍정어는 **서술까지 포함한 형태**로만 둔다.
POS = ["뛰어", "우수", "탁월", "명확히", "자연스럽", "돋보", "충실",
       "적절히", "체계적", "일관되", "설득력 있", "풍부", "정확히", "매끄럽게",
       "완결성을 갖추", "완결성이 잘", "완결된", "잘 갖춰", "잘 유지", "효과적"]


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def polarity(text):
    """(긍정어 수 − 부정어 수) / 전체. 어휘가 없으면 None(집계 제외)."""
    t = text or ""
    n = sum(t.count(w) for w in NEG)
    p = sum(t.count(w) for w in POS)
    return None if (n + p) == 0 else (p - n) / (p + n)


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    a, b = _rank(x), _rank(y)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return 0.0 if den == 0 else num / den


def load_pred(run):
    out = {}
    for k in range(N_FOLDS):
        p = os.path.join(RUNS, run, f"fold{k}", "predictions.jsonl")
        if not os.path.exists(p):
            continue
        for r in load_jsonl(p):
            out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref-file", required=True)
    args = ap.parse_args()

    ref = {r["id"]: r for r in load_jsonl(args.ref_file)}
    P = {a: load_pred(a) for a in args.runs}
    ids = sorted(set.intersection(*(set(p) for p in P.values())) & set(ref))
    print(f"공통 항목 n={len(ids)}")

    def row(tag, get):
        cells, cov = [], []
        for t in TRAITS:
            xs, ys = [], []
            miss = 0
            for i in ids:
                pol = polarity(get(i, t))
                if pol is None:
                    miss += 1
                    continue
                xs.append(pol); ys.append(float(ref[i]["score"][t]))
            cells.append(spearman(xs, ys))
            cov.append(1 - miss / len(ids))
        print(f"{tag:16s} " + "  ".join(f"{KOR[t]} {c:+.3f}" for t, c in zip(TRAITS, cells))
              + f"   | 평균 {sum(cells)/3:+.3f}  (어휘 검출률 {sum(cov)/3*100:.0f}%)")

    print("\n근거 극성 ↔ gold 점수의 Spearman (높을수록 점수에 부합하는 근거)")
    print("-" * 88)
    row("teacher(상한)", lambda i, t: (ref[i].get("rationale") or {}).get(t, ""))
    for a in args.runs:
        row(a, lambda i, t, a=a: ((P[a].get(i) or {}).get(t) or {}).get("rationale", ""))

    print("\n⚠️ 사전 기반 극성이라 반어·완곡을 놓친다. arm 간 상대 비교로만 읽을 것.")


if __name__ == "__main__":
    main()
