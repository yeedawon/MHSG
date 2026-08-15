"""build_fold_rationales.py — 합성 근거 원본을 fold별 학습 입력으로 변환한다.

합성 결과물은 `{id, score:{...}, rationale:{content, organization, expression}}`
형태인데, 학습 로더(`awes_common/data.py`)는 제출 스키마
`{id, content:{score, rationale}, ...}`를 읽는다. 이 스크립트가 그 변환과
**fold별 누수 제거**를 함께 한다.

누수 규칙: fold-k 학습 입력에서 holdout-k의 id를 뺀다. 이걸 빠뜨리면 지표가
통째로 무의미해지므로, 변환 시점에 강제하고 결과를 세어 보고한다.

입력 근거 파일은 두 스키마를 모두 받는다:
  · `rationale: {content, organization, expression}`                 (trait 키)
  · `rationale: {content_rationale, organization_rationale, ...}`    (구 스키마)

    python3 build_fold_rationales.py ~/Downloads/rationales_all.jsonl \\
        --root /path/to/data

의존성 없음. GPU 불필요.
"""
import argparse
import json
import os

TRAITS = ("content", "organization", "expression")


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extract(row):
    """근거 행 → {trait: rationale}. 두 스키마를 모두 받는다."""
    r = row.get("rationale") or {}
    out = {}
    for t in TRAITS:
        v = r.get(t) or r.get(f"{t}_rationale") or ""
        out[t] = v.strip() if isinstance(v, str) else ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="합성 근거 jsonl (2,000건)")
    ap.add_argument("--root", default=None, help="AWES_ROOT (기본: 환경변수)")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--require-ok", action="store_true",
                    help="ok 필드가 True인 행만 사용(검증기 통과분만)")
    args = ap.parse_args()

    root = args.root or os.environ.get("AWES_ROOT") or os.getcwd()
    data = os.path.join(root, "AWES", "data")
    out_dir = os.path.join(root, "AWES", "rationales", "fold_train")
    os.makedirs(out_dir, exist_ok=True)

    rows = load_jsonl(args.source)
    if args.require_ok:
        before = len(rows)
        rows = [r for r in rows if r.get("ok") is not False]
        if len(rows) != before:
            print(f"ok=False 제외: {before - len(rows)}건")

    corpus = {r["id"] for r in load_jsonl(os.path.join(data, "train_with_folds.jsonl"))}
    print(f"근거 {len(rows)}건 · 코퍼스 {len(corpus)}건")

    by_id, empty = {}, []
    for r in rows:
        rat = extract(r)
        if not all(rat[t] for t in TRAITS):
            empty.append(r["id"])
            continue
        by_id[r["id"]] = {
            "id": r["id"],
            **{t: {"score": float(r["score"][t]), "rationale": rat[t]} for t in TRAITS},
        }
    if empty:
        print(f"⚠️ 3영역 근거가 안 채워진 행 {len(empty)}건 제외 (예: {empty[:3]})")

    unknown = sorted(set(by_id) - corpus)
    if unknown:
        print(f"⚠️ 코퍼스에 없는 근거 id {len(unknown)}건 제외 (예: {unknown[:3]})")
        for i in unknown:
            by_id.pop(i, None)

    missing = sorted(corpus - set(by_id))
    if missing:
        print(f"⚠️ 근거가 없는 코퍼스 id {len(missing)}건 — 그 글은 학습셋에서 빠진다 "
              f"(예: {missing[:3]})")

    total = 0
    for k in range(args.folds):
        hp = os.path.join(data, f"holdout_fold{k}.jsonl")
        if not os.path.exists(hp):
            raise SystemExit(f"holdout_fold{k}.jsonl 없음: {data}")
        holdout = {r["id"] for r in load_jsonl(hp)}
        train = [by_id[i] for i in sorted(by_id) if i not in holdout]
        op = os.path.join(out_dir, f"rationale_train_fold{k}.jsonl")
        with open(op, "w", encoding="utf-8") as f:
            for r in train:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        leaked = {r["id"] for r in train} & holdout
        assert not leaked, f"fold{k} 누수 {len(leaked)}건 — 변환 로직 오류"
        print(f"  fold{k}: 학습 {len(train)}건 (holdout {len(holdout)}건 제외) → {op}")
        total += len(train)

    print(f"\n✅ 완료 · 총 {total}행 기록")
    print("   다음: python3 check_data.py 로 배치를 검증하세요.")


if __name__ == "__main__":
    main()
