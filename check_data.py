"""check_data.py — 학습 전에 데이터 배치를 검증한다. 학습·GPU 불필요.

이 저장소는 데이터를 포함하지 않는다(README의 라이선스 항목 참조). 각자 받은
데이터를 아래 배치로 두었는지, 그리고 **학습을 조용히 망치는 세 가지**가 없는지
확인한다.

  1. 경로 오지정 — AWES_ROOT가 틀리면 gold를 부분만 읽고도 그냥 돌아간다
  2. id 불일치 — 근거 파일과 본문 코퍼스의 id 집합이 어긋나면 학습셋이 조용히 줄어든다
  3. 누수 — fold-k 학습 입력에 holdout-k가 섞이면 지표가 통째로 무의미해진다

    AWES_ROOT=/path/to/data python3 check_data.py
"""
import json
import os
import sys

TRAITS = ("content", "organization", "expression")
ROOT = os.environ.get("AWES_ROOT", os.path.dirname(os.path.abspath(__file__)))
N_FOLDS = int(os.environ.get("AWES_NFOLDS", "4"))

DATA = os.path.join(ROOT, "AWES", "data")
RAT = os.path.join(ROOT, "AWES", "rationales", "fold_train")

ok_n, bad_n = 0, 0


def ok(msg):
    global ok_n
    ok_n += 1
    print(f"  ✅ {msg}")


def bad(msg):
    global bad_n
    bad_n += 1
    print(f"  ❌ {msg}")


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    print(f"AWES_ROOT = {ROOT}")
    print(f"  데이터   {DATA}")
    print(f"  근거     {RAT}\n")

    if not os.path.isdir(DATA):
        bad(f"데이터 폴더 없음: {DATA}")
        print("\nREADME의 '데이터 배치'대로 두었는지 확인하세요.")
        return 1

    # --- 1. 본문 코퍼스 -----------------------------------------------------
    corpus_p = os.path.join(DATA, "train_with_folds.jsonl")
    if not os.path.exists(corpus_p):
        bad(f"train_with_folds.jsonl 없음 ({DATA})")
        return 1
    corpus = {r["id"]: r for r in load_jsonl(corpus_p)}
    ok(f"본문 코퍼스 {len(corpus)}건")

    need = ("id", "prompt", "essay", "score")
    miss = [k for k in need if k not in next(iter(corpus.values()))]
    if miss:
        bad(f"코퍼스 필수 필드 누락: {miss} (필요: {need})")
    else:
        ok(f"코퍼스 스키마 {need}")

    # --- 2. fold별 holdout / 근거 ------------------------------------------
    holdout = {}
    for k in range(N_FOLDS):
        p = os.path.join(DATA, f"holdout_fold{k}.jsonl")
        if not os.path.exists(p):
            bad(f"holdout_fold{k}.jsonl 없음")
            continue
        rows = load_jsonl(p)
        holdout[k] = {r["id"] for r in rows}
        unknown = [r["id"] for r in rows if r["id"] not in corpus]
        if unknown:
            bad(f"fold{k} holdout에 코퍼스에 없는 id {len(unknown)}건 (예: {unknown[:3]})")
        else:
            ok(f"fold{k} holdout {len(rows)}건 — 코퍼스와 id 일치")

    union = set().union(*holdout.values()) if holdout else set()
    if holdout:
        overlap = sum(len(a & b) for i, a in holdout.items()
                      for j, b in holdout.items() if i < j)
        if overlap:
            bad(f"holdout끼리 겹침 {overlap}건 — fold 분할이 배타적이지 않다")
        else:
            ok(f"holdout 4분할 배타 · 합집합 {len(union)}건")
        if len(union) != len(corpus):
            bad(f"holdout 합집합({len(union)}) ≠ 코퍼스({len(corpus)}) — 분할 누락/중복")

    # --- 3. 근거 파일과 누수 -------------------------------------------------
    if not os.path.isdir(RAT):
        bad(f"근거 폴더 없음: {RAT} — 생성 헤드 학습에 필수")
    else:
        for k in range(N_FOLDS):
            p = os.path.join(RAT, f"rationale_train_fold{k}.jsonl")
            if not os.path.exists(p):
                bad(f"rationale_train_fold{k}.jsonl 없음")
                continue
            rows = load_jsonl(p)
            ids = {r["id"] for r in rows}
            missing_text = [i for i in ids if i not in corpus]
            if missing_text:
                bad(f"fold{k} 근거 {len(missing_text)}건이 코퍼스에 없다 "
                    f"(예: {missing_text[:3]}) — 학습셋이 조용히 줄어든다")
            leaked = ids & holdout.get(k, set())
            if leaked:
                bad(f"🔴 fold{k} 누수: 학습 근거에 holdout id {len(leaked)}건 "
                    f"(예: {sorted(leaked)[:3]})")
            empty = [r["id"] for r in rows
                     if not all((r.get(t) or {}).get("rationale", "").strip()
                                for t in TRAITS)]
            if empty:
                bad(f"fold{k} 근거 비어 있는 샘플 {len(empty)}건 (예: {empty[:3]})")
            if not (missing_text or leaked or empty):
                ok(f"fold{k} 근거 {len(rows)}건 — 조인·누수·공백 이상 없음")

    print(f"\n통과 {ok_n} / 실패 {bad_n}")
    if bad_n:
        print("❌ 위 항목을 고친 뒤 학습하세요. 이 상태로 돌리면 지표가 무의미합니다.")
    else:
        print("✅ 데이터 배치 정상 — PAPER_TRACK.md의 Phase 1로 진행하세요.")
    return 1 if bad_n else 0


if __name__ == "__main__":
    sys.exit(main())
