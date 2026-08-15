import json, os
SRC = "/Users/idawon/Downloads/train_with_folds.jsonl"
D  = "./fake_root/AWES/data"; FT = "./fake_root/AWES/rationales/fold_train"
os.makedirs(D, exist_ok=True); os.makedirs(FT, exist_ok=True)
T = ("content","organization","expression")

def wr(rows, p):
    with open(p,"w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

rows = [json.loads(l) for l in open(SRC,encoding="utf-8") if l.strip()]
wr(rows, f"{D}/train_with_folds.jsonl")
for k in range(5):
    hold  = [r for r in rows if r["_fold"] == k]
    train = [r for r in rows if r["_fold"] != k]
    wr(hold, f"{D}/holdout_fold{k}.jsonl")
    # 근거 파일은 제출 스키마 — 본문 없이 점수+근거만
    wr([{"id": r["id"], **{t: {"score": r["score"][t], "rationale": f"{t} 더미 근거."} for t in T}}
        for r in train], f"{FT}/rationale_train_fold{k}.jsonl")
    print(f"fold{k}: train={len(train)} holdout={len(hold)}")
