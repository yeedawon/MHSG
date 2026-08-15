# -*- coding: utf-8 -*-
"""
GPU 없이 공통 인터페이스 배선 전체를 검증한다.

    python3 -m awes_common.test_interface

fold 픽스처는 train_with_folds.jsonl에서 자동 재구성한다. 코퍼스 위치는
AWES_CORPUS로 지정할 수 있고, 없으면 아래 후보 경로를 순서대로 찾는다.
서버에서는 실제 AWES_ROOT를 그대로 두면 재구성 없이 실데이터로 돈다.
"""
import json, os, sys, math, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

_TRAITS = ("content", "organization", "expression")


def _build_fixture(corpus_path):
    """train_with_folds.jsonl → 임시 AWES_ROOT (fold별 holdout + 근거 파일 재구성).
    코퍼스의 기존 _fold 수와 무관하게 C.N_FOLDS로 재배정해 자체 일관성을 보장한다."""
    from awes_common import config as C
    nf = C.N_FOLDS
    root = tempfile.mkdtemp(prefix="awes_fixture_")
    D = os.path.join(root, "AWES", "data")
    FT = os.path.join(root, "AWES", "rationales", "fold_train")
    os.makedirs(D); os.makedirs(FT)
    rows = [json.loads(l) for l in open(corpus_path, encoding="utf-8") if l.strip()]
    for i, r in enumerate(rows):            # C.N_FOLDS로 재배정 (테스트 자체 일관성)
        r["_fold"] = i % nf

    def wr(rs, p):
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    wr(rows, os.path.join(D, "train_with_folds.jsonl"))
    for k in range(nf):
        wr([r for r in rows if r["_fold"] == k], os.path.join(D, f"holdout_fold{k}.jsonl"))
        wr([{"id": r["id"], **{t: {"score": r["score"][t], "rationale": f"{t} 더미 근거."}
             for t in _TRAITS}} for r in rows if r["_fold"] != k],
           os.path.join(FT, f"rationale_train_fold{k}.jsonl"))
    return root


# 이미 실데이터 AWES_ROOT가 준비됐으면(서버) 그대로 쓰고, 아니면 픽스처 재구성
if not os.path.exists(os.path.join(os.environ.get("AWES_ROOT", ""), "AWES", "data",
                                   "train_with_folds.jsonl")):
    _cands = [os.environ.get("AWES_CORPUS"),
              os.path.join(PROJECT, "train_with_folds.jsonl"),
              os.path.expanduser("~/Downloads/train_with_folds.jsonl")]
    _corpus = next((c for c in _cands if c and os.path.exists(c)), None)
    if _corpus is None:
        sys.exit("train_with_folds.jsonl을 찾을 수 없습니다. AWES_CORPUS로 경로를 지정하세요.")
    os.environ["AWES_ROOT"] = _build_fixture(_corpus)
    print(f"(픽스처 재구성: {_corpus})")

os.environ["AWES_RUNS"] = tempfile.mkdtemp(prefix="awes_runs_test_")  # 매 실행 깨끗이

from awes_common import (verify_all_folds, load_fold, evaluate, Prediction,
                         rmse, spearman, run_cv, TrainConfig, GlobalMeanPipeline)
from awes_common.judge import parse_judge_output, build_judge_prompt

print("=== 1. 5-fold 무결성 ===")
assert verify_all_folds()

print("\n=== 2. 조인 확인 (근거 파일에 본문 없음 → 코퍼스와 조인돼야 함) ===")
fd = load_fold(0)
e = fd.train[0]
print(f"  {e.id}: essay={len(e.essay)}자, prompt={len(e.prompt)}자, 근거={e.has_rationale}")
assert e.essay and e.prompt and e.has_rationale
assert fd.eval[0].rationales is None, "평가셋에 근거가 있으면 안 됨"

print("\n=== 3. 누수 탐지가 실제로 작동하는가 (음성 테스트) ===")
# 실데이터(프로덕션 fold_train)를 절대 건드리지 않는다 — 실코퍼스로 격리 픽스처를
# 새로 만들어 거기에만 누수를 주입하고, 그 동안만 AWES_ROOT를 임시로 가리킨다.
import awes_common.data as D, json, awes_common.config as C
T = ("content", "organization", "expression")
_saved_root = os.environ.get("AWES_ROOT")            # 서버에선 None일 수 있음
_iso = _build_fixture(C.TRAIN_WITH_FOLDS)            # 실코퍼스에서 5-fold 재구성 (격리)
os.environ["AWES_ROOT"] = _iso
try:
    bad = C.fold_train_path(0)                        # 격리본의 fold0 학습 파일
    hold0 = D.load_jsonl(C.holdout_path(0))
    leak = {"id": hold0[0]["id"],
            **{t: {"score": hold0[0]["score"][t], "rationale": "x"} for t in T}}
    with open(bad, "a", encoding="utf-8") as _f:
        _f.write(json.dumps(leak, ensure_ascii=False) + "\n")
    try:
        load_fold(0); print("  ❌ 누수를 잡지 못함"); sys.exit(1)
    except AssertionError as err:
        print(f"  ✅ 누수 차단됨: {str(err)[:70]}...")
finally:
    if _saved_root is None:
        os.environ.pop("AWES_ROOT", None)
    else:
        os.environ["AWES_ROOT"] = _saved_root

print("\n=== 4. 지표 정확도 (scipy 대조) ===")
yt = [3.5, 4.0, 2.0, 5.0, 3.5, 1.0, 4.25, 3.0]
yp = [3.0, 4.5, 2.5, 4.0, 3.5, 2.0, 4.00, 2.5]
try:
    from scipy.stats import spearmanr
    import numpy as np
    ref_s = spearmanr(yt, yp).correlation
    ref_r = float(np.sqrt(np.mean((np.array(yt)-np.array(yp))**2)))
    print(f"  spearman ours={spearman(yt,yp):.10f} scipy={ref_s:.10f}")
    print(f"  rmse     ours={rmse(yt,yp):.10f} numpy={ref_r:.10f}")
    assert abs(spearman(yt,yp)-ref_s) < 1e-9 and abs(rmse(yt,yp)-ref_r) < 1e-9
    print("  ✅ 일치 (동점 보정 포함)")
except ImportError:
    print("  (scipy 없음 — 대조 생략)")
assert spearman([1,2,3],[9,9,9]) == 0.0, "상수 예측은 ρ=0"
print("  ✅ 상수 예측 ρ=0.0 (zero-signal floor)")

print("\n=== 5. 예측 누락 거부 ===")
try:
    evaluate(fd.eval, [Prediction(id=x.id, scores={t:3.0 for t in T}) for x in fd.eval[:-1]])
    print("  ❌ 누락을 통과시킴"); sys.exit(1)
except ValueError as err:
    print(f"  ✅ 거부됨: {str(err)[:60]}...")

print("\n=== 6. judge 파서 (thinking 블록 + 후행 콤마) ===")
p = parse_judge_output('<think>음...</think>```json\n{"content":4,"organization":3,"expression":5,}\n```')
print(f"  {p}"); assert p == {"content":4.0,"organization":3.0,"expression":5.0}
assert parse_judge_output("설명만 있고 JSON 없음") is None
assert parse_judge_output('{"content":9}') is None  # 필드 누락
print("  ✅ 파싱/거부 정상")
jp = build_judge_prompt(fd.eval[0], Prediction(id=fd.eval[0].id, scores={t:3.5 for t in T},
                                               rationales={t:"근거." for t in T}))
assert "학생 글" in jp and "1-5 정수" in jp
print(f"  ✅ judge 프롬프트 {len(jp)}자")

print("\n=== 7. run_cv 배선 (GlobalMean 5-fold) ===")
cv = run_cv(GlobalMeanPipeline, TrainConfig(), check_gpu=False)
d = cv.to_dict()
print(f"\n  기대치(핸드오프 1.1): RMSE≈0.735, ρ=0.00")
print(f"  실측: RMSE={d['rmse_mean']:.4f}, ρ={d['spearman_mean']:.4f}")
assert abs(d["rmse_mean"] - 0.735) < 0.05, d["rmse_mean"]
assert abs(d["spearman_mean"]) < 1e-9
run = os.path.join(os.environ["AWES_RUNS"], "global_mean")
for f in ["config.json", "cv_result.json", "fold0/predictions.jsonl"]:
    assert os.path.exists(os.path.join(run, f)), f
print(f"  ✅ 산출물 저장 확인")

print("\n=== 8. CUDA_VISIBLE_DEVICES 가드 ===")
import awes_common.config as C
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
try:
    C.require_gpu_pinned(); print("  ❌ 가드 미작동"); sys.exit(1)
except RuntimeError:
    print("  ✅ 미지정 시 중단됨")

print("\n" + "="*50 + "\n✅ 전체 통과")
