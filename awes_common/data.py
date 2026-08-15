# -*- coding: utf-8 -*-
"""
data.py — 멀티태스크/autoregressive 두 파이프라인이 공유하는 데이터 로더.

설계 배경 (핸드오프 3.3, 4.1):
  - fold_train/rationale_train_fold{k}.jsonl 은 "제출 스키마"라서 점수+근거만 있고
    essay/prompt 본문이 없다. 학습에는 본문이 필요하므로 train_with_folds.jsonl 과
    id로 조인한다.
  - holdout_fold{k}.jsonl 은 본문+점수는 있지만 근거가 없다 (평가용이므로 정상).
  - hard rule: fold-k 학습 입력에 holdout_fold{k} id가 단 하나라도 섞이면 안 된다.
    → load_fold()가 매번 assert로 재검증한다. 파일 생성 시점의 assert를 믿지 않고
      학습 직전에 한 번 더 확인하는 이중 방어선.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from . import config as C


# 주최측 _extract_first_json은 문자열 내부 중괄호까지 세므로, 근거의 ASCII 중괄호를
# 전각으로 바꿔 어떤 생성 출력이 와도 첫 JSON 추출이 깨지지 않게 한다.
_BRACE_SAFE = str.maketrans({"{": "｛", "}": "｝"})


def _brace_safe(s: str) -> str:
    return (s or "").translate(_BRACE_SAFE)


# ---------------------------------------------------------------------------
# 1. 레코드
# ---------------------------------------------------------------------------
@dataclass
class Example:
    """두 아키텍처가 공통으로 소비하는 최소 단위."""

    id: str
    prompt_num: str
    prompt: str
    essay: str
    scores: Dict[str, float]                        # {trait: 1.0~5.0}
    rationales: Optional[Dict[str, str]] = None     # {trait: str}, 평가셋은 None
    meta: Dict = field(default_factory=dict)

    @property
    def has_rationale(self) -> bool:
        return self.rationales is not None

    def score_vector(self) -> List[float]:
        return [self.scores[t] for t in C.TRAITS]


@dataclass
class Prediction:
    """파이프라인 출력. rationale은 LLM Judge 항목(10%)에만 쓰인다."""

    id: str
    scores: Dict[str, float]
    rationales: Optional[Dict[str, str]] = None

    def to_submission(self) -> Dict:
        """제출 스키마: {"id", "content": {"score", "rationale"}, ...}"""
        out = {"id": self.id}
        for t in C.TRAITS:
            out[t] = {
                "score": round(float(self.scores[t]), 4),
                "rationale": (self.rationales or {}).get(t, ""),
            }
        return out

    def to_model_output(self, decimals: int = 4) -> str:
        """주최측 파서(_parse_model_output)가 읽는 '모델 원본 출력' 문자열.

        시나리오 A: score = 회귀헤드 값(self.scores), rationale = 생성.
        주최측 파서 요구사항에 정확히 맞춘다:
          · top-level 키 content/organization/expression (판정 대상은 top-level).
          · 각 영역은 dict이며 score 필수(+judge 10%용 rationale).
          · JSON 객체 '하나만', 첫 문자가 '{' (앞에 think/코드펜스/설명 금지).
        id는 넣지 않는다 — 주최측이 essay_id를 외부에서 대응. 주최측 _extract_first_json은
        문자열 속 중괄호까지 세므로, 근거에 '짝 안 맞는' '{'/'}'가 섞이면 첫 JSON 추출이
        깨져 파싱 실패(→0점)한다. 근거의 중괄호는 의미가 없으므로 전각 '｛｝'로 치환해
        어떤 생성 출력이 와도 파서가 안전하게 통과하도록 방어한다(FAQ 경고 대응).
        """
        obj = {
            t: {
                "rationale": _brace_safe((self.rationales or {}).get(t, "")),
                "score": round(float(self.scores[t]), decimals),
            }
            for t in C.TRAITS
        }
        return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 2. 저수준 IO
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 없음. AWES_ROOT가 맞는지 확인하세요 (현재: {C.ROOT})"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 3. 파싱
# ---------------------------------------------------------------------------
def _clip(x: float) -> float:
    return max(C.SCORE_MIN, min(C.SCORE_MAX, float(x)))


def _example_from_corpus_row(row: Dict, rationales=None) -> Example:
    """train_with_folds.jsonl / holdout_fold{k}.jsonl 행 → Example."""
    return Example(
        id=row["id"],
        prompt_num=row.get("prompt_num", ""),
        prompt=row["prompt"],
        essay=row["essay"],
        scores={t: _clip(row["score"][t]) for t in C.TRAITS},
        rationales=rationales,
        meta={"document_id": row.get("document_id"), "_fold": row.get("_fold")},
    )


def examples_from_jsonl(path: str) -> List[Example]:
    """코퍼스 스키마 jsonl → Example 목록.

    fold와 무관하게 임의 파일을 평가할 때 쓴다(합성본 `synth_*.jsonl` 등).
    `id / prompt / essay / score{content,organization,expression}` 만 있으면 되고
    나머지 필드(`synthetic`, `donor_id`, `shot_ids` …)는 무시된다.
    """
    return [_example_from_corpus_row(r) for r in load_jsonl(path)]


def _rationales_from_submit_row(row: Dict) -> Dict[str, str]:
    """제출 스키마 행 → {trait: rationale}."""
    return {t: row[t]["rationale"] for t in C.TRAITS}


def load_corpus(path: Optional[str] = None) -> Dict[str, Dict]:
    """id → 원본 코퍼스 행. 근거 파일과 조인할 본문 소스."""
    rows = load_jsonl(path or C.TRAIN_WITH_FOLDS)
    return {r["id"]: r for r in rows}


# ---------------------------------------------------------------------------
# 4. 공개 API — fold 단위 로딩
# ---------------------------------------------------------------------------
@dataclass
class FoldData:
    fold: int
    train: List[Example]
    eval: List[Example]

    def __repr__(self) -> str:
        return f"FoldData(fold={self.fold}, train={len(self.train)}, eval={len(self.eval)})"


def load_fold(fold: int, require_rationale: bool = True) -> FoldData:
    """
    fold-k 학습셋 + 평가셋을 로드하고 누수를 재검증한다.

    require_rationale=True (기본): 근거가 없는 학습 샘플이 있으면 에러.
      멀티태스크의 근거 생성 헤드도, autoregressive의 근거 생성 단계도 근거가 필수다.
      근거 없이 점수만으로 돌려보고 싶을 때만 False로 내린다.
    """
    corpus = load_corpus()

    # --- 평가셋: holdout_fold{k} (본문 + 정답 점수, 근거 없음) ---
    holdout_rows = load_jsonl(C.holdout_path(fold))
    eval_examples = [_example_from_corpus_row(r) for r in holdout_rows]
    holdout_ids = {r["id"] for r in holdout_rows}

    # --- 학습셋: 근거 파일(제출 스키마) × 코퍼스 본문 조인 ---
    train_rows = load_jsonl(C.fold_train_path(fold))
    train_examples, missing = [], []
    for r in train_rows:
        src = corpus.get(r["id"])
        if src is None:
            missing.append(r["id"])
            continue
        train_examples.append(
            _example_from_corpus_row(src, rationales=_rationales_from_submit_row(r))
        )

    if missing:
        raise ValueError(
            f"fold{fold}: 근거는 있으나 본문을 찾을 수 없는 id {len(missing)}건 "
            f"(예: {missing[:5]}). train_with_folds.jsonl과 근거 파일의 id 집합이 어긋납니다."
        )

    # --- hard rule 재검증 (핸드오프 3.3) ---
    leaked = holdout_ids & {e.id for e in train_examples}
    assert not leaked, (
        f"누수 발견 — fold{fold} 학습 입력에 holdout id {len(leaked)}건 포함: "
        f"{sorted(leaked)[:10]}"
    )

    if require_rationale:
        no_rat = [e.id for e in train_examples if not e.has_rationale]
        assert not no_rat, f"fold{fold}: 근거 없는 학습 샘플 {len(no_rat)}건 ({no_rat[:5]})"

    return FoldData(fold=fold, train=train_examples, eval=eval_examples)


def iter_folds(folds=None, **kwargs) -> Iterator[FoldData]:
    """5-fold 순차 순회 (GPU 1장 제약이라 병렬화 없음, 핸드오프 4.2)."""
    for k in (range(C.N_FOLDS) if folds is None else folds):
        yield load_fold(k, **kwargs)


def load_official_val() -> List[Example]:
    """
    official_val.jsonl — 최종 제출 판단 전용.

    반복 실험에 쓰면 adaptive data reuse로 오염된다 (Dwork et al. 2015, 핸드오프 1.2).
    제출 4회/72시간 제약과 직결되므로 실험 루프에서는 절대 호출하지 말 것.
    """
    return [_example_from_corpus_row(r) for r in load_jsonl(C.OFFICIAL_VAL)]


# ---------------------------------------------------------------------------
# 5. 무결성 점검 (실험 시작 전 1회)
# ---------------------------------------------------------------------------
def verify_all_folds(verbose: bool = True) -> bool:
    """5-fold 전체를 로드해 누수/조인 결손을 한 번에 확인한다."""
    ok = True
    for k in range(C.N_FOLDS):
        try:
            fd = load_fold(k)
            if verbose:
                print(f"  ✅ fold{k}: train={len(fd.train)}, eval={len(fd.eval)}")
        except (AssertionError, ValueError, FileNotFoundError) as e:
            ok = False
            print(f"  ❌ fold{k}: {e}")
    if verbose:
        print("✅ 전체 fold 무결성 확인" if ok else "❌ 무결성 실패 — 학습 시작 금지")
    return ok


if __name__ == "__main__":
    verify_all_folds()
