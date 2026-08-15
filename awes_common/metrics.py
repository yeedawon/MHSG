# -*- coding: utf-8 -*-
"""
metrics.py — 공식 평가지표 (RMSE 45% + Spearman 45% + LLM Judge 10%).

두 아키텍처가 같은 숫자로 비교되도록 계산을 여기 한 곳에 고정한다.

주의 — Spearman은 영역별로 따로 계산한 뒤 평균낸다. 3영역 점수를 한 벡터로
이어붙여 계산하면 영역 간 난이도 차이가 순위상관을 부풀린다 (organization이
전 fold에서 구조적으로 가장 어렵다는 관측이 묻힘, 핸드오프 1.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import config as C
from .data import Example, Prediction


# ---------------------------------------------------------------------------
# 1. 기본 통계 (numpy/scipy 없이 동작 — 서버 venv 의존성 최소화)
# ---------------------------------------------------------------------------
def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n)


def _rankdata(xs: Sequence[float]) -> List[float]:
    """동점은 평균 순위 (scipy.stats.rankdata의 'average'와 동일)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    동점 보정 포함 Spearman ρ (= 순위에 대한 Pearson 상관).

    예측이 전부 같은 값이면 분산이 0이라 정의되지 않는다. 이때 0.0을 반환한다
    — GlobalMean 베이스라인의 zero-signal floor와 같은 취급 (핸드오프 1.1).
    """
    n = len(y_true)
    if n < 2:
        return float("nan")
    rt, rp = _rankdata(y_true), _rankdata(y_pred)
    mt, mp = sum(rt) / n, sum(rp) / n
    num = sum((a - mt) * (b - mp) for a, b in zip(rt, rp))
    den = math.sqrt(sum((a - mt) ** 2 for a in rt) * sum((b - mp) ** 2 for b in rp))
    return 0.0 if den == 0 else num / den


# ---------------------------------------------------------------------------
# 2. 결과 컨테이너
# ---------------------------------------------------------------------------
@dataclass
class TraitMetrics:
    trait: str
    rmse: float
    spearman: float
    n: int


@dataclass
class EvalResult:
    per_trait: Dict[str, TraitMetrics]
    rmse: float                       # 3영역 매크로 평균
    spearman: float
    judge: Optional[float] = None     # 1~5 척도, 미측정 시 None
    n: int = 0
    extra: Dict = field(default_factory=dict)

    @property
    def official_score(self) -> Optional[float]:
        """
        공식 종합점수 근사 (0~1, 높을수록 좋음).

        주최 측 정규화 공식이 공개되지 않았으므로 아래 변환을 사용한다:
          - RMSE  → 1 - RMSE/4   (4 = 1~5 척도의 최대 오차)
          - ρ     → (ρ+1)/2
          - judge → (judge-1)/4
        절대값 자체보다 "두 아키텍처 간 상대 비교"에 쓰라. judge 미측정이면 None.
        """
        if self.judge is None:
            return None
        return (
            C.W_RMSE * (1.0 - self.rmse / 4.0)
            + C.W_SPEARMAN * (self.spearman + 1.0) / 2.0
            + C.W_JUDGE * (self.judge - 1.0) / 4.0
        )

    @property
    def score_only(self) -> float:
        """judge 없이 점수 파트(90%)만 정규화한 값. 실험 루프의 주 지표."""
        return (
            C.W_RMSE * (1.0 - self.rmse / 4.0)
            + C.W_SPEARMAN * (self.spearman + 1.0) / 2.0
        ) / (C.W_RMSE + C.W_SPEARMAN)

    def summary(self) -> str:
        lines = [
            f"n={self.n}  RMSE={self.rmse:.4f}  Spearman={self.spearman:.4f}"
            + (f"  Judge={self.judge:.3f}" if self.judge is not None else "")
        ]
        for t in C.TRAITS:
            m = self.per_trait[t]
            lines.append(
                f"    {C.TRAIT_KOR[t]:2s}({t:<12s}) RMSE={m.rmse:.4f}  ρ={m.spearman:.4f}"
            )
        official = self.official_score
        lines.append(
            f"  score_only(90%)={self.score_only:.4f}"
            + (f"  official≈{official:.4f}" if official is not None else "")
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. 평가 진입점
# ---------------------------------------------------------------------------
def evaluate(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    judge: Optional[float] = None,
) -> EvalResult:
    """
    정답(examples)과 예측(predictions)을 id로 정렬해 지표를 계산한다.

    예측 누락은 조용히 넘기지 않는다 — JSON 파싱 실패 시 0점 처리되는 과제
    특성상, 누락을 평균으로 메우면 실제 제출 점수를 과대추정하게 된다.
    """
    pred_by_id = {p.id: p for p in predictions}
    missing = [e.id for e in examples if e.id not in pred_by_id]
    if missing:
        raise ValueError(
            f"예측 누락 {len(missing)}건 (예: {missing[:5]}). "
            "파싱 실패 샘플은 fallback 점수를 명시적으로 채워 넣고 평가하세요 "
            "— 누락 제외는 실제 점수를 과대추정합니다."
        )

    per_trait = {}
    for t in C.TRAITS:
        yt = [e.scores[t] for e in examples]
        yp = [float(pred_by_id[e.id].scores[t]) for e in examples]
        per_trait[t] = TraitMetrics(t, rmse(yt, yp), spearman(yt, yp), len(yt))

    return EvalResult(
        per_trait=per_trait,
        rmse=sum(m.rmse for m in per_trait.values()) / len(C.TRAITS),
        spearman=sum(m.spearman for m in per_trait.values()) / len(C.TRAITS),
        judge=judge,
        n=len(examples),
    )


# ---------------------------------------------------------------------------
# 4. fold 집계
# ---------------------------------------------------------------------------
@dataclass
class CVResult:
    name: str
    folds: List[EvalResult]

    def _agg(self, attr: str):
        vals = [getattr(f, attr) for f in self.folds]
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        if not vals:
            return float("nan"), float("nan")
        mean = sum(vals) / len(vals)
        if len(vals) < 2:
            return mean, 0.0
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
        return mean, std

    def summary(self) -> str:
        rm, rs = self._agg("rmse")
        sm, ss = self._agg("spearman")
        lines = [
            f"=== {self.name} — {len(self.folds)}-fold CV ===",
            f"RMSE      {rm:.4f} ± {rs:.4f}",
            f"Spearman  {sm:.4f} ± {ss:.4f}",
        ]
        jm, js = self._agg("judge")
        if not math.isnan(jm):
            lines.append(f"Judge     {jm:.3f} ± {js:.3f}")
        for t in C.TRAITS:
            tr = [f.per_trait[t].rmse for f in self.folds]
            ts = [f.per_trait[t].spearman for f in self.folds]
            lines.append(
                f"  {C.TRAIT_KOR[t]:2s} RMSE={sum(tr)/len(tr):.4f}  ρ={sum(ts)/len(ts):.4f}"
            )
        for i, f in enumerate(self.folds):
            lines.append(f"  fold{i}: RMSE={f.rmse:.4f}  ρ={f.spearman:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        rm, rs = self._agg("rmse")
        sm, ss = self._agg("spearman")
        return {
            "name": self.name,
            "rmse_mean": rm, "rmse_std": rs,
            "spearman_mean": sm, "spearman_std": ss,
            "folds": [
                {
                    "rmse": f.rmse, "spearman": f.spearman, "judge": f.judge, "n": f.n,
                    "per_trait": {
                        t: {"rmse": m.rmse, "spearman": m.spearman}
                        for t, m in f.per_trait.items()
                    },
                }
                for f in self.folds
            ],
        }


# ---------------------------------------------------------------------------
# 5. 베이스라인 — 새 아키텍처가 이걸 못 넘으면 신호가 없는 것
# ---------------------------------------------------------------------------
def global_mean_predictions(train, eval_examples) -> List[Prediction]:
    """
    학습셋 영역별 평균으로 전부 예측 (zero-signal floor).
    기대치: RMSE ≈ 0.735, Spearman = 0.00 (핸드오프 1.1)
    """
    means = {
        t: sum(e.scores[t] for e in train) / len(train) for t in C.TRAITS
    }
    return [Prediction(id=e.id, scores=dict(means)) for e in eval_examples]
