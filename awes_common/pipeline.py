# -*- coding: utf-8 -*-
"""
pipeline.py — 멀티태스크(VGGT식)와 autoregressive(RaDME식)가 공유하는 계약.

두 아키텍처가 다른 것은 "어떻게 학습하고 어떻게 추론하는가"뿐이다.
데이터 로딩·누수 검증·지표 계산·fold 루프·산출물 저장은 전부 동일해야
비교가 오염되지 않는다 (핸드오프 4.2: 과거 A vs B 비교가 한쪽만 파인튜닝해서
결론이 오염된 전례 있음).

구현자는 ScoringPipeline을 상속해 fit/predict 두 개만 채우면 된다:

    class MultiTaskPipeline(ScoringPipeline):
        name = "multitask"
        def fit(self, fold_data): ...
        def predict(self, examples): return [Prediction(...), ...]

    run_cv(MultiTaskPipeline, TrainConfig(...))

대칭성 규칙 — 두 아키텍처는 반드시 같은 값을 공유한다:
  백본, LoRA 설정, epoch/lr/batch, fold 분할, 평가 코드, 학습 예산.
TrainConfig가 그 공유 지점이다. 한쪽만 바꿔야 한다면 그 사실을 리포트에
명시적으로 남길 것 (자동으로 config가 산출물에 함께 저장된다).
"""

from __future__ import annotations

import abc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Type

from . import config as C
from .data import Example, FoldData, Prediction, load_fold, load_jsonl, save_jsonl
from .metrics import CVResult, EvalResult, evaluate


def set_seed(seed: int, deterministic: bool = True):
    """
    두 파이프라인이 공유하는 단일 시딩 루틴 (대칭 + 재현성).
    random/numpy/torch/cuda를 모두 같은 seed로 고정하고, DataLoader shuffle에
    쓸 generator를 반환한다. deterministic=True면 cudnn을 결정적 모드로 둔다
    (soft determinism — 안전. 완전 bitwise 결정성은 GPU matmul 특성상 보장 못 하지만,
    두 아키텍처가 동일 seed·동일 루틴을 쓰므로 잔여 비결정성은 양쪽에 동일하게 작용).

    torch는 함수 내부에서 import — 이 모듈은 torch 없는 환경(GlobalMean 테스트)에서도
    임포트돼야 하므로.
    """
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)   # DataLoader(generator=...) 용


# ---------------------------------------------------------------------------
# 1. 공유 학습 설정 — 대칭 비교의 기준선
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    backbone: str = C.BACKBONE
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    epochs: int = 3
    lr: float = 1e-4
    # 유효 배치 = batch_size × grad_accum = 16 (고정). batch_size를 키우면 H100 활용률과
    # throughput이 오른다. 단 이 모델은 vocab이 15만이라 손실용 logits(batch×seq×vocab)가
    # 메모리 병목 → batch_size 4~6이 80GB 현실적 상한. 유효 배치 16만 지키면 품질은 불변.
    # ⚠️ 두 아키텍처가 반드시 같은 값 공유. 바꾸면 fold 전체를 처음부터 다시 돌려야 대칭.
    batch_size: int = 4
    grad_accum: int = 4
    max_len: int = 2048
    # gradient checkpointing: 활성값 재계산으로 메모리↓, 연산 ~30%↑. 메모리 여유가 크면
    # False로 꺼서 속도를 얻을 수 있으나, eager attention은 O(seq²)라 끄면 batch를 크게 못 준다.
    grad_checkpointing: bool = True
    warmup_ratio: float = 0.03
    seed: int = 42
    bf16: bool = True
    # attention 구현: "eager"(안전, 느림, 현재 기본) / "sdpa"(PyTorch 내장 flash, 빠름/저메모리).
    # sdpa는 이 스택(torch2.11+tf5.14+bf16)에서 NaN을 낸 전례가 있어 기본은 eager.
    # "sdpa"로 바꿔 1 epoch 테스트해서 NaN(skipped 급증) 없으면 채택 → 학습·추론 대폭 가속.
    attn_impl: str = "eager"

    # 근거 생성 (양쪽 공통 — 멀티태스크는 생성 헤드, AR은 2단계 생성)
    gen_rationale: bool = True
    max_new_tokens: int = 512

    # epoch마다 holdout에서 점수 지표(RMSE/Spearman/std비)를 찍어 학습 곡선을 본다.
    # 과소학습(곡선 계속 개선)과 평균회귀(std비 낮음)를 학습 중에 진단하기 위함.
    # data.eval을 "측정"만 하지 backprop하지 않으므로 누수 아님(fold holdout 모니터링).
    # eval_subset>0이면 앞에서 그만큼만 써서 비용을 제한(멀티태스크는 점수만이라 저렴,
    # AR은 생성이라 캡이 필요). 최종 fold 지표는 run_cv가 전체로 다시 평가한다.
    eval_every_epoch: bool = True
    eval_subset: int = 200

    # 아키텍처 고유 하이퍼파라미터는 여기에. 대칭성 검토 시 이 필드만 보면 된다.
    arch: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["lora_targets"] = list(self.lora_targets)
        return d


# ---------------------------------------------------------------------------
# 2. 파이프라인 계약
# ---------------------------------------------------------------------------
class ScoringPipeline(abc.ABC):
    """fold 하나를 학습하고 예측하는 단위. fold마다 새 인스턴스가 생성된다."""

    name: str = "unnamed"

    def __init__(self, cfg: TrainConfig, fold: int, out_dir: str):
        self.cfg = cfg
        self.fold = fold
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    @abc.abstractmethod
    def fit(self, data: FoldData) -> None:
        """data.train으로 학습. data.eval은 절대 보지 말 것 (누수)."""

    @abc.abstractmethod
    def predict(self, examples: Sequence[Example]) -> List[Prediction]:
        """
        모든 입력에 대해 예측을 반환해야 한다 — 길이가 맞지 않으면 evaluate가 거부한다.
        생성 파싱 실패 시에는 건너뛰지 말고 fallback_prediction()으로 채울 것
        (실제 제출도 파싱 실패는 0점 처리이므로, 누락 제외는 과대추정).
        """

    # --- 선택 훅 ---
    def teardown(self) -> None:
        """GPU 메모리 해제. fold 순차 실행이라 다음 fold 전에 반드시 비워야 한다."""

    def fallback_prediction(self, ex: Example) -> Prediction:
        """파싱 실패 시 중앙값. 근거는 빈 문자열이라 judge에서 최저점을 받는다."""
        mid = (C.SCORE_MIN + C.SCORE_MAX) / 2.0
        return Prediction(id=ex.id, scores={t: mid for t in C.TRAITS},
                          rationales={t: "" for t in C.TRAITS})


# ---------------------------------------------------------------------------
# 3. 5-fold 순차 실행기
# ---------------------------------------------------------------------------
def _load_saved_predictions(fold_dir: str, eval_examples) -> Optional[List[Prediction]]:
    """
    이미 저장된 predictions.jsonl(제출 스키마)을 Prediction으로 복원한다.
    파일이 없거나, 평가셋 id를 전부 커버하지 못하면(중단으로 부분 기록된 경우)
    None을 돌려 재학습하게 한다.
    """
    path = os.path.join(fold_dir, "predictions.jsonl")
    if not os.path.exists(path):
        return None
    try:
        rows = load_jsonl(path)
    except Exception:
        return None
    by_id = {r["id"]: r for r in rows}
    if not all(e.id in by_id for e in eval_examples):   # 부분 기록 → 신뢰 불가
        return None
    preds = []
    for e in eval_examples:
        r = by_id[e.id]
        preds.append(Prediction(
            id=e.id,
            scores={t: float(r[t]["score"]) for t in C.TRAITS},
            rationales={t: r[t].get("rationale", "") for t in C.TRAITS},
        ))
    return preds


def run_cv(
    pipeline_cls: Type[ScoringPipeline],
    cfg: TrainConfig,
    folds: Optional[Sequence[int]] = None,
    run_name: Optional[str] = None,
    runs_dir: str = None,
    save_predictions: bool = True,
    check_gpu: bool = True,
    resume: bool = True,
) -> CVResult:
    """
    fold별로 학습 → 예측 → 평가 → 저장을 순차 수행한다 (GPU 1장 제약, 핸드오프 4.2).

    resume=True(기본): 이미 predictions.jsonl이 완결된 fold는 재학습을 건너뛰고
    저장된 예측을 재사용한다. 중단 후 tmux 등에서 재시작하면 끝난 fold는 스킵되고
    중단 지점부터 이어진다. 처음부터 다시 하려면 resume=False로 두거나 해당 fold
    디렉토리를 지운다.

    LLM Judge는 여기서 돌리지 않는다. judge(35B)와 학습 프로세스가 같은 GPU에
    동시에 못 올라가므로, 저장된 예측에 대해 judge.score_predictions_file()로
    사후 별도 패스를 실행한다.
    """
    if check_gpu:
        C.require_gpu_pinned()

    name = run_name or pipeline_cls.name
    base = os.path.join(runs_dir or C.RUNS_DIR, name)
    os.makedirs(base, exist_ok=True)

    with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"run": name, "pipeline": pipeline_cls.__name__,
                   "config": cfg.to_dict()}, f, ensure_ascii=False, indent=2)

    fold_list = list(range(C.N_FOLDS)) if folds is None else list(folds)
    results: List[EvalResult] = []

    for k in fold_list:
        print(f"\n{'=' * 60}\n[{name}] fold {k}\n{'=' * 60}", flush=True)
        t0 = time.time()

        data = load_fold(k)   # 누수 assert가 여기서 다시 돈다
        print(f"  train={len(data.train)}  eval={len(data.eval)}")

        fold_dir = os.path.join(base, f"fold{k}")

        # --- 재개: 완결된 예측이 있으면 재학습 스킵 ---
        preds = _load_saved_predictions(fold_dir, data.eval) if resume else None
        if preds is not None:
            res = evaluate(data.eval, preds)
            results.append(res)
            print(f"  ↻ 재개: 저장된 예측 재사용 (재학습 스킵)\n  {res.summary()}")
            continue

        pipe = pipeline_cls(cfg, fold=k, out_dir=fold_dir)
        try:
            pipe.fit(data)
            preds = pipe.predict(data.eval)
        finally:
            pipe.teardown()

        # 평가 전에 먼저 저장 — 뒤에서 죽어도 이 fold는 재개 가능하게.
        if save_predictions:
            save_jsonl([p.to_submission() for p in preds],
                       os.path.join(fold_dir, "predictions.jsonl"))
            # 시나리오 A: 주최측 파서가 읽는 '모델 원본 출력' 문자열도 남긴다.
            # (회귀헤드 점수 + 생성 근거. verify_submission.py가 이걸 파싱 검증.)
            save_jsonl([{"id": p.id, "output": p.to_model_output()} for p in preds],
                       os.path.join(fold_dir, "submission_output.jsonl"))

        res = evaluate(data.eval, preds)
        results.append(res)
        print(f"  {res.summary()}\n  ({time.time() - t0:.0f}s)")

    cv = CVResult(name=name, folds=results)
    print("\n" + cv.summary())

    with open(os.path.join(base, "cv_result.json"), "w", encoding="utf-8") as f:
        json.dump(cv.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n산출물 → {base}")
    return cv


# ---------------------------------------------------------------------------
# 4. 인터페이스 검증용 더미 — GPU 없이 전체 배선을 확인한다
# ---------------------------------------------------------------------------
class GlobalMeanPipeline(ScoringPipeline):
    """
    학습셋 영역별 평균만 예측하는 zero-signal 베이스라인.
    기대치: RMSE ≈ 0.735, Spearman = 0.00 (핸드오프 1.1)

    새 아키텍처를 붙이기 전에 이걸로 run_cv를 돌려 배선을 검증하라:
        run_cv(GlobalMeanPipeline, TrainConfig(), check_gpu=False)
    """

    name = "global_mean"

    def fit(self, data: FoldData) -> None:
        self._means = {
            t: sum(e.scores[t] for e in data.train) / len(data.train)
            for t in C.TRAITS
        }

    def predict(self, examples):
        return [
            Prediction(id=e.id, scores=dict(self._means),
                       rationales={t: "" for t in C.TRAITS})
            for e in examples
        ]
