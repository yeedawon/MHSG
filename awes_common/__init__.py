# -*- coding: utf-8 -*-
"""
awes_common — 멀티태스크(VGGT식) / autoregressive(RaDME식) 공통 인터페이스.

    from awes_common import ScoringPipeline, TrainConfig, run_cv, Prediction

두 아키텍처는 fit/predict만 구현하고, 데이터·평가·fold 루프는 공유한다.
"""

from .config import TRAITS, TRAIT_KOR, N_FOLDS
from .data import (
    Example, Prediction, FoldData,
    load_fold, iter_folds, load_official_val, verify_all_folds,
    load_jsonl, save_jsonl,
)
from .metrics import (
    EvalResult, CVResult, TraitMetrics,
    evaluate, rmse, spearman, global_mean_predictions,
)
from .pipeline import ScoringPipeline, TrainConfig, run_cv, GlobalMeanPipeline

# 멀티태스크·autoregressive 파이프라인은 torch/transformers/peft 에 의존하지만,
# 모듈 자체는 지연 임포트라 맥에서도 임포트된다 (fit/predict 호출 시에만 무거운 의존성 필요).
from .multitask import MultiTaskPipeline
from .autoregressive import (
    AutoregressivePipeline, build_scoring_messages, build_target_json,
    parse_scoring_output,
)

__all__ = [
    "TRAITS", "TRAIT_KOR", "N_FOLDS",
    "Example", "Prediction", "FoldData",
    "load_fold", "iter_folds", "load_official_val", "verify_all_folds",
    "load_jsonl", "save_jsonl",
    "EvalResult", "CVResult", "TraitMetrics",
    "evaluate", "rmse", "spearman", "global_mean_predictions",
    "ScoringPipeline", "TrainConfig", "run_cv", "GlobalMeanPipeline",
    "MultiTaskPipeline",
    "AutoregressivePipeline", "build_scoring_messages", "build_target_json",
    "parse_scoring_output",
]
