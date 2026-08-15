# -*- coding: utf-8 -*-
"""
config.py — 경로/상수 단일 진실원(single source of truth).

경로는 환경변수로 덮어쓸 수 있다. H100 서버가 기본값이고,
맥 로컬에서 데이터 없이 임포트 테스트만 할 때는 AWES_ROOT를 지정한다.

    export AWES_ROOT=/path/to/data     # 미지정 시 저장소 루트
    export CUDA_VISIBLE_DEVICES=4                         # 2.3절 hard rule
"""

import os

# ---------------------------------------------------------------------------
# 경로 — 전부 os.environ에서 "접근 시점에" 해석한다 (지연 바인딩).
#
# 이유: import 시점에 상수로 고정하면, 패키지를 먼저 import한 뒤 프로그램이
# AWES_ROOT를 바꿔도 반영되지 않는다 (self-test가 픽스처 경로로 갈아끼우는 경우).
# 아래 module-level __getattr__(PEP 562)로 C.ROOT / C.TRAIN_WITH_FOLDS 등을
# 호출 시점마다 다시 계산한다 — 기존 호출부는 한 줄도 바꿀 필요 없다.
# (셸에서 export AWES_ROOT 후 실행하는 일반적 경우엔 애초에 문제가 없다.)
# ---------------------------------------------------------------------------
# 공개 저장소 기본값은 저장소 루트다. 데이터는 포함돼 있지 않으므로
# 실제 경로는 AWES_ROOT로 지정한다(README의 "데이터 배치" 참조).
DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _root() -> str:
    return os.environ.get("AWES_ROOT", DEFAULT_ROOT)


def data_dir() -> str:
    return os.path.join(_root(), "AWES", "data")


def rationale_dir() -> str:
    return os.path.join(_root(), "AWES", "rationales")


def fold_train_dir() -> str:
    # AWES_FOLD_TRAIN_DIR로 덮어쓸 수 있다 (distillation 데이터로 AR 학습 시 사용).
    return os.environ.get("AWES_FOLD_TRAIN_DIR", os.path.join(rationale_dir(), "fold_train"))


def runs_dir() -> str:
    return os.environ.get("AWES_RUNS", os.path.join(_root(), "runs"))


def holdout_path(fold: int) -> str:
    return os.path.join(data_dir(), f"holdout_fold{fold}.jsonl")


def fold_train_path(fold: int) -> str:
    return os.path.join(fold_train_dir(), f"rationale_train_fold{fold}.jsonl")


# C.ROOT / C.DATA_DIR / C.TRAIN_WITH_FOLDS ... 를 접근 시점에 해석
_LAZY = {
    "ROOT": _root,
    "DATA_DIR": data_dir,
    "RATIONALE_DIR": rationale_dir,
    "FOLD_TRAIN_DIR": fold_train_dir,
    "RUNS_DIR": runs_dir,
    "TRAIN_WITH_FOLDS": lambda: os.path.join(data_dir(), "train_with_folds.jsonl"),
    "OFFICIAL_VAL": lambda: os.path.join(data_dir(), "official_val.jsonl"),
}


def __getattr__(name):  # PEP 562
    if name in _LAZY:
        return _LAZY[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# 과제 상수
# ---------------------------------------------------------------------------
TRAITS = ("content", "organization", "expression")
TRAIT_KOR = {"content": "내용", "organization": "구성", "expression": "표현"}

# fold 수. 4 GPU에 정확히 1 웨이브로 맞추기 위해 4분할 사용 (env로 재정의 가능).
# 바꾸면 make_folds.py로 holdout/rationale fold 파일을 반드시 재생성해야 한다.
N_FOLDS = int(os.environ.get("AWES_NFOLDS", "4"))
SCORE_MIN, SCORE_MAX = 1.0, 5.0

# 공식 평가 가중치 (RMSE 45% + Spearman 45% + LLM Judge 10%)
W_RMSE, W_SPEARMAN, W_JUDGE = 0.45, 0.45, 0.10

# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
BACKBONE = os.environ.get("AWES_BACKBONE", "Qwen/Qwen2.5-7B-Instruct")
JUDGE_MODEL = os.environ.get("AWES_JUDGE_MODEL", "Qwen/Qwen3.6-35B-A3B")

# 제출 제약: L40s 48GB, 14B 이하 — 백본 교체 시 반드시 재확인
MAX_PARAMS_B = 14.0


def require_gpu_pinned():
    """CUDA_VISIBLE_DEVICES 미지정 시 즉시 중단 (기본 GPU 0은 점유 중, 2.3절)."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not dev:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES가 지정되지 않았습니다. GPU 0은 다른 프로세스가 점유 중입니다.\n"
            "  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv\n"
            "로 빈 GPU를 확인한 뒤 CUDA_VISIBLE_DEVICES=<번호>를 지정하세요."
        )
    return dev
