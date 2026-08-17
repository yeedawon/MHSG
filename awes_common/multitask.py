# -*- coding: utf-8 -*-
"""
multitask.py — 멀티태스크(VGGT식) 채점 파이프라인.

공유 백본(LoRA) 하나 위에 두 개의 머리를 얹는다:
  - 회귀 헤드   : 마지막 입력 토큰의 은닉표현 → content/organization/expression 3점
  - 생성 헤드   : 백본의 LM 헤드로 근거(rationale) JSON을 생성

핵심 설계 (CLAUDE2.md 아키텍처 결정, awes_common/README.md 대칭성 규칙):

1) 한 forward에서 점수와 근거를 함께 학습한다. 손실은
      L = 1/(2σ_r²)·L_reg + 1/(2σ_g²)·L_gen + log σ_r + log σ_g
   Kendall & Gal(2018)의 homoscedastic uncertainty weighting. σ는 학습 가능한
   log-variance 파라미터라, 두 손실의 스케일 차이를 사람이 손으로 맞추지 않는다.
   (분리형에서 λ를 손으로 튜닝하던 문제를 없애는 것이 멀티태스크 채택 이유 중 하나)

2) 회귀 헤드가 읽는 위치는 "입력의 마지막 토큰"이다. 근거 토큰이 아니다.
   추론 시엔 근거가 아직 없으므로, 학습 때도 근거를 보지 않는 위치에서 점수를
   뽑아야 학습/추론 분포가 일치한다 (causal mask 덕분에 그 위치는 입력만 attend).

3) 점수는 [0,1]로 정규화해 sigmoid로 회귀하고, 추론 때 [1,5]로 되돌린다.
   척도를 벗어난 예측을 구조적으로 막아 RMSE 폭주를 방지한다.

대칭성: 백본/LoRA/epoch/lr/batch/max_len/seed는 전부 TrainConfig에서 온다. AR
파이프라인과 반드시 같은 값을 써야 "아키텍처 차이"만 비교된다. 멀티태스크 고유
하이퍼파라미터(회귀 헤드 크기, 근거 손실 초기 가중 등)는 cfg.arch에 둔다.

torch/transformers/peft는 함수 안에서 지연 임포트한다 — GPU 없는 맥에서도
`from awes_common import MultiTaskPipeline` 가 되도록 (judge.py의 vLLM과 동일 규약).
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence

LOG_EVERY = 25  # 배치 N개마다 진행률 출력

from . import config as C
from .data import Example, FoldData, Prediction, save_jsonl
from .pipeline import ScoringPipeline, TrainConfig


# ---------------------------------------------------------------------------
# 0. 아키텍처 고유 기본값 (cfg.arch 로 덮어씀)
# ---------------------------------------------------------------------------
ARCH_DEFAULTS = {
    "reg_hidden": 512,          # 회귀 헤드 MLP 은닉폭
    "reg_dropout": 0.1,
    "init_log_var_reg": 0.0,    # σ_r 초기값 (exp(0)=1)
    "init_log_var_gen": 0.0,    # σ_g 초기값
    "gen_temperature": 0.3,     # 근거 생성 온도 (teacher 합성과 동일 대역)
    "gen_top_p": 0.9,
    "gen_batch_size": 8,        # 추론 배치 (AR과 동일). 한 개씩 생성하면 GPU를 못 채워 느림.
    "rank_weight": 0.0,         # 순위 손실 가중 (0=off). Spearman 특화, RMSE엔 역효과 가능.
    "rank_margin": 0.05,        # pairwise margin (정규화 [0,1] 스케일 기준)
    # --- 소격차 쌍 집중 (중간구간 병목 대응) ---
    # 배치 내 쌍의 대부분은 격차가 커서 이미 맞고, hinge가 0이 된다. 그런데 분모
    # (mask.sum())에는 계속 잡혀 정작 훈련이 필요한 소격차 쌍의 gradient가 희석된다.
    # rank_hard_gap>0이면 |Δgold| ≤ 값 인 쌍만 손실에 넣어 병목 해상도에 집중한다.
    # 단위는 정규화 [0,1] (= 원점수/4). 0.05 → 원점수 0.2점. 0=off(전체 쌍, 현행).
    "rank_hard_gap": 0.0,
    # margin 해석. "fixed"(현행)는 실제 격차보다 큰 분리를 요구할 수 있어, 소격차 쌍만
    # 남기면 모순 신호가 된다(0.05점 차 쌍에 0.2점 분리 요구). "gap"은 margin을
    # min(rank_margin, |Δgold|)로 낮춰 "정답 격차만큼만 벌려라"로 바꾼다.
    "rank_margin_mode": "fixed",  # "fixed" | "gap"
    # --- 합성 대조쌍 주입 (HANDOFF §8, awes_common/contrastive.py) ---
    # contrastive_dir 지정 시에만 켜진다. 합성본은 labels=None 경로로만 forward돼
    # MSE/gen에 절대 닿지 않는다(계약 1). 손실은 열화 영역 차원에만 margin*gap hinge.
    "contrastive_dir": None,        # make_contrastive_pairs build --out 경로. None=off
    "contrastive_weight": 0.0,      # 대조 손실 가중. 0이면 로드해도 backward 안 함
    "contrastive_margin": 0.05,     # 정규화 [0,1] 스케일 margin (gap배로 확대됨)
    "contrastive_batch_pairs": 16,  # optim step당 forward할 쌍 수 (고유 변형 ≤2배)
    "contrastive_max_gap": None,    # gap 상한 필터(예: 2면 원본>강만 있는 3-gap 제외). None=전체
    "contrastive_traits": None,     # 특정 영역만 사용(예: ["expression"]). None=전체
    "corr_weight": 0.0,         # 상관 손실 가중 (0=off). RMSE·Spearman 동시 겨냥. 0.1~0.5 권장.
    "soft_weight": 0.0,         # soft-Spearman 손실 가중 (0=off). 지표 직접 최적화. 0.1~0.5 권장.
    "soft_tau": 0.1,            # soft rank 온도. 작을수록 실제 순위에 근접(gradient는 날카로움).
    # 생성 항 불확실성 계수 (핸드오프 §1): regression(현재)/kendall_cls/classification.
    "gen_uncertainty": "regression",
    # 시나리오 B 대응 — 생성 헤드가 점수까지 JSON으로 출력(제출 파싱 형식).
    # True면 predict가 생성 JSON에서 점수를 읽는다(회귀헤드는 보조로 학습만).
    "gen_scores": False,
    "score_decimals": 2,
    # 회귀 헤드가 읽는 표현 풀링 (실험: 마지막토큰이 순위 신호 병목일 수 있음).
    #   "last"(현재/기본) / "mean"(마스크 평균) / "attention"(학습형 가중합)
    "pooling": "last",
    # 생성 CE를 라벨 위치에서만 계산 (기본 off — 기존 run 재현성 보존).
    # HF 내부 CE는 전체 위치에 [B,T,V] logits를 fp32로 만들어 backward까지 ~40GB를
    # 쓴다(vocab ~20만). 라벨 위치는 전체의 ~17%뿐이라, 그 행만 lm_head를 적용하면
    # 수학적으로 동일한 손실을 ~3GB로 계산한다. 긴 입력(앵커 등)·대배치의 해금 열쇠.
    "sparse_ce": False,
    # 생성 손실 완전 off (외부 데이터 사전학습용 — 근거 라벨이 없는 코퍼스).
    # True면 labels를 백본에 넘기지 않고 gen_loss=0, 불확실성 가중의 gen 항도 제외.
    "gen_off": False,
    # 회귀 손실 완전 off (2026-08-17, 논문 트랙). gen_off의 대칭 — 생성만 학습하는
    # "근거 전용 어댑터"를 만든다. 분업 구조(점수 어댑터 + 근거 어댑터, 백본 공유)의
    # 근거 절반이다. ⚠️ 이 arm의 predictions.jsonl 점수는 학습되지 않은 헤드의 출력이라
    # 의미가 없다 — 채점 지표로 읽지 말 것.
    "reg_off": False,
    # 사전학습 산출물 디렉토리(adapter/ + heads.pt 포함)에서 LoRA·헤드를 초기화.
    # None이면 기존처럼 새 LoRA. Stage A(외부 45k) → Stage B(AWES 파인튜닝) 연결 고리.
    "init_from": None,
    # init_from 시 회귀 헤드까지 웜스타트할지. False면 **LoRA만** 이어받고 헤드는 새로 시작.
    # 근거: 외부 사전학습 헤드가 시그모이드 포화 구간(예측 μ4.80/σ0.09 vs gold σ0.67)에
    # 갇혀 순서 정보를 상수로 뭉개는 현상을 관측했다(2026-07-27). 표현(LoRA)은 ρ0.54로
    # 쓸 만하므로, 헤드만 새로 학습시켜 AWES 스케일을 처음부터 잡게 하는 경로.
    "init_head": True,
    # 생성 항 불확실성 계수 형태 (§1): "regression"(0.5,0.5) / "kendall_cls"(1.0,0.5)
    "gen_uncertainty": "regression",
    # 하위 준거 감독 (2026-07-28). gold 생성 과정이 "채점자 2인 × 준거 9개
    # (con1~5/org1~2/exp1~2) 평균"임을 역산으로 확정(make_nikl_data.py)한 데 따른
    # 구조 반영: 헤드가 trait 3개 대신 준거 9개를 회귀하고 trait은 소속 준거의
    # 평균으로 결정적으로 집계한다. 평균은 선형이라 [1,5]→[0,1] 정규화와 가환.
    # AWES 샘플은 집계된 trait으로(기존 손실 그대로), 준거 라벨이 있는 샘플
    # (NIKL 혼합분, Example.meta["subcrit"])만 준거 MSE가 추가된다.
    "subcrit": False,
    "subcrit_weight": 1.0,      # 준거 MSE 가중 (trait MSE와 같은 1/2σ² 안에서)
    # 영역별 회귀 손실 가중 (C.TRAITS 순서: content/organization/expression).
    # None이면 균등(=기존 동작). 근거(2026-07-30 ceiling_eda 실측):
    #   라벨 천장이 organization 0.830 > content 0.792 > expression 0.764 인데
    #   우리 달성률은 org 80.6% / con 85.8% / exp 92.1% 로 **역순**이다.
    #   즉 더 짜낼 게 없는 expression(남은 여지 0.060)에 organization(0.161)과
    #   같은 용량을 쓰고 있다. 여지에 비례해 재배분하면 3영역 평균이 오를 여지가 있다.
    # ⚠️ 가중 합이 아니라 **평균으로 정규화**하므로 전체 손실 스케일은 불변 —
    #    log_var·corr·gen 등 다른 항과의 균형이 흔들리지 않는다.
    "trait_weights": None,
    # 총점(3영역 평균) 일관성 손실 가중. 0=off(기존 동작).
    # 왜 (2026-07-30): **공식 지표는 3영역 평균점수 위에서 RMSE/ρ를 계산**하는데
    # 우리 손실은 영역별 MSE 합이라 지표와 정렬돼 있지 않다. 그래서 세 헤드가
    # "각자 맞히기"만 학습하고 오차가 서로 상쇄될 유인이 없다 — 실측 영역 간 오차
    # 상관 0.45~0.53, 평균오차 σ 0.4427(독립이면 0.277).
    # VGGT가 4개 헤드를 분리하지 않고 태스크 간 일관성 제약으로 공유 표현을 밀어붙여
    # SOTA를 낸 구조를 우리 문제로 옮긴 것: 헤드는 공유하되 **총점 항을 추가**해
    # 교차공분산을 직접 겨냥한다. (헤드 분리는 2026-07-30 실측에서 -0.0131로 기각됐고
    # 오히려 상관이 0.533→0.572로 올랐다 — 원인이 헤드가 아니라 공유 pooled였다.)
    "total_weight": 0.0,
    # 평균점수(3영역 평균)에 대한 pairwise rank 손실. 0=off.
    # 왜 (2026-07-31, 병행 세션 카드15 제안): 공식 지표는 RMSE 45% + Spearman 45%를
    # 둘 다 평균점수 위에서 계산하는데, total_weight(MSE)는 RMSE 쪽만 겨냥한다.
    # Spearman-on-평균점수는 이 항이 생기기 전까지 아무 손실도 안 건드리고 있었다.
    # 구현은 기존 _rank_loss를 [B,3] 대신 [B,1](평균)에 그대로 재사용 — pairwise
    # unsqueeze 브로드캐스트가 마지막 차원 크기에 무관해 별도 함수가 필요 없다.
    "total_rank_weight": 0.0,
}

CRITERIA = {
    "content": "주제 이해도, 주장의 타당성, 근거의 충실성, 논증의 설득력",
    "organization": "글의 구조적 완결성, 문단 간 논리적 연결, 서론-본론-결론의 짜임새",
    "expression": "어휘의 적절성, 문장의 정확성과 유창성, 표현의 명료성",
}

# 하위 준거 구성(subcrit) — NIKL 채점 도구의 준거 수. trait = 소속 준거 평균.
# 평탄화 순서는 C.TRAITS 순서: content 5(con1~5) + organization 2 + expression 2.
SUBCRIT_N = {"content": 5, "organization": 2, "expression": 2}
SUB_TOTAL = sum(SUBCRIT_N.values())  # 9


# ---------------------------------------------------------------------------
# 1. 프롬프트 / 타깃 (torch 불필요 — 자체 테스트 대상)
# ---------------------------------------------------------------------------
SCORING_SYSTEM = (
    "너는 한국어 논증적 글을 채점하는 국어 교육 전문가이다. "
    "주어진 글을 content(내용)/organization(구성)/expression(표현) 세 영역에서 "
    "평가하고, 각 영역 점수의 근거를 글의 구체적 특징에 기대어 서술하라."
)


def build_input_text(ex: Example) -> str:
    """백본에 넣을 입력(프롬프트 part). 근거는 포함하지 않는다 — 여기까지가 회귀 위치."""
    parts = [
        SCORING_SYSTEM, "",
        "[글의 주제]", ex.prompt.strip(), "",
        "[학생 글]", ex.essay.strip(), "",
        "[채점 기준]",
    ]
    for t in C.TRAITS:
        parts.append(f"  - {C.TRAIT_KOR[t]}({t}): {CRITERIA[t]}")
    parts += [
        "",
        "위 글을 채점하고, 각 영역 점수의 근거를 아래 JSON 형식으로만 출력하라:",
    ]
    return "\n".join(parts)


def build_target_text(ex: Example, gen_scores: bool = False, decimals: int = 2,
                      score_override=None) -> str:
    """
    생성 헤드가 배울 타깃 JSON.

    gen_scores=False (기본): 근거만 {trait: rationale}. (제출엔 점수가 안 실림)
    gen_scores=True (시나리오 B): 근거+점수 전체 {trait: {rationale, score}}.
      → 제출 시 주최측이 파싱하는 형식과 동일. 점수를 텍스트로 생성.
      키 순서 rationale→score로 CoT(근거 먼저) 강제.
    score_override: {trait: float} 주면 그 점수를 타깃으로 씀 (distillation 2단계용;
      회귀헤드 예측을 타깃으로). None이면 정답 점수(gold) 사용.
    """
    rats = ex.rationales or {}
    if not gen_scores:
        return json.dumps({t: rats.get(t, "") for t in C.TRAITS}, ensure_ascii=False)
    src = score_override if score_override is not None else ex.scores
    obj = {t: {"rationale": rats.get(t, "").strip(),
               "score": round(float(src[t]), decimals)} for t in C.TRAITS}
    return json.dumps(obj, ensure_ascii=False)


def parse_rationale_json(text: str) -> Optional[Dict[str, str]]:
    """
    생성된 텍스트에서 근거 3개를 추출. 실패 시 None (fallback 대상).
    judge.py / rationale_synthesis_utils 의 파서와 같은 방어 전략:
      thinking 블록 제거 → 코드펜스 제거 → 첫 균형 중괄호 추출 → 후행 콤마 제거.
    """
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        return None
    depth, blob = 0, None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                break
    if blob is None:
        return None
    try:
        data = json.loads(re.sub(r",\s*([}\]])", r"\1", blob))
    except json.JSONDecodeError:
        return None
    out = {}
    for t in C.TRAITS:
        v = data.get(t)
        if not isinstance(v, str) or not v.strip():
            return None
        out[t] = v.strip()
    return out


# 점수 [1,5] <-> 정규화 [0,1]
def _norm_score(s: float) -> float:
    return (float(s) - C.SCORE_MIN) / (C.SCORE_MAX - C.SCORE_MIN)


def _denorm_score(u: float) -> float:
    s = C.SCORE_MIN + float(u) * (C.SCORE_MAX - C.SCORE_MIN)
    return max(C.SCORE_MIN, min(C.SCORE_MAX, s))


# ---------------------------------------------------------------------------
# 2. torch 의존부 — 지연 임포트 팩토리
# ---------------------------------------------------------------------------
def _build_torch_parts():
    """
    torch/nn 이 필요한 클래스(회귀헤드 결합 모델, 데이터셋, 콜레이터)를 만들어 반환.
    모듈 임포트 시점이 아니라 fit() 안에서만 호출된다.
    """
    import torch
    import torch.nn as nn

    class MultiTaskModel(nn.Module):
        """base causal LM(LoRA 적용) + 회귀 헤드 + 학습가능 log-variance 2개."""

        def __init__(self, base_lm, hidden_size: int, arch: Dict):
            super().__init__()
            self.base = base_lm  # AutoModelForCausalLM (+ LoRA), LM 손실/헤드 포함
            h = arch["reg_hidden"]
            # subcrit=True면 출력이 준거 9개, 아니면 trait 3개 (완전 하위호환)
            self.subcrit = bool(arch.get("subcrit", False))
            self.subcrit_weight = float(arch.get("subcrit_weight", 1.0))
            if self.subcrit:
                self._sub_slices, s = [], 0
                for t in C.TRAITS:
                    self._sub_slices.append((s, s + SUBCRIT_N[t]))
                    s += SUBCRIT_N[t]
                out_dim = s                      # = SUB_TOTAL
            else:
                out_dim = len(C.TRAITS)
            # split_heads=True면 영역마다 **독립 MLP**를 둔다(표현 공유 차단).
            # 근거(2026-07-30 실측): 영역 간 오차 상관 0.47~0.59로 높아, 평균오차 σ가
            # 0.442다. 독립이면 0.277이므로 상관이 60%를 물리고 있다. 원인은 세 영역이
            # 같은 512차원 은닉을 공유해 그 표현의 오차가 세 출력에 그대로 복제되는 구조.
            # 공식 지표가 3영역 **평균점수** 위에서 계산되므로 이 교차공분산이 직접 손해다.
            self.split_heads = bool(arch.get("split_heads", False)) and not self.subcrit
            self.total_weight = float(arch.get("total_weight", 0.0))
            self.total_rank_weight = float(arch.get("total_rank_weight", 0.0))
            if self.split_heads:
                self.reg_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_size, h),
                        nn.GELU(),
                        nn.Dropout(arch["reg_dropout"]),
                        nn.Linear(h, 1),
                    ) for _ in C.TRAITS
                ])
                self.reg_head = None
            else:
                self.reg_heads = None
                self.reg_head = nn.Sequential(
                    nn.Linear(hidden_size, h),
                    nn.GELU(),
                    nn.Dropout(arch["reg_dropout"]),
                    nn.Linear(h, out_dim),
                )
            # 영역별 회귀 가중 (없으면 None → 기존 균등 MSE 경로 그대로)
            tw = arch.get("trait_weights")
            if tw:
                assert len(tw) == len(C.TRAITS), f"trait_weights는 {len(C.TRAITS)}개여야 함"
                w = torch.tensor([float(x) for x in tw], dtype=torch.float32)
                self.register_buffer("trait_w", w / w.mean())   # 평균 1로 정규화
            else:
                self.trait_w = None
            # Kendall homoscedastic uncertainty — log σ² 를 직접 학습
            self.log_var_reg = nn.Parameter(torch.tensor(float(arch["init_log_var_reg"])))
            self.log_var_gen = nn.Parameter(torch.tensor(float(arch["init_log_var_gen"])))
            # 순위 손실(평균회귀 대응) — 기본 0(off). >0이면 MSE에 pairwise ranking을 더한다.
            self.rank_weight = float(arch.get("rank_weight", 0.0))
            self.rank_margin = float(arch.get("rank_margin", 0.05))
            self.rank_hard_gap = float(arch.get("rank_hard_gap", 0.0))
            self.rank_margin_mode = str(arch.get("rank_margin_mode", "fixed"))
            self.sparse_ce = bool(arch.get("sparse_ce", False))
            self.gen_off = bool(arch.get("gen_off", False))
            self.reg_off = bool(arch.get("reg_off", False))
            if self.gen_off and self.reg_off:
                raise ValueError("gen_off와 reg_off를 동시에 켜면 학습할 손실이 없다.")
            self._ltk_ok = None   # base가 logits_to_keep을 받는지 (첫 호출에서 판정)
            # 상관 손실 — 배치 Pearson을 최대화. RMSE·Spearman을 동시에 겨냥(순위 손실보다
            # RMSE 친화적). 대회가 RMSE45%+Spearman45%라 이쪽이 더 적합. 기본 0(off).
            self.corr_weight = float(arch.get("corr_weight", 0.0))
            # soft-Spearman 손실 — 미분가능 순위로 지표(Spearman)를 직접 최적화.
            # pairwise hinge(rank)가 '근사의 근사'인 데 반해 순위상관 자체를 목적함수로
            # 삼는다. 손실 계열이 rank/corr 어느 쪽과도 달라 앙상블 멤버 다양성에도 기여
            # 기대. 기본 0(off). soft_tau는 순위 근사의 날카로움(작을수록 실제 순위에 근접).
            self.soft_weight = float(arch.get("soft_weight", 0.0))
            self.soft_tau = float(arch.get("soft_tau", 0.1))
            # 불확실성 가중 형태 (핸드오프 2026-07-22 §1) — 생성 항 계수 재유도.
            #   회귀(가우시안): 1/(2σ²)L + (1/2)logσ²  → (0.5, 0.5)
            #   분류/생성(Kendall softmax): (1/σ²)CE + logσ = (1.0, 0.5)  [엄밀]
            #   핸드오프 제안(1/2 둘 다 제거):           = (1.0, 1.0)
            # 회귀 항은 항상 (0.5,0.5). 생성 항만 아래 계수로 A/B 테스트한다. 기본=현재값.
            _GEN_COEF = {"regression": (0.5, 0.5),   # 현재 (기본)
                         "kendall_cls": (1.0, 0.5),  # Kendall 분류 엄밀형
                         "classification": (1.0, 1.0)}  # 핸드오프 §1 제안형
            self.gen_c_loss, self.gen_c_log = _GEN_COEF.get(
                arch.get("gen_uncertainty", "regression"), (0.5, 0.5))
            # 풀링 방식 — last(마지막토큰)/mean(마스크평균)/attention(학습형).
            self.pooling = arch.get("pooling", "last")
            if self.pooling == "attention":
                # 토큰별 스칼라 점수 → 마스크 softmax → 가중합. fp32(안정), trainable
                # (optim이 requires_grad로 자동 수집).
                self.attn_pool = nn.Linear(hidden_size, 1)

        def head_forward(self, pf):
            """공유 헤드/영역별 독립 헤드를 한 인터페이스로 — [B, out] (sigmoid 전)."""
            if self.split_heads:
                return torch.cat([hd(pf) for hd in self.reg_heads], dim=1)
            return self.reg_head(pf)

        def head_modules(self):
            """dtype 캐스팅·state_dict 등에 쓸 실제 헤드 모듈 (분리 시 ModuleList)."""
            return self.reg_heads if self.split_heads else self.reg_head

        def _corr_loss(self, pred, target):
            """1 - 영역별 배치 Pearson 평균. std=0(초기 뭉침) 방어로 eps 추가."""
            eps = 1e-6
            p = pred - pred.mean(dim=0, keepdim=True)
            t = target - target.mean(dim=0, keepdim=True)
            num = (p * t).sum(dim=0)
            den = torch.sqrt((p * p).sum(dim=0) * (t * t).sum(dim=0) + eps)
            r = num / (den + eps)                    # [3] 영역별 상관
            return (1.0 - r).mean()

        def _pooled(self, hidden_last, pool_pos, attention_mask=None):
            # hidden_last: [B,T,H], pool_pos: [B] (입력 마지막 실토큰 위치)
            if self.pooling == "last" or attention_mask is None:
                idx = pool_pos.view(-1, 1, 1).expand(-1, 1, hidden_last.size(-1))
                return hidden_last.gather(1, idx).squeeze(1)  # [B,H]
            m = attention_mask.unsqueeze(-1).to(hidden_last.dtype)      # [B,T,1]
            if self.pooling == "mean":
                return (hidden_last * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            if self.pooling == "attention":
                sc = self.attn_pool(hidden_last.float()).squeeze(-1)    # [B,T] fp32
                sc = sc.masked_fill(attention_mask == 0, float("-inf"))
                w = torch.softmax(sc, dim=1).unsqueeze(-1).to(hidden_last.dtype)
                return (hidden_last * w).sum(dim=1)                     # [B,H]
            idx = pool_pos.view(-1, 1, 1).expand(-1, 1, hidden_last.size(-1))
            return hidden_last.gather(1, idx).squeeze(1)

        def _soft_rank(self, x):
            """미분가능 순위. r_i = 1 + Σ_j σ((x_j - x_i)/τ) — τ→0이면 실제 순위로 수렴.

            hard rank는 계단함수라 gradient가 0이다. sigmoid로 '몇 개가 나보다 큰가'를
            부드럽게 세어 순위를 근사한다. τ가 작을수록 실제 순위에 가깝지만 gradient가
            날카로워진다(τ=0.1 부근이 무난).
            """
            d = (x.unsqueeze(0) - x.unsqueeze(1)) / self.soft_tau   # [B,B,3] x_j - x_i
            s = torch.sigmoid(d).sum(dim=0)                          # [B,3] 나보다 큰 개수(자기 쌍 0.5 포함)
            return s + 0.5                                           # r = 1 + (s - 0.5)

        def _soft_spearman_loss(self, pred, target):
            """1 - soft-Spearman ρ (영역별 평균). 지표를 직접 최적화한다.

            pairwise hinge(_rank_loss)는 '순위 지표의 근사의 근사'다. 여기서는 예측을
            미분가능 순위로 바꾼 뒤 정답 순위와의 Pearson을 직접 최대화한다
            (Spearman = 순위에 대한 Pearson). 정답 순위는 상수라 hard rank를 써도 되지만,
            동점 처리를 예측과 대칭으로 두려고 같은 soft rank를 쓴다.
            """
            B = pred.shape[0]
            if B < 3:                      # 순위상관은 3개 미만이면 의미 없음
                return pred.new_zeros(())
            eps = 1e-6
            rp = self._soft_rank(pred)
            rt = self._soft_rank(target)
            p = rp - rp.mean(dim=0, keepdim=True)
            t = rt - rt.mean(dim=0, keepdim=True)
            num = (p * t).sum(dim=0)
            den = torch.sqrt((p * p).sum(dim=0) * (t * t).sum(dim=0) + eps)
            return (1.0 - num / (den + eps)).mean()

        def _rank_loss(self, pred, target):
            """
            배치 내 pairwise margin ranking loss (영역별). 정답 순위와 예측 순위를
            일치시켜 '평균으로 뭉치는' 현상을 직접 억제한다 → Spearman 직접 최적화.
            pred/target: [B,3] in [0,1]. 정답 차가 있는 쌍만 계산.
            """
            B = pred.shape[0]
            if B < 2:
                return pred.new_zeros(())
            dp = pred.unsqueeze(1) - pred.unsqueeze(0)      # [B,B,3] pred_i - pred_j
            dt = target.unsqueeze(1) - target.unsqueeze(0)  # [B,B,3] target_i - target_j
            adt = dt.abs()
            sign = torch.sign(dt)                            # +1: i가 j보다 높아야
            # margin: gap 모드면 정답 격차 이상은 요구하지 않는다(소격차 쌍과 정합).
            m = torch.clamp(adt, max=self.rank_margin) if self.rank_margin_mode == "gap" \
                else self.rank_margin
            # hinge: 올바른 방향으로 margin만큼 벌어지지 않으면 벌점
            loss = torch.relu(m - sign * dp)
            mask = (adt > 1e-6).float()                      # 동점 쌍 제외
            if self.rank_hard_gap > 0:
                # 소격차 쌍만 남긴다. 배치에 해당 쌍이 하나도 없으면 분모 clamp로 손실 0.
                mask = mask * (adt <= self.rank_hard_gap).float()
            return (loss * mask).sum() / mask.sum().clamp(min=1.0)

        def _sparse_ce_loss(self, hidden_last, labels):
            """라벨 위치만 lm_head를 적용하는 생성 CE — full [B,T,V] logits 미생성.

            HF 내부 CE(라벨 우측 시프트, ignore_index=-100, valid 토큰 평균)와
            수학적으로 동일. hidden_states[-1]은 최종 norm 이후 표현이라
            lm_head(hidden_states[-1]) == out.logits (로컬 등가성 검증 완료).
            """
            shift_h = hidden_last[:, :-1, :]
            shift_l = labels[:, 1:]
            m = shift_l != -100
            n_valid = int(m.sum())
            if n_valid == 0:
                return hidden_last.new_zeros((), dtype=torch.float32)
            hv = shift_h[m]                            # [N,H] — N ≈ 전체의 ~17%
            lv = shift_l[m]
            lm_head = self.base.get_output_embeddings()
            loss_sum = None
            CH = 2048                                  # 행 청크 — fp32 logits 피크 제한
            for s in range(0, n_valid, CH):
                lg = lm_head(hv[s:s + CH]).float()
                part = nn.functional.cross_entropy(lg, lv[s:s + CH], reduction="sum")
                loss_sum = part if loss_sum is None else loss_sum + part
            return loss_sum / n_valid

        def forward(self, input_ids, attention_mask, pool_pos,
                    labels=None, score_targets=None,
                    sub_targets=None, sub_mask=None):
            if self.gen_off:
                labels_for_gen = None      # 생성 손실 자체를 끔 (외부 사전학습)
            else:
                labels_for_gen = labels
            use_sparse = self.sparse_ce and labels_for_gen is not None
            kw = dict(input_ids=input_ids, attention_mask=attention_mask,
                      output_hidden_states=True, use_cache=False)
            if self.gen_off:
                out = self.base(**kw)
            elif use_sparse:
                # full logits 자체를 안 만들도록 시도 (HF: logits_to_keep=1 → 마지막 1개만)
                if self._ltk_ok is None:
                    try:
                        out = self.base(**kw, logits_to_keep=1)
                        self._ltk_ok = True
                    except TypeError:
                        self._ltk_ok = False
                        out = self.base(**kw)
                elif self._ltk_ok:
                    out = self.base(**kw, logits_to_keep=1)
                else:
                    out = self.base(**kw)
            else:
                out = self.base(**kw, labels=labels_for_gen)   # 근거 구간만 라벨, 입력은 -100
            pooled = self._pooled(out.hidden_states[-1], pool_pos, attention_mask)
            # reg_head는 fp32(안정성). bf16 백본에서 온 pooled를 fp32로 캐스팅해 dtype 정합.
            head_out = torch.sigmoid(self.head_forward(pooled.float()))  # [B,3|9]
            if self.subcrit:
                sub_pred = head_out                                   # [B,9] 준거 예측
                score_pred = torch.stack(                             # [B,3] trait = 준거 평균
                    [sub_pred[:, a:b].mean(dim=1) for a, b in self._sub_slices], dim=1)
            else:
                sub_pred = None
                score_pred = head_out                                 # [B,3]

            result = {"score_pred": score_pred}
            if labels is None or score_targets is None:
                return result

            # 손실은 전부 fp32에서 계산 (bf16 gen_loss가 섞여 들어와도 승격/안정화).
            if self.gen_off:
                gen_loss = score_pred.new_zeros(())              # 생성 손실 없음 (사전학습)
            elif use_sparse:
                gen_loss = self._sparse_ce_loss(out.hidden_states[-1], labels)
            else:
                gen_loss = out.loss.float()                     # LM cross-entropy (근거 구간)
            if self.trait_w is None:
                reg_loss = nn.functional.mse_loss(score_pred, score_targets.float())
            else:
                # 영역별 가중 MSE. trait_w는 평균 1로 정규화돼 있어 전체 스케일 불변.
                se = (score_pred - score_targets.float()) ** 2          # [B,3]
                reg_loss = (se * self.trait_w.to(se.device)).mean()

            # 총점(3영역 평균) 일관성 — 공식 지표가 평균점수 위에서 계산되므로
            # 그 단위에 직접 손실을 건다. 영역별 MSE만으로는 오차가 상쇄될 유인이
            # 없어 교차공분산이 그대로 남는다(실측 상관 0.45~0.53).
            total_loss_term = score_pred.new_zeros(())
            if self.total_weight > 0:
                total_loss_term = nn.functional.mse_loss(
                    score_pred.mean(dim=1), score_targets.float().mean(dim=1))
                reg_loss = reg_loss + self.total_weight * total_loss_term
            # 하위 준거 MSE — 준거 라벨이 있는 샘플(sub_mask=1, NIKL 혼합분)만.
            # trait MSE와 같은 1/(2σ²) 가중 안에 넣어 스케일을 공유한다.
            sub_loss = score_pred.new_zeros(())
            if self.subcrit and sub_targets is not None and sub_mask is not None:
                m = sub_mask.bool()
                if m.any():
                    sub_loss = nn.functional.mse_loss(
                        sub_pred[m], sub_targets[m].float())

            # 회귀 항: 1/(2σ²)L + (1/2)logσ²  (가우시안, 항상 (0.5,0.5))
            # 생성 항: (c_loss/σ²)L + c_log·logσ²  (c는 gen_uncertainty로 A/B, §1)
            # exp 오버플로 방지로 log_var를 [-7,7]로 클램프 (σ² ∈ [~9e-4, ~1e3]).
            log_var_reg = self.log_var_reg.clamp(-7.0, 7.0)
            log_var_gen = self.log_var_gen.clamp(-7.0, 7.0)
            prec_reg = torch.exp(-log_var_reg)
            prec_gen = torch.exp(-log_var_gen)
            if self.reg_off:   # 생성 전용 arm — 회귀 항 전체 제외(log_var_reg 표류 방지)
                total = score_pred.new_zeros(())
            else:
                total = 0.5 * prec_reg * (reg_loss + self.subcrit_weight * sub_loss) \
                        + 0.5 * log_var_reg
            if not self.gen_off:   # gen_off면 gen 항 전체 제외 (log_var_gen 표류 방지)
                total = total + self.gen_c_loss * prec_gen * gen_loss \
                              + self.gen_c_log * log_var_gen

            rank_loss = score_pred.new_zeros(())
            if self.rank_weight > 0:
                rank_loss = self._rank_loss(score_pred, score_targets.float())
                total = total + self.rank_weight * rank_loss
            if self.corr_weight > 0:
                corr_loss = self._corr_loss(score_pred, score_targets.float())
                total = total + self.corr_weight * corr_loss
                rank_loss = rank_loss + corr_loss  # 로그 표시용 합산
            total_rank_loss = score_pred.new_zeros(())
            if self.total_rank_weight > 0:
                total_rank_loss = self._rank_loss(
                    score_pred.mean(dim=1, keepdim=True),
                    score_targets.float().mean(dim=1, keepdim=True))
                total = total + self.total_rank_weight * total_rank_loss
                rank_loss = rank_loss + total_rank_loss  # 로그 표시용 합산
            if self.soft_weight > 0:
                soft_loss = self._soft_spearman_loss(score_pred, score_targets.float())
                total = total + self.soft_weight * soft_loss
                rank_loss = rank_loss + soft_loss  # 로그 표시용 합산(aux)

            result.update(total_loss=total, reg_loss=reg_loss, gen_loss=gen_loss,
                          rank_loss=rank_loss, sub_loss=sub_loss,
                          tot_loss=total_loss_term)
            return result

    class MTDataset(torch.utils.data.Dataset):
        """예제 → (input_ids, labels(입력 -100/근거 tokens), pool_pos, score_targets)."""

        def __init__(self, examples: Sequence[Example], tokenizer, max_len: int,
                     gen_scores: bool = False, decimals: int = 2):
            self.rows = []
            for ex in examples:
                inp = build_input_text(ex)
                inp_ids = tokenizer(inp, add_special_tokens=True).input_ids
                # 근거가 없는 샘플(외부 코퍼스 혼합 학습)은 생성 타깃을 아예 붙이지
                # 않는다 → labels 전부 -100 → 그 샘플만 생성 손실에서 제외되고
                # 회귀/상관 손실에는 정상 참여한다. AWES 샘플의 생성 학습은 불변.
                _rats = ex.rationales or {}
                _has_gen = any((_rats.get(t) or "").strip() for t in C.TRAITS)
                if not _has_gen:
                    tgt_ids = []
                else:
                    tgt = build_target_text(ex, gen_scores=gen_scores, decimals=decimals)
                    # 근거 앞에 개행 하나 — 입력/근거 경계를 명확히
                    tgt_ids = tokenizer("\n" + tgt, add_special_tokens=False).input_ids
                    tgt_ids = tgt_ids + [tokenizer.eos_token_id]

                # 입력이 너무 길면 학생 글 쪽(입력 앞부분 유지 어려움)을 뒤에서 자른다.
                # 회귀 위치(입력 마지막 토큰)와 근거 타깃은 보존해야 하므로 입력을 자른다.
                budget = max_len - len(tgt_ids)
                if budget < 8:                       # 근거가 max_len에 육박하면 방어
                    tgt_ids = tgt_ids[:max_len - 8]
                    budget = 8
                if len(inp_ids) > budget:
                    inp_ids = inp_ids[:budget]

                input_ids = inp_ids + tgt_ids
                labels = [-100] * len(inp_ids) + list(tgt_ids)
                # 하위 준거 라벨 — Example.meta["subcrit"]: [1,5] 스케일 9개
                # (C.TRAITS 순서 평탄화: con1~5 + org1~2 + exp1~2). 없으면 0벡터
                # + mask 0 → 준거 손실에서 제외 (subcrit=False 아키에선 무시됨).
                _sub = (ex.meta or {}).get("subcrit")
                if _sub is not None and len(_sub) == SUB_TOTAL:
                    sub_targets = [_norm_score(float(v)) for v in _sub]
                    sub_mask = 1.0
                else:
                    sub_targets = [0.0] * SUB_TOTAL
                    sub_mask = 0.0
                self.rows.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "pool_pos": len(inp_ids) - 1,     # 입력 마지막 토큰
                    "score_targets": [_norm_score(ex.scores[t]) for t in C.TRAITS],
                    "sub_targets": sub_targets,
                    "sub_mask": sub_mask,
                })

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch, pad_id):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels, pool_pos, scores = [], [], [], [], []
        subs, submask = [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [pad_id] * pad)   # right padding
            attn.append([1] * n + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)
            pool_pos.append(b["pool_pos"])
            scores.append(b["score_targets"])
            subs.append(b["sub_targets"])
            submask.append(b["sub_mask"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pool_pos": torch.tensor(pool_pos, dtype=torch.long),
            "score_targets": torch.tensor(scores, dtype=torch.float),
            "sub_targets": torch.tensor(subs, dtype=torch.float),
            "sub_mask": torch.tensor(submask, dtype=torch.float),
        }

    return MultiTaskModel, MTDataset, collate


# ---------------------------------------------------------------------------
# 3. 파이프라인
# ---------------------------------------------------------------------------
class MultiTaskPipeline(ScoringPipeline):
    """
    VGGT식 멀티태스크 채점기. run_cv 가 fold마다 새 인스턴스를 만든다.

        from awes_common import run_cv, TrainConfig
        from awes_common.multitask import MultiTaskPipeline
        run_cv(MultiTaskPipeline, TrainConfig())
    """

    name = "multitask"

    def __init__(self, cfg: TrainConfig, fold: int, out_dir: str):
        super().__init__(cfg, fold, out_dir)
        self.arch = {**ARCH_DEFAULTS, **(cfg.arch or {})}
        self.model = None
        self.tokenizer = None
        self._pad_id = None

    # --- 학습 -------------------------------------------------------------
    def fit(self, data: FoldData) -> None:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
        from peft import LoraConfig, get_peft_model

        from .pipeline import set_seed
        MultiTaskModel, MTDataset, collate = _build_torch_parts()
        self._gen = set_seed(self.cfg.seed)   # random/numpy/torch/cuda + cudnn 결정적

        tok = AutoTokenizer.from_pretrained(self.cfg.backbone, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        self.tokenizer = tok
        self._pad_id = tok.pad_token_id

        dtype = torch.bfloat16 if self.cfg.bf16 else torch.float32
        # tf5.14는 flash_attention_2를 from_pretrained의 init(meta device)에서 검사해
        # "flash is not available on CPU"로 막는다(배치가 init보다 나중이라 device_map도 무효).
        # 정식 해결: eager로 로드해 init을 통과시키고 → GPU에 올린 뒤 → set_attn_implementation로
        # flash로 전환한다. 그 시점엔 cuda라 검사 통과 → forward에서 실제 flash 디스패치.
        _want_flash = self.cfg.attn_impl == "flash_attention_2"
        base = AutoModelForCausalLM.from_pretrained(
            self.cfg.backbone, dtype=dtype, trust_remote_code=True,  # tf 5.x: torch_dtype→dtype
            attn_implementation=("eager" if _want_flash else self.cfg.attn_impl),
        )
        if _want_flash and torch.cuda.is_available():
            base = base.to("cuda")                        # init 통과 후 GPU에서 flash로 전환
            if hasattr(base, "set_attn_implementation"):
                base.set_attn_implementation("flash_attention_2")
            else:                                          # 구/신 API 폴백
                base.config._attn_implementation = "flash_attention_2"
            eff = getattr(base.config, "_attn_implementation", "?")
            print(f"    [attn] flash 전환 → config._attn_implementation={eff}", flush=True)
        base.config.use_cache = False
        init_from = self.arch.get("init_from")
        if init_from:
            # Stage A(외부 사전학습) 어댑터에서 이어 학습. r/targets는 저장된
            # adapter_config를 따르므로 cfg와 다르면 저장 config가 우선한다.
            from peft import PeftModel
            adir = os.path.join(init_from, "adapter")
            base = PeftModel.from_pretrained(base, adir, is_trainable=True)
            print(f"    [init] 사전학습 어댑터에서 시작: {adir}", flush=True)
        else:
            lora = LoraConfig(
                r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=list(self.cfg.lora_targets),
                task_type="CAUSAL_LM", bias="none",
            )
            base = get_peft_model(base, lora)

        # gradient checkpointing (cfg로 토글) — 7B를 seq 2048·batch>1로 돌리면 활성값이
        # 80GB를 넘긴다. 활성값 재계산으로 메모리 확보(연산 ~30%↑). use_cache=False 필수.
        # LoRA(동결 백본)라 입력 임베딩에 grad가 흐르도록 enable_input_require_grads도 켠다.
        if self.cfg.grad_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()

        hidden = base.config.hidden_size
        model = MultiTaskModel(base, hidden, self.arch)
        if init_from and not self.arch.get("init_head", True):
            print("    [init] 헤드는 새로 시작 (LoRA만 웜스타트)", flush=True)
        elif init_from:
            hp = os.path.join(init_from, "heads.pt")
            if os.path.exists(hp):
                ck = torch.load(hp, map_location="cpu")
                _k = "reg_heads" if model.split_heads else "reg_head"
                if _k in ck:
                    model.head_modules().load_state_dict(ck[_k])
                else:   # 구조가 바뀐 산출물에서 이어받는 경우 — 헤드는 새로 학습
                    print(f"    [init] heads.pt에 {_k} 없음 → 헤드 웜스타트 생략",
                          flush=True)
                with torch.no_grad():
                    model.log_var_reg.copy_(ck["log_var_reg"])
                    model.log_var_gen.copy_(ck["log_var_gen"])
                print("    [init] 회귀 헤드/log_var 웜스타트", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device=device)

        # ── 학습 파라미터는 전부 fp32 마스터로 (핵심 안정화) ──────────────────
        # LoRA 어댑터는 bf16 백본을 따라 bf16으로 생성되는데, AdamW eps(1e-8)가
        # bf16에서 0으로 언더플로해 m/(√v+0) → NaN을 유발한다(첫 step부터 total=nan).
        # 동결 백본은 bf16 유지(메모리), 학습 파라미터만 fp32로 올리고 forward는
        # autocast(bf16)로 계산한다 = fp32 옵티마이저 안정성 + bf16 속도/메모리.
        model.head_modules().to(torch.float32)               # 회귀 헤드 fp32(분리 시 전체)
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:  # LoRA 어댑터 fp32 승격
                p.data = p.data.float()
        self.model = model

        ds = MTDataset(data.train, tok, self.cfg.max_len,
                       gen_scores=bool(self.arch.get("gen_scores", False)),
                       decimals=int(self.arch.get("score_decimals", 2)))
        loader = DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=True,
            collate_fn=lambda b: collate(b, self._pad_id),
            generator=self._gen,   # shuffle 순서 고정
        )

        # 합성 대조쌍 배처 (contrastive_dir 없으면 None → 대조 손실 경로 전체 skip).
        # eval fold를 exclude_fold로 넘겨 누수 차단(변형은 원본 _fold 상속).
        from . import contrastive as _contrastive
        self._pair_batcher = _contrastive.make_batcher(
            self.arch, self.fold, tok, self.cfg.max_len, self.cfg.seed,
            build_input_text, self._pad_id)
        self._pair_w = float(self.arch.get("contrastive_weight", 0.0))
        self._pair_margin = float(self.arch.get("contrastive_margin", 0.05))

        # ── best-epoch 체크포인트 ──────────────────────────────────────────
        # arch["save_best"]가 참이면 매 에폭 holdout ρ를 보고 최고 시점의 학습 가능
        # 파라미터(LoRA A/B + 회귀 헤드 + log_var)를 CPU에 떠 둔다. 학습이 끝나면
        # 그 시점으로 되돌린 뒤 저장·예측한다.
        # ⚠️ **eval 셋으로 에폭을 고르는 것**이므로, 그 eval이 정직한 추정치여야 하는
        #    상황에서는 켜면 안 된다. 특히 final_train.py는 eval=official_val(400)이라
        #    켜면 val ρ가 낙관적으로 편향되고 '제출 없는 검증'의 의미가 사라진다.
        #    fold CV(mix_train/run_experiment)는 holdout이 이미 선택에 쓰이는 셋이라
        #    문제 없다. 그래서 기본값은 False이고 호출 측에서 명시적으로 켠다.
        save_best = bool(self.arch.get("save_best", False)) and self.cfg.eval_every_epoch
        best_rho, best_epoch, best_state = float("-inf"), 0, None

        trainable = [p for p in model.parameters() if p.requires_grad]
        # eps도 bf16 대비 여유를 두어 명시 (fp32라 1e-8도 되지만 방어적으로 1e-6).
        optim = torch.optim.AdamW(trainable, lr=self.cfg.lr, eps=1e-6)
        steps_per_epoch = max(1, len(loader) // self.cfg.grad_accum)
        total_steps = steps_per_epoch * self.cfg.epochs
        sched = get_cosine_schedule_with_warmup(
            optim, int(total_steps * self.cfg.warmup_ratio), total_steps,
        )
        use_amp = (device == "cuda")

        model.train()
        step = 0
        n_batches = len(loader)
        for epoch in range(self.cfg.epochs):
            running = {"total": 0.0, "reg": 0.0, "gen": 0.0, "rank": 0.0, "pair": 0.0}
            n_skipped, n_bad_grad = 0, 0
            t_epoch = time.time()
            optim.zero_grad()
            for i, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(**batch)
                total_loss = out["total_loss"]

                # 첫 배치 1회 진단 — NaN이 재발하면 reg/gen 중 어디서 나는지 즉시 식별.
                if epoch == 0 and i == 0:
                    _tl = out.get('tot_loss')
                    _tls = f" tot={_tl.item():.4f}" if _tl is not None else ""
                    print(f"    [diag] reg={out['reg_loss'].item():.4f}{_tls} "
                          f"sub={out.get('sub_loss', total_loss.new_zeros(())).item():.4f} "
                          f"gen={out['gen_loss'].item():.4f} total={total_loss.item():.4f} "
                          f"score_pred_finite={bool(torch.isfinite(out['score_pred']).all())} "
                          f"labels_valid={int((batch['labels'] != -100).sum())}/"
                          f"{batch['labels'].numel()}", flush=True)

                # NaN/inf 안전망 — 한 step의 비정상 손실이 가중치를 오염시키지 않게 스킵.
                if not torch.isfinite(total_loss):
                    n_skipped += 1
                    optim.zero_grad()
                    continue

                (total_loss / self.cfg.grad_accum).backward()
                running["total"] += total_loss.item()
                running["reg"] += out["reg_loss"].item()
                running["gen"] += out["gen_loss"].item()
                running["rank"] += float(out.get("rank_loss", 0.0))

                if (i + 1) % self.cfg.grad_accum == 0:
                    # 합성 대조쌍 손실 — optim step당 1회. 합성본은 labels=None으로
                    # forward돼 score_pred만 나오므로(계약 1) MSE/gen에 닿을 경로가 없다.
                    # grad는 이번 step의 실데이터 grad에 누적된 뒤 함께 clip·step된다.
                    if self._pair_batcher is not None and self._pair_w > 0:
                        pb = self._pair_batcher.next_batch(device)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                            enabled=use_amp):
                            sp = model(input_ids=pb["input_ids"],
                                       attention_mask=pb["attention_mask"],
                                       pool_pos=pb["pool_pos"])["score_pred"]
                            pair_loss = _contrastive.pair_rank_loss(
                                sp, pb["high_idx"], pb["low_idx"], pb["trait_idx"],
                                pb["gap"], self._pair_margin)
                        if torch.isfinite(pair_loss):
                            (self._pair_w * pair_loss).backward()
                            running["pair"] += float(pair_loss)

                    # grad 유한성 가드 — 손실이 유한해도 backward에서 NaN grad(0×NaN)가
                    # 나올 수 있다. clip이 돌려주는 norm이 비유한이면 step을 건너뛰어
                    # 파라미터(특히 log_var) 오염을 막는다.
                    gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    if torch.isfinite(gnorm):
                        optim.step()
                        sched.step()
                    else:
                        n_bad_grad += 1
                    optim.zero_grad()
                    step += 1

                # 진행률 — epoch 내부에서도 살아있음을 보이도록 배치 N개마다 출력.
                if (i + 1) % LOG_EVERY == 0:
                    done = i + 1 - n_skipped
                    rate = (i + 1) / (time.time() - t_epoch)
                    eta = (n_batches - (i + 1)) / rate if rate > 0 else 0
                    print(f"    [fold{self.fold}] epoch {epoch + 1}/{self.cfg.epochs} "
                          f"batch {i + 1}/{n_batches} loss={running['total']/max(1,done):.4f} "
                          f"({rate:.2f} it/s, ETA {eta/60:.1f}m)", flush=True)

            n = max(1, len(loader) - n_skipped)
            with torch.no_grad():
                sig_r = float(torch.exp(0.5 * model.log_var_reg.detach()))
                sig_g = float(torch.exp(0.5 * model.log_var_gen.detach()))
            bad_msg = ""
            if n_skipped or n_bad_grad:
                bad_msg = f" (skipped {n_skipped}, bad_grad {n_bad_grad})"
            _aux = self.arch.get("rank_weight", 0) > 0 or self.arch.get("corr_weight", 0) > 0
            rank_msg = f" aux={running['rank']/n:.4f}" if _aux else ""
            # 대조 손실은 optim step당 1회라 step 수(=total_steps/epoch)로 평균낸다.
            pair_msg = ""
            if self._pair_batcher is not None and self._pair_w > 0:
                pair_msg = f" pair={running['pair']/max(1,steps_per_epoch):.4f}"
            dt = time.time() - t_epoch
            print(f"  [fold{self.fold}] epoch {epoch + 1}/{self.cfg.epochs} "
                  f"total={running['total']/n:.4f} reg={running['reg']/n:.4f} "
                  f"gen={running['gen']/n:.4f}{rank_msg}{pair_msg} σr={sig_r:.3f} σg={sig_g:.3f}"
                  f"{bad_msg} [{dt/60:.1f}m]", flush=True)

            # 검증 곡선 — 점수 헤드만(생성 없이) 저렴하게 holdout 지표를 찍는다.
            if self.cfg.eval_every_epoch:
                vres, ratios = self._score_only_eval(data.eval)
                rstr = " ".join(f"{C.TRAIT_KOR[t]}={ratios[t]:.2f}" for t in C.TRAITS)
                print(f"    [fold{self.fold}] └ VAL RMSE={vres.rmse:.4f} "
                      f"ρ={vres.spearman:.4f} | std비 {rstr}", flush=True)
                # best-epoch 스냅샷 — 2026-08-01 실측으로 E*=2~4가 확인됐다. 마지막
                # 에폭만 저장하면 매 실험이 정점에서 −0.03~−0.08 내려온 모델을 남긴다.
                if save_best and vres.spearman > best_rho:
                    best_rho, best_epoch = vres.spearman, epoch + 1
                    best_state = self._snapshot_trainable()
                model.train()   # eval 모드에서 복귀

        # 정점 가중치로 되돌린 뒤 저장한다 — predict()도 이 가중치를 쓴다.
        if save_best and best_state is not None and best_epoch != self.cfg.epochs:
            self._restore_trainable(best_state)
            print(f"  [fold{self.fold}] ⤺ best-epoch 복원: epoch {best_epoch}/"
                  f"{self.cfg.epochs} (ρ={best_rho:.4f})", flush=True)
        elif save_best:
            print(f"  [fold{self.fold}] ⤺ best-epoch = 최종 epoch "
                  f"{best_epoch} (ρ={best_rho:.4f}) — 복원 불필요", flush=True)

        self._save_adapter()
        # distillation 교사용: train 회귀 예측 저장 (다음 단계 AR이 이 점수로 학습)
        self._save_train_reg_scores(data.train)

    # --- 검증(점수만) — 과소학습/평균회귀 진단용 ---------------------------
    def _reg_predict(self, examples):
        """회귀 헤드만으로 점수 예측(생성 없음). [Prediction(score-only)] 반환."""
        import torch
        model, tok = self.model, self.tokenizer
        device = next(model.parameters()).device
        model.eval()
        bs = int(self.arch.get("gen_batch_size", 8))
        preds = []
        with torch.no_grad():
            for s in range(0, len(examples), bs):
                chunk = examples[s:s + bs]
                tok.padding_side = "right"
                enc = tok([build_input_text(e) for e in chunk], return_tensors="pt",
                          padding=True, truncation=True, max_length=self.cfg.max_len).to(device)
                pool_pos = enc["attention_mask"].sum(dim=1) - 1
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    out = model.base(input_ids=enc["input_ids"],
                                     attention_mask=enc["attention_mask"],
                                     output_hidden_states=True, use_cache=False)
                    u = torch.sigmoid(model.head_forward(
                        model._pooled(out.hidden_states[-1], pool_pos,
                                      enc["attention_mask"]).float()))
                for j, ex in enumerate(chunk):
                    preds.append(Prediction(
                        id=ex.id,
                        scores={t: _denorm_score(u[j, k].item()) for k, t in enumerate(C.TRAITS)}))
        return preds

    def _save_train_reg_scores(self, train_examples):
        """distillation 교사용 — train 데이터에 대한 회귀헤드 예측을 저장.
        AR이 이 점수를 gold 대신 타깃으로 학습하면 회귀헤드 정확도를 이전받는다."""
        preds = self._reg_predict(list(train_examples))
        rows = [{"id": p.id, "score": {t: round(p.scores[t], 4) for t in C.TRAITS}} for p in preds]
        save_jsonl(rows, os.path.join(self.out_dir, "reg_scores_train.jsonl"))
        print(f"    [fold{self.fold}] 회귀 train 예측 저장: {len(rows)}건 → reg_scores_train.jsonl",
              flush=True)

    def _score_only_eval(self, examples):
        """생성 없이 회귀 헤드만으로 holdout 점수 지표. (EvalResult, 영역별 std비) 반환."""
        import statistics as st
        from .metrics import evaluate

        cap = self.cfg.eval_subset
        subset = list(examples)[:cap] if cap and cap > 0 else list(examples)
        preds = self._reg_predict(subset)
        res = evaluate(subset, preds)
        ratios = {}
        for t in C.TRAITS:
            ps = st.pstdev([p.scores[t] for p in preds]) if len(preds) > 1 else 0.0
            ts = st.pstdev([e.scores[t] for e in subset]) if len(subset) > 1 else 0.0
            ratios[t] = ps / ts if ts > 0 else 0.0
        return res, ratios

    # --- 추론 -------------------------------------------------------------
    def predict(self, examples: Sequence[Example]) -> List[Prediction]:
        import torch

        assert self.model is not None, "fit() 먼저 호출해야 한다"
        from .pipeline import set_seed
        set_seed(self.cfg.seed)   # 근거 생성이 샘플링(temp>0)이라 예측 재현 위해 재시딩
        model, tok = self.model, self.tokenizer
        device = next(model.parameters()).device
        model.eval()
        use_amp = (device.type == "cuda")
        bs = int(self.arch.get("gen_batch_size", 8))
        do_sample = self.cfg.gen_rationale and self.arch["gen_temperature"] > 0

        results: Dict[str, Prediction] = {}
        total = len(examples)
        t0 = time.time()

        for s in range(0, total, bs):
            chunk = examples[s:s + bs]
            texts = [build_input_text(e) for e in chunk]

            # 1) 점수: 우측 패딩(학습과 동일한 위치 → RoPE 정합), per-example 마지막 실토큰 pool.
            tok.padding_side = "right"
            enc_r = tok(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=self.cfg.max_len).to(device)
            pool_pos = enc_r["attention_mask"].sum(dim=1) - 1
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                 enabled=use_amp):
                base_out = model.base(
                    input_ids=enc_r["input_ids"], attention_mask=enc_r["attention_mask"],
                    output_hidden_states=True, use_cache=False,
                )
                pooled = model._pooled(base_out.hidden_states[-1], pool_pos,
                                       enc_r["attention_mask"])  # [B,H]
                u = torch.sigmoid(model.head_forward(pooled.float()))             # [B,3]

            # 2) 근거: 좌측 패딩(배치 생성 표준 — generate가 position_ids를 마스크로 보정).
            tok.padding_side = "left"
            enc_l = tok(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=self.cfg.max_len).to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                 enabled=use_amp):
                gen_ids = model.base.generate(
                    input_ids=enc_l["input_ids"], attention_mask=enc_l["attention_mask"],
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=do_sample,
                    temperature=self.arch["gen_temperature"],
                    top_p=self.arch["gen_top_p"],
                    pad_token_id=self._pad_id,
                )
            new = gen_ids[:, enc_l["input_ids"].shape[1]:]
            texts_out = tok.batch_decode(new, skip_special_tokens=True)

            gen_scores_mode = bool(self.arch.get("gen_scores", False))
            if gen_scores_mode:
                from .autoregressive import parse_scoring_output

            for j, ex in enumerate(chunk):
                reg_scores = {t: _denorm_score(u[j, k].item()) for k, t in enumerate(C.TRAITS)}
                if gen_scores_mode:
                    # 시나리오 B: 제출이 보는 것 = 생성된 JSON. 점수·근거를 생성에서 파싱.
                    parsed = parse_scoring_output(texts_out[j])
                    if parsed is not None:
                        gsc, grat = parsed
                        results[ex.id] = Prediction(id=ex.id, scores=gsc, rationales=grat)
                    else:
                        results[ex.id] = self.fallback_prediction(ex)  # 제출 시 파싱실패 대응
                else:
                    # 기존: 점수=회귀헤드, 근거=생성. (회귀 헤드 방식, 시나리오 A)
                    rationales = parse_rationale_json(texts_out[j]) or {t: "" for t in C.TRAITS}
                    results[ex.id] = Prediction(id=ex.id, scores=reg_scores, rationales=rationales)

            done = min(s + bs, total)
            rate = done / max(1e-6, time.time() - t0)
            eta = (total - done) / rate if rate > 0 else 0
            print(f"    [fold{self.fold}] predict {done}/{total} "
                  f"({rate:.1f} ex/s, ETA {eta/60:.1f}m)", flush=True)

        return [results[e.id] for e in examples]

    # --- 정리 -------------------------------------------------------------
    def teardown(self) -> None:
        import torch
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- best-epoch 스냅샷 -------------------------------------------------
    # 학습 가능 파라미터만 뜬다(LoRA A/B + 회귀 헤드 + log_var). 동결된 백본은
    # 제외되므로 9B/LoRA r32 기준 CPU에서 100MB 남짓이다 — GPU 메모리는 안 쓴다.
    def _snapshot_trainable(self) -> dict:
        return {n: p.detach().to("cpu", copy=True)
                for n, p in self.model.named_parameters() if p.requires_grad}

    def _restore_trainable(self, snap: dict) -> None:
        import torch
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in snap:
                    p.copy_(snap[n].to(device=p.device, dtype=p.dtype))

    # --- 저장 -------------------------------------------------------------
    def _save_adapter(self) -> None:
        import torch
        adapter_dir = os.path.join(self.out_dir, "adapter")
        self.model.base.save_pretrained(adapter_dir)       # LoRA 어댑터
        # split_heads면 reg_heads(ModuleList)를, 아니면 기존 reg_head를 저장한다.
        # 키 이름이 다르므로 로더(serve.py/eval_adapter.py)가 자동 분기할 수 있다.
        _head_sd = {("reg_heads" if getattr(self.model, "split_heads", False)
                     else "reg_head"): self.model.head_modules().state_dict()}
        torch.save(
            {**_head_sd,
             "log_var_reg": self.model.log_var_reg.detach().cpu(),
             "log_var_gen": self.model.log_var_gen.detach().cpu()},
            os.path.join(self.out_dir, "heads.pt"),
        )
        with open(os.path.join(self.out_dir, "arch.json"), "w", encoding="utf-8") as f:
            json.dump(self.arch, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 4. GPU 없이 도는 자체 테스트 (프롬프트/타깃/파서/정규화)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ex = Example(
        id="t1", prompt_num="1",
        prompt="인공지능 시대에 인간의 역할은 무엇인가?",
        essay="인공지능이 발전하면서 인간의 역할이 축소된다는 우려가 있다. 그러나 창의성은 여전히 인간의 몫이다.",
        scores={"content": 3.5, "organization": 3.0, "expression": 4.0},
        rationales={"content": "주제 이해는 명확하나 근거가 부족하다.",
                    "organization": "서론-본론 연결이 다소 느슨하다.",
                    "expression": "어휘가 정확하고 문장이 유창하다."},
    )

    print("=== 1. 입력 프롬프트 (근거 미포함) ===")
    inp = build_input_text(ex)
    print(inp[:400], "...\n")
    assert "학생 글" in inp and ex.essay[:10] in inp
    assert "근거가 부족" not in inp, "입력에 근거가 새면 안 됨(학습/추론 분포 불일치)"

    print("=== 2. 생성 타깃 (근거 JSON) ===")
    tgt = build_target_text(ex)
    print(tgt, "\n")
    assert json.loads(tgt)["content"].startswith("주제 이해")

    print("=== 3. 파서 (thinking + 후행콤마 + 코드펜스) ===")
    raw = ('<think>채점...</think>```json\n'
           '{"content":"내용 근거.","organization":"구성 근거.","expression":"표현 근거.",}\n```')
    p = parse_rationale_json(raw)
    print(p)
    assert p == {"content": "내용 근거.", "organization": "구성 근거.", "expression": "표현 근거."}
    assert parse_rationale_json("근거 없음") is None
    assert parse_rationale_json('{"content":"x"}') is None   # 필드 누락
    assert parse_rationale_json('{"content":"a","organization":"","expression":"c"}') is None  # 빈 근거

    print("=== 4. 점수 정규화 왕복 ===")
    for s in (1.0, 2.5, 3.0, 4.25, 5.0):
        assert abs(_denorm_score(_norm_score(s)) - s) < 1e-9, s
    assert _denorm_score(-0.5) == C.SCORE_MIN and _denorm_score(2.0) == C.SCORE_MAX  # 클립
    print("  1↔0.0, 5↔1.0, 클립 OK")

    print("\n✅ multitask 순수 로직 자체 테스트 통과 (torch 불필요 부분)")
