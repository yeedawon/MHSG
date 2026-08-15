# -*- coding: utf-8 -*-
"""
autoregressive.py — autoregressive(RaDME式) 채점 파이프라인 (방식 B').

멀티태스크(VGGT式)와 갈리는 축:
  멀티태스크      = 공유 hidden state → 회귀 헤드(숫자) ∥ 생성 헤드(근거). 두 헤드 병렬.
  autoregressive = 단일 LM 헤드로 근거와 점수를 "한 줄기 토큰열"로 생성한다.
                   각 영역마다 근거를 먼저 뱉고 그 뒤에 점수를 뱉게 하여
                   (rationale-then-score) 점수가 근거에 조건화되는 CoT식 채점.
                   (핸드오프 6장의 방식 B' = rationale-augmented)

왜 이렇게 하나 (검증 포인트):
  - 점수 정밀도: 회귀 헤드가 이론상 유리하다(연속값을 직접 회귀). AR은 점수를 텍스트
    토큰으로 내보내므로 정밀도 손해가 있으나, 근거→점수 CoT가 그 손해를 얼마나
    상쇄하는지가 이 아키텍처의 존재 이유다. 학술적으로도 생성형은 ordinal 지표
    (Spearman)에서 회귀에 불리한 경향이 보고됨(CLAUDE2.md 아키텍처 결정) → 실측 필요.
  - 근거 품질: 근거가 점수 산출의 "입력"이라 단순 부산물이 아니다. 근거와 점수가
    정합하도록 학습되므로 LLM Judge(10%)에 유리할 것으로 기대. 근거 우세는 B' 예상.

학습 = bf16 full-precision LoRA. 손실은 target(JSON) 토큰에만 걸고 프롬프트 토큰은
       -100으로 마스킹한다(지시문을 외우지 않게). 근거는 teacher가 정답 점수 조건부로
       합성해 둔 것(rationale_synthesis)을 target 재료로 쓴다.
       ⚠️ 정밀도는 multitask.py와 동일하게 bf16로 맞춘다 — QLoRA(4bit)를 쓰면 비교에
          "양자화 차이"가 섞여 아키텍처 비교가 오염된다(핸드오프 4.2, README 대칭성 규칙).
추론 = 생성 → JSON 파싱 → 실패 시 temperature 상향 재시도 → 그래도 실패면 fallback.
       실제 대회가 파싱 실패를 0점 처리하므로(2회 재시도 후) 이 방어선이 점수를 지킨다.

대칭성 (README '설계 결정' 4번): 백본/LoRA/epoch/lr/예산은 TrainConfig에서 공유한다.
AR 고유 노브(생성 온도, 재시도 횟수, 점수 소수 자릿수)만 cfg.arch에 둔다.

torch/transformers/peft는 메서드 안에서 로컬 임포트한다 — 맥 로컬에서 모듈 임포트만
해도 깨지지 않게(judge.py의 vLLM 로컬 임포트와 같은 방침). 그래서 이 파일의
프롬프트 빌더·파서는 GPU 없이 test로 검증할 수 있다.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

LOG_EVERY = 25  # 배치 N개마다 진행률 출력

from . import config as C
from .data import Example, FoldData, Prediction
from .pipeline import ScoringPipeline


# ===========================================================================
# 1. 프롬프트 / 타깃 빌더 (GPU 불필요 — 순수 문자열)
# ===========================================================================
# 채점 기준은 근거 합성(rationale_synthesis_utils.CRITERIA)과 동일 문구를 쓴다.
# 학습 target을 만든 teacher와 student가 같은 기준을 보게 하여 정합성을 높인다.
CRITERIA = {
    "content": "주제에 대한 이해도, 주장의 타당성, 근거의 충실성, 논증의 설득력",
    "organization": "글의 구조적 완결성, 문단 간 논리적 연결, 서론-본론-결론의 짜임새",
    "expression": "어휘의 적절성, 문장의 정확성과 유창성, 표현의 명료성",
}

# ---------------------------------------------------------------------------
# 생성 순서 (target_order) — 논문 트랙에서 추가. 대회 경로의 기본값은 "rs" 그대로다.
#   "rs" : rationale → score  (근거를 먼저 쓰고 그 결론으로 점수)  ← 기존 동작
#   "sr" : score → rationale  (점수를 먼저 내고 그 이유를 설명)
# JSON 키 순서가 곧 생성 토큰 순서이므로, 이 값이 조건부 구조를 결정한다.
# 세 arm(parallel / rs / sr) 비교에서 이 축만 바뀌어야 하므로 프롬프트·스키마·타깃
# **세 곳을 함께** 뒤집는다. 하나라도 어긋나면 지시와 타깃이 모순돼 학습이 깨진다.
# ---------------------------------------------------------------------------
ORDERS = ("rs", "sr")

SCORING_SYSTEM = (
    "너는 한국어 논증적 글을 채점하는 국어 교육 전문가이다. "
    "학생 글을 읽고 내용(content)·구성(organization)·표현(expression) 세 영역을 "
    "각각 1.0~5.0점(소수 둘째 자리까지)으로 채점한다. "
    "각 영역마다 먼저 글의 구체적 특징을 근거로 든 뒤, 그 근거에 따라 점수를 매겨라. "
    "점수는 근거의 결론이어야 하며, 근거 없이 점수부터 정하지 마라."
)

SCORING_SYSTEM_SR = (
    "너는 한국어 논증적 글을 채점하는 국어 교육 전문가이다. "
    "학생 글을 읽고 내용(content)·구성(organization)·표현(expression) 세 영역을 "
    "각각 1.0~5.0점(소수 둘째 자리까지)으로 채점한다. "
    "각 영역마다 먼저 점수를 매기고, 그 점수를 준 이유를 글의 구체적 특징으로 설명하라."
)


def _criteria_block() -> List[str]:
    lines = ["[채점 기준]"]
    for t in C.TRAITS:
        lines.append(f"  - {C.TRAIT_KOR[t]}({t}): {CRITERIA[t]}")
    return lines


# 출력 스키마: 영역마다 rationale를 먼저, score를 뒤에 둔다.
# JSON 키 순서 = 생성 토큰 순서이므로, 이 순서가 곧 "근거→점수" CoT를 강제한다.
_SCHEMA_HINT = "\n".join([
    "{",
    '  "content":      {"rationale": "<내용 근거 2~3문장>", "score": <1.0~5.0>},',
    '  "organization": {"rationale": "<구성 근거 2~3문장>", "score": <1.0~5.0>},',
    '  "expression":   {"rationale": "<표현 근거 2~3문장>", "score": <1.0~5.0>}',
    "}",
])

_SCHEMA_HINT_SR = "\n".join([
    "{",
    '  "content":      {"score": <1.0~5.0>, "rationale": "<내용 근거 2~3문장>"},',
    '  "organization": {"score": <1.0~5.0>, "rationale": "<구성 근거 2~3문장>"},',
    '  "expression":   {"score": <1.0~5.0>, "rationale": "<표현 근거 2~3문장>"}',
    "}",
])

_ORDER_INSTR = {
    "rs": "1. 각 영역마다 근거(rationale)를 먼저 쓰고, 그 근거에 따라 점수(score)를 정하라.",
    "sr": "1. 각 영역마다 점수(score)를 먼저 쓰고, 그 점수를 준 근거(rationale)를 뒤에 쓰라.",
}


def build_scoring_messages(ex: Example, order: str = "rs") -> List[Dict[str, str]]:
    """추론/학습 공용 대화 프롬프트(system+user). 정답 점수는 절대 넣지 않는다."""
    if order not in ORDERS:
        raise ValueError(f"target_order는 {ORDERS} 중 하나여야 한다: {order}")
    user = "\n".join([
        *_criteria_block(),
        "",
        "[글의 주제]",
        ex.prompt.strip(),
        "",
        "[학생 글]",
        ex.essay.strip(),
        "",
        "[출력 지침]",
        _ORDER_INSTR[order],
        "2. 근거는 글의 구체적 특징(특정 논거, 문단 구성, 표현)을 지목하라.",
        "3. 점수는 1.00~5.00 범위의 실수이며 소수 둘째 자리까지 쓴다.",
        "4. 아래 JSON 형식으로만 출력하고 다른 말은 덧붙이지 마라:",
        "",
        _SCHEMA_HINT if order == "rs" else _SCHEMA_HINT_SR,
    ])
    return [{"role": "system",
             "content": SCORING_SYSTEM if order == "rs" else SCORING_SYSTEM_SR},
            {"role": "user", "content": user}]


def build_target_json(ex: Example, decimals: int = 2, order: str = "rs") -> str:
    """
    학습 target(assistant 발화) 문자열.

    정답 점수(ex.scores)와 teacher 합성 근거(ex.rationales)를 결합한 정답 JSON.
    근거가 없으면(rationales=None) target을 만들 수 없다 — load_fold(require_rationale=True)가
    이미 걸러내지만 방어적으로 확인한다.
    키 순서를 rationale→score로 고정해 생성 순서와 일치시킨다(json.dumps는 dict 순서 보존).
    """
    if ex.rationales is None:
        raise ValueError(f"{ex.id}: 근거 없는 샘플로는 AR target을 만들 수 없다.")
    if order not in ORDERS:
        raise ValueError(f"target_order는 {ORDERS} 중 하나여야 한다: {order}")
    obj = {}
    for t in C.TRAITS:
        r = ex.rationales[t].strip()
        v = round(float(ex.scores[t]), decimals)
        # dict 리터럴 순서가 곧 json.dumps 출력 순서 = 생성 토큰 순서다.
        obj[t] = {"rationale": r, "score": v} if order == "rs" \
            else {"score": v, "rationale": r}
    return json.dumps(obj, ensure_ascii=False)


# ===========================================================================
# 2. 파서 (GPU 불필요 — 파싱 실패=0점 방어선의 핵심)
# ===========================================================================
def _extract_json_blob(text: str) -> Optional[str]:
    """가장 바깥 중괄호 쌍을 depth 카운팅으로 뽑는다(judge/synthesis와 동일 규약)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _coerce_score(v) -> Optional[float]:
    """score 값을 1.0~5.0 실수로 강제. 문자열 안에 숫자가 섞여 와도 첫 실수를 취한다."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if not m:
            return None
        x = float(m.group())
    else:
        return None
    return max(C.SCORE_MIN, min(C.SCORE_MAX, x))


def parse_scoring_output(text: str) -> Optional[Tuple[Dict[str, float], Dict[str, str]]]:
    """
    생성 텍스트 → (scores, rationales). 셋 중 하나라도 점수 파싱이 안 되면 None(재시도 대상).

    관대 처리: <think> 블록, 코드펜스, 후행 콤마, {trait:{rationale,score}} 중첩 구조 및
    {trait: <number>} 납작 구조를 모두 받는다. 근거가 비어도 점수만 유효하면 통과시킨다
    — 점수(90%)를 살리는 게 우선이고, 빈 근거는 judge에서 최저점을 받을 뿐 파싱 실패(0점)와
    다르다.
    """
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    blob = _extract_json_blob(text)
    if blob is None:
        return None
    blob = re.sub(r",\s*([}\]])", r"\1", blob)  # 후행 콤마 제거
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    scores: Dict[str, float] = {}
    rationales: Dict[str, str] = {}
    for t in C.TRAITS:
        if t not in data:
            return None
        cell = data[t]
        if isinstance(cell, dict):                       # 중첩 구조 {rationale, score}
            sc = _coerce_score(cell.get("score"))
            rat = cell.get("rationale", "")
        else:                                            # 납작 구조 {trait: 3.5}
            sc = _coerce_score(cell)
            rat = ""
        if sc is None:
            return None
        scores[t] = sc
        rationales[t] = rat.strip() if isinstance(rat, str) else ""
    return scores, rationales


# ===========================================================================
# 3. 파이프라인 — bf16 LoRA SFT + 생성 추론
# ===========================================================================
class AutoregressivePipeline(ScoringPipeline):
    """
    단일 causal LM을 bf16 LoRA로 SFT하여 근거→점수 JSON을 생성하는 채점기.

    fit    : (프롬프트, 정답 JSON) 쌍으로 SFT. 손실은 정답 JSON 토큰에만.
    predict: 생성 → parse_scoring_output → 실패 재시도 → fallback.
    teardown: 어댑터/모델 해제 + CUDA 캐시 비움(다음 fold 전 필수).

    cfg.arch 로 조절하는 AR 고유 노브(대칭성 검토 시 여기만 보면 된다):
        gen_temperature      (기본 0.0  — 그리디, 재현성 우선)
        gen_retry_temperature(기본 0.7  — 파싱 실패 재시도 시 샘플링 온도)
        parse_retries        (기본 2    — 대회 규정과 동일: 2회 재시도 후 fallback)
        gen_batch_size       (기본 8)
        score_decimals       (기본 2)
    """

    name = "autoregressive"

    # --- fit -----------------------------------------------------------------
    def fit(self, data: FoldData) -> None:
        import torch
        from torch.utils.data import DataLoader
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   get_cosine_schedule_with_warmup)
        from peft import LoraConfig, get_peft_model

        from .pipeline import set_seed
        cfg = self.cfg
        self._gen = set_seed(cfg.seed)   # multitask와 동일 루틴 (대칭 + 재현)

        # 방어적 누수 재확인 — load_fold가 이미 assert하지만 fit도 data.eval을 만지지 않음을
        # 코드로 못박는다(README '설계 결정' 2번의 이중 방어선 정신).
        assert data.eval is not None  # 존재만 확인, 학습에 사용 금지

        tok = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"  # 학습은 오른쪽 패딩
        self.tokenizer = tok

        # bf16 full-precision + LoRA. multitask.py와 동일한 로딩 정밀도로 맞춘다
        # (대칭성, README '설계 결정' 4번). 양자화(QLoRA)를 쓰면 "아키텍처 차이"에
        # "양자화 차이"가 섞여 비교가 오염된다. 7B는 H100에서 bf16으로 충분히 올라간다.
        dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            cfg.backbone, dtype=dtype,               # tf 5.x: torch_dtype→dtype
            device_map={"": 0}, trust_remote_code=True,
            attn_implementation=cfg.attn_impl,       # multitask와 동일(대칭). "eager"/"sdpa"
        )
        model.config.use_cache = False          # gradient checkpointing과 상충 → 끔
        # 활성화 재계산으로 메모리 절약(정밀도와 무관). cfg로 토글, multitask와 동일 규약.
        # LoRA(동결 백본)라 입력 임베딩에 grad가 흐르도록 enable_input_require_grads도 켠다
        # (checkpointing 켤 때만 필요).
        if cfg.grad_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()

        lora = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_targets), bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

        # 학습 파라미터(LoRA)를 fp32 마스터로 승격 — multitask와 동일한 안정화(대칭성).
        # bf16 LoRA + AdamW eps 언더플로 → NaN을 막는다. forward는 autocast(bf16)로 계산.
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        self.model = model

        # --- 데이터셋: (prompt 마스킹된) 라벨 텐서 리스트 ---
        features = [self._encode_train(ex) for ex in data.train]
        features = [f for f in features if f is not None]
        loader = DataLoader(
            features, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=self._collate, generator=self._gen,   # shuffle 순서 고정
        )

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, eps=1e-6,
        )
        steps_per_epoch = max(1, len(loader) // cfg.grad_accum)
        total_steps = steps_per_epoch * cfg.epochs
        sched = get_cosine_schedule_with_warmup(
            opt, int(total_steps * cfg.warmup_ratio), total_steps,
        )
        use_amp = (model.device.type == "cuda")

        model.train()
        gstep = 0
        n_batches = len(loader)
        trainable = [p for p in model.parameters() if p.requires_grad]
        for epoch in range(cfg.epochs):
            running, n_skipped, n_bad_grad = 0.0, 0, 0
            t_epoch = time.time()
            for i, batch in enumerate(loader):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(**batch)
                if not torch.isfinite(out.loss):     # NaN/inf step 스킵 (가중치 보호)
                    n_skipped += 1
                    opt.zero_grad()
                    continue
                (out.loss / cfg.grad_accum).backward()
                running += out.loss.item()
                if (i + 1) % cfg.grad_accum == 0:
                    # grad 유한성 가드 — 손실 유한해도 NaN grad면 step 건너뜀(파라미터 보호).
                    gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    if torch.isfinite(gnorm):
                        opt.step(); sched.step()
                    else:
                        n_bad_grad += 1
                    opt.zero_grad()
                    gstep += 1
                # 진행률 — epoch 내부 생존 표시.
                if (i + 1) % LOG_EVERY == 0:
                    done = i + 1 - n_skipped
                    rate = (i + 1) / (time.time() - t_epoch)
                    eta = (n_batches - (i + 1)) / rate if rate > 0 else 0
                    print(f"    [fold{self.fold}] epoch {epoch+1}/{cfg.epochs} "
                          f"batch {i+1}/{n_batches} loss={running/max(1,done):.4f} "
                          f"({rate:.2f} it/s, ETA {eta/60:.1f}m)", flush=True)
            n = max(1, len(loader) - n_skipped)
            bad_msg = ""
            if n_skipped or n_bad_grad:
                bad_msg = f" (skipped {n_skipped}, bad_grad {n_bad_grad})"
            dt = time.time() - t_epoch
            print(f"    [fold{self.fold}] epoch {epoch+1}/{cfg.epochs} "
                  f"loss={running/n:.4f}{bad_msg} [{dt/60:.1f}m]", flush=True)

            # 검증 곡선 — 캡한 subset에서 생성→파싱→지표. (AR은 점수도 생성이라 캡 필요)
            if cfg.eval_every_epoch:
                vres, ratios = self._epoch_eval(data.eval)
                rstr = " ".join(f"{C.TRAIT_KOR[t]}={ratios[t]:.2f}" for t in C.TRAITS)
                print(f"    [fold{self.fold}] └ VAL RMSE={vres.rmse:.4f} "
                      f"ρ={vres.spearman:.4f} | std비 {rstr}", flush=True)

        opt.zero_grad(set_to_none=True)
        model.config.use_cache = True
        model.eval()
        # 재현/제출용으로 어댑터 저장
        model.save_pretrained(os.path.join(self.out_dir, "adapter"))
        tok.save_pretrained(os.path.join(self.out_dir, "adapter"))

    def _epoch_eval(self, examples):
        """검증 곡선용 — 캡한 subset을 greedy 생성→파싱→평가. (EvalResult, 영역별 std비)."""
        import statistics as st
        from .metrics import evaluate

        model, tok = self.model, self.tokenizer
        cap = self.cfg.eval_subset
        subset = list(examples)[:cap] if cap and cap > 0 else list(examples)

        prev_cache, was_training = model.config.use_cache, model.training
        model.config.use_cache = True   # 생성 속도 (학습 땐 checkpointing으로 꺼둠)
        model.eval()
        tok.padding_side = "left"
        bs = self.arch_int("gen_batch_size", 8)
        od = self.cfg.arch.get("target_order", "rs")
        raws = self._generate([build_scoring_messages(e, od) for e in subset], 0.0, bs)  # greedy

        preds = []
        for ex, raw in zip(subset, raws):
            parsed = parse_scoring_output(raw)
            if parsed is None:
                preds.append(self.fallback_prediction(ex))
            else:
                sc, rt = parsed
                preds.append(Prediction(id=ex.id, scores=sc, rationales=rt))

        if was_training:
            model.train()
        model.config.use_cache = prev_cache

        res = evaluate(subset, preds)
        ratios = {}
        for t in C.TRAITS:
            ps = st.pstdev([p.scores[t] for p in preds]) if len(preds) > 1 else 0.0
            ts = st.pstdev([e.scores[t] for e in subset]) if len(subset) > 1 else 0.0
            ratios[t] = ps / ts if ts > 0 else 0.0
        return res, ratios

    def _encode_train(self, ex: Example):
        """(prompt, target) → input_ids/labels. prompt 구간 라벨은 -100(손실 제외)."""
        cfg = self.cfg
        tok = self.tokenizer
        prompt_ids = tok.apply_chat_template(
            build_scoring_messages(ex, self.cfg.arch.get("target_order", "rs")),
        tokenize=True, add_generation_prompt=True,
        )
        # tf 5.x: apply_chat_template(tokenize=True)가 BatchEncoding/dict/텐서를 반환할 수 있어
        # list[int]로 정규화한다 (예전엔 항상 list[int]였음).
        if not isinstance(prompt_ids, list):
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids["input_ids"]
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
        # 배치 차원이 씌워져 [[...]] 로 오면 벗긴다.
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]

        target_text = build_target_json(ex, self.arch_int("score_decimals", 2),
                                        self.cfg.arch.get("target_order", "rs"))
        target_ids = tok(target_text, add_special_tokens=False)["input_ids"]
        if tok.eos_token_id is not None:
            target_ids = target_ids + [tok.eos_token_id]

        # 길이 초과 시 target은 절대 자르지 않고 prompt 앞부분을 왼쪽에서 자른다.
        # (근거→점수 정답이 잘리면 학습 신호 자체가 깨지므로)
        overflow = len(prompt_ids) + len(target_ids) - cfg.max_len
        if overflow > 0:
            if overflow >= len(prompt_ids):
                return None  # target만으로도 max_len 초과하는 비정상 샘플은 버린다
            prompt_ids = prompt_ids[overflow:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)
        return {"input_ids": input_ids, "labels": labels}

    def _collate(self, feats):
        import torch
        pad_id = self.tokenizer.pad_token_id
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * n)
            labels.append(f["labels"] + [-100] * n)
            attn.append([1] * len(f["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    # --- predict -------------------------------------------------------------
    def predict(self, examples: Sequence[Example]) -> List[Prediction]:
        # 생성은 왼쪽 패딩(마지막 토큰 정렬). torch는 _generate 안에서만 쓴다 →
        # _generate를 스텁하면 이 retry/fallback 제어흐름은 GPU 없이 검증 가능.
        # 재시딩 — 재시도 샘플링(temp>0) 재현. torch 없는 자체테스트에선 건너뛴다.
        try:
            from .pipeline import set_seed
            set_seed(self.cfg.seed)
        except ImportError:
            pass
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"

        bs = self.arch_int("gen_batch_size", 8)
        greedy_t = self.arch_float("gen_temperature", 0.0)
        retry_t = self.arch_float("gen_retry_temperature", 0.7)
        retries = self.arch_int("parse_retries", 2)
        od = self.cfg.arch.get("target_order", "rs")   # 학습 타깃과 반드시 같은 순서

        results: Dict[str, Prediction] = {}
        pending = list(examples)

        total = len(examples)
        for attempt in range(retries + 1):
            if not pending:
                break
            temp = greedy_t if attempt == 0 else retry_t
            print(f"    [fold{self.fold}] predict 시도 {attempt}: {len(pending)}건 생성 중 "
                  f"(temp={temp})...", flush=True)
            t0 = time.time()
            raws = self._generate([build_scoring_messages(e, od) for e in pending],
                                  temp, bs)
            still = []
            for ex, raw in zip(pending, raws):
                parsed = parse_scoring_output(raw)
                if parsed is None:
                    still.append(ex)
                else:
                    scores, rationales = parsed
                    results[ex.id] = Prediction(id=ex.id, scores=scores, rationales=rationales)
            print(f"    [fold{self.fold}] 시도 {attempt} 완료: "
                  f"{total - len(still)}/{total} 파싱 성공 ({time.time()-t0:.0f}s)", flush=True)
            pending = still

        # 최종 실패는 fallback(중앙값·빈 근거) — 대회 0점 처리와 대응(제외는 과대추정)
        if pending:
            print(f"    [fold{self.fold}] 최종 파싱 실패 {len(pending)}/{len(examples)}건 → fallback")
        for ex in pending:
            results[ex.id] = self.fallback_prediction(ex)

        return [results[e.id] for e in examples]

    def _generate(self, batch_messages, temperature: float, bs: int) -> List[str]:
        import torch

        tok, model = self.tokenizer, self.model
        prompts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=0.95)

        outs: List[str] = []
        total, t0 = len(prompts), time.time()
        use_amp = (model.device.type == "cuda")
        for s in range(0, total, bs):
            chunk = prompts[s:s + bs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=self.cfg.max_len).to(model.device)
            # fp32 마스터 파라미터 → 추론도 autocast(bf16)로 계산 경로 정합.
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                 enabled=use_amp):
                gen = model.generate(**enc, **gen_kwargs)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(tok.batch_decode(new, skip_special_tokens=True))
            done = min(s + bs, total)
            rate = done / max(1e-6, time.time() - t0)
            print(f"      생성 {done}/{total} ({rate:.1f} ex/s, "
                  f"ETA {(total-done)/rate/60:.1f}m)", flush=True)
        return outs

    # --- teardown ------------------------------------------------------------
    def teardown(self) -> None:
        import torch
        for attr in ("model", "tokenizer"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- helpers -------------------------------------------------------------
    def arch_int(self, key: str, default: int) -> int:
        return int(self.cfg.arch.get(key, default))

    def arch_float(self, key: str, default: float) -> float:
        return float(self.cfg.arch.get(key, default))


# ===========================================================================
# 4. 자체 테스트 — GPU 없이 프롬프트/타깃/파서만 검증 (rationale_synthesis_utils 방식)
# ===========================================================================
if __name__ == "__main__":
    class _Ex:
        id = "T1"
        prompt = "인공지능 시대에 인간의 역할은 무엇인가?"
        essay = "인공지능이 발전하면서 인간의 역할이 축소된다는 우려가 있다. 그러나 창의성은..."
        scores = {"content": 3.5, "organization": 3.0, "expression": 4.25}
        rationales = {
            "content": "주제 이해는 명확하나 반론 검토가 얕아 3.5점이 적절하다.",
            "organization": "서론-본론 연결이 다소 느슨하여 3.0점이다.",
            "expression": "어휘가 정확하고 문장이 유창하여 4.25점을 받았다.",
        }

    ex = _Ex()

    print("=== 1. 채점 프롬프트 (정답 점수 미포함 확인) ===")
    msgs = build_scoring_messages(ex)
    user = msgs[1]["content"]
    assert "학생 글" in user and "채점 기준" in user
    for s in ("3.5", "3.0", "4.25"):
        assert s not in user, f"정답 점수 {s}가 프롬프트에 노출됨 — 누수"
    print(f"  ✅ system {len(msgs[0]['content'])}자 / user {len(user)}자, 정답 점수 미노출")

    print("\n=== 2. 학습 target (rationale→score 순서 = CoT 강제) ===")
    tgt = build_target_json(ex)
    assert tgt.index("rationale") < tgt.index("score"), "score가 rationale보다 앞 — CoT 깨짐"
    print(f"  {tgt[:90]}...")
    print("  ✅ 근거가 점수보다 앞에 온다")

    print("\n=== 3. 파서 라운드트립 (target을 그대로 파싱) ===")
    parsed = parse_scoring_output(tgt)
    assert parsed is not None
    sc, rt = parsed
    assert sc == {"content": 3.5, "organization": 3.0, "expression": 4.25}, sc
    assert all(len(rt[t]) > 5 for t in sc)
    print(f"  ✅ scores={sc}")

    print("\n=== 4. 관대 파싱 (think 블록/코드펜스/후행콤마/납작구조/범위클립) ===")
    p = parse_scoring_output(
        '<think>고민...</think>```json\n'
        '{"content":{"rationale":"근거.","score":6},'   # 6 → 5.0 클립
        '"organization":{"score":"약 2.5점","rationale":"근거."},'  # 문자열 속 숫자
        '"expression":4,}\n```')                          # 납작 구조 + 후행콤마
    assert p is not None, "관대 파싱 실패"
    sc2, _ = p
    assert sc2 == {"content": 5.0, "organization": 2.5, "expression": 4.0}, sc2
    print(f"  ✅ scores={sc2}")

    print("\n=== 5. 실패 케이스는 None (재시도/ fallback 유발) ===")
    assert parse_scoring_output("설명만 있고 JSON 없음") is None
    assert parse_scoring_output('{"content":{"score":3}}') is None            # 영역 누락
    assert parse_scoring_output('{"content":{"rationale":"x"},'
                                '"organization":{"score":3},'
                                '"expression":{"score":3}}') is None          # content score 없음
    print("  ✅ 결측/무JSON 모두 거부")

    print("\n=== 6. predict 재시도→fallback 제어흐름 (_generate 스텁, GPU 불필요) ===")
    import tempfile
    from awes_common.data import Example as _E
    from awes_common.pipeline import TrainConfig as _TC

    exs = [
        _E(id="A", prompt_num="Q1", prompt="p", essay="AAA-essay",
           scores={"content": 0, "organization": 0, "expression": 0}),  # 재시도에서 성공
        _E(id="B", prompt_num="Q1", prompt="p", essay="BBB-essay",
           scores={"content": 0, "organization": 0, "expression": 0}),  # 끝까지 실패 → fallback
    ]
    pipe = AutoregressivePipeline(_TC(arch={"parse_retries": 2}), fold=0,
                                  out_dir=tempfile.mkdtemp())
    pipe.tokenizer = None  # padding_side 접근 가드 확인
    calls = {"n": 0}
    good = '{"content":{"rationale":"r","score":3.5},' \
           '"organization":{"rationale":"r","score":3.0},' \
           '"expression":{"rationale":"r","score":4.0}}'

    def _fake_generate(batch_messages, temperature, bs):
        # 정체성(essay)으로 판정 — pending 배치가 줄어도 B가 위치로 성공하지 않게.
        calls["n"] += 1
        out = []
        for m in batch_messages:
            essay_in = m[1]["content"]
            # attempt0(첫 호출)은 전부 실패, 이후 A만 성공, B는 영원히 실패.
            if calls["n"] >= 2 and "AAA-essay" in essay_in:
                out.append(good)
            else:
                out.append("garbage")
        return out

    pipe._generate = _fake_generate
    preds = pipe.predict(exs)
    assert len(preds) == len(exs), "모든 입력에 예측이 반환돼야 한다"
    assert [p.id for p in preds] == ["A", "B"], "입력 순서 보존"
    assert preds[0].scores["content"] == 3.5, "A는 재시도에서 파싱 성공해야"
    mid = (C.SCORE_MIN + C.SCORE_MAX) / 2.0
    assert preds[1].scores == {t: mid for t in C.TRAITS}, "B는 fallback(중앙값)"
    assert preds[1].rationales == {t: "" for t in C.TRAITS}, "fallback 근거는 빈 문자열"
    assert calls["n"] == 3, f"attempt0 + 재시도2 = 3회 생성이어야 (실측 {calls['n']})"
    print(f"  ✅ {calls['n']}회 생성, A=파싱성공 B=fallback, 순서/완전성 보존")

    print("\n" + "=" * 50 + "\n✅ autoregressive 순수 로직 전체 통과")
