# -*- coding: utf-8 -*-
"""
contrastive.py — 순위 대조쌍을 멀티태스크 학습에 주입하는 어댑터.

`make_contrastive_pairs.py`가 만든 (variants, pairs)를 읽어, 학습 중
**pairwise margin ranking 손실**만 추가한다. HANDOFF §8 "1순위" 레버의 학습측 절반.

계약 (make_contrastive_pairs와 동일 — 여기서도 타입으로 강제한다)
------------------------------------------------------------------
1. 합성본은 절대점수(MSE)에 절대 닿지 않는다.
   → 모델 forward를 **labels=None / score_targets=None** 으로 호출한다.
     MultiTaskModel.forward는 이 경우 score_pred만 반환하고 조기 리턴하므로
     (multitask.py의 `if labels is None or score_targets is None: return result`),
     reg_loss/gen_loss를 계산할 경로 자체가 없다. 규율이 아니라 구조로 보장.
2. 순위 주장은 **열화한 영역 차원에만** 건다.
   → 손실은 pair의 trait 인덱스 하나만 인덱싱한다(다른 영역엔 gradient 0).
3. fold 누수 차단.
   → load_pairs(exclude_fold=self.fold): 변형은 원본 _fold를 상속하므로
     eval fold에 속한 원본에서 파생된 쌍이 전부 빠진다.

손실 정의
---------
    margin hinge:  relu(margin * gap - (pred_high[trait] - pred_low[trait]))
  · pred_*  : 정규화 점수 [0,1] (학습 내부 스케일, _norm_score와 동일)
  · gap     : 레벨 차(1~3). margin을 gap배로 키워 **더 벌어진 쌍일수록 더 큰 간격**을
              요구한다. 반대로 gap=1(원본 vs 약한 열화)은 약한 제약 → 미묘한 쌍이
              과도한 penalty를 만들지 않게 한다(편집비율이 아니라 level로 강도를 본다).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from . import config as C
from .data import Example


TRAIT_IDX = {t: i for i, t in enumerate(C.TRAITS)}


# ---------------------------------------------------------------------------
# 1. 산출물 로딩 (make_contrastive_pairs.load_pairs의 얇은 재구현 — 순환 import 회피)
# ---------------------------------------------------------------------------
def load_pairs(out_dir: str, exclude_fold: Optional[int] = None,
               traits: Optional[Sequence[str]] = None,
               max_gap: Optional[int] = None,
               ) -> Tuple[Dict[str, Dict], List[Dict]]:
    """(id→variant, pairs). exclude_fold=k 로 fold-k 원본 파생 레코드를 전부 제외."""
    import json

    variants: Dict[str, Dict] = {}
    with open(os.path.join(out_dir, "variants.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            v = json.loads(line)
            if exclude_fold is not None and v.get("_fold") == exclude_fold:
                continue
            variants[v["id"]] = v

    pairs: List[Dict] = []
    with open(os.path.join(out_dir, "pairs.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            if p["high_id"] not in variants or p["low_id"] not in variants:
                continue                                   # exclude_fold로 빠진 쪽
            if traits and p["trait"] not in traits:
                continue
            if max_gap is not None and p["gap"] > max_gap:
                continue
            pairs.append(p)
    return variants, pairs


# ---------------------------------------------------------------------------
# 2. 순수 손실 함수 (GPU/transformers 없이 CPU에서 단위 테스트 가능)
# ---------------------------------------------------------------------------
def pair_rank_loss(score_pred, high_idx, low_idx, trait_idx, gap, margin: float):
    """
    score_pred: [M,3] — 배치 내 M개 고유 변형의 정규화 예측점수.
    high_idx/low_idx/trait_idx/gap: [B] long/float 텐서 — 각 쌍의 인덱스.
    반환: 스칼라. 쌍이 없으면 0.

    high가 low보다 해당 trait에서 margin*gap 이상 높지 않으면 hinge penalty.
    """
    import torch

    if high_idx.numel() == 0:
        return score_pred.new_zeros(())
    ph = score_pred[high_idx, trait_idx]                   # [B]
    pl = score_pred[low_idx, trait_idx]                    # [B]
    return torch.relu(margin * gap - (ph - pl)).mean()


# ---------------------------------------------------------------------------
# 3. 배처 — 쌍을 미니배치로, 고유 변형만 forward하도록 인덱스 재사상
# ---------------------------------------------------------------------------
def _variant_to_example(vid: str, v: Dict) -> Example:
    """build_input_text가 읽는 최소 Example. 점수는 forward에 안 쓰이나 dataclass 필수라 더미."""
    return Example(
        id=vid, prompt_num=v.get("prompt_num", ""), prompt=v.get("prompt", ""),
        essay=v["essay"], scores={t: 1.0 for t in C.TRAITS}, rationales=None,
    )


class PairBatcher:
    """
    쌍 리스트를 셔플·순환하며 배치를 낸다. 각 배치는:
      · input_ids/attention_mask/pool_pos : 배치에 등장하는 **고유 변형 M개**만 (중복 제거)
      · high_idx/low_idx : [B] — M개 안에서의 위치
      · trait_idx/gap    : [B]
    한 변형이 여러 쌍에 걸쳐도 forward는 1회만 → 연산 절약.

    결정성: torch.Generator(seed)로 셔플. 같은 seed면 배치 순서가 재현된다.
    """

    def __init__(self, variants: Dict[str, Dict], pairs: List[Dict], tokenizer,
                 max_len: int, batch_pairs: int, seed: int,
                 build_input_text, pad_id: int):
        import torch

        self.pairs = pairs
        self.batch_pairs = batch_pairs
        self.pad_id = pad_id
        self._torch = torch

        # 고유 변형 essay를 한 번씩만 토크나이즈 (입력 마지막 실토큰이 pool 위치).
        self._ids: Dict[str, List[int]] = {}
        need = {p["high_id"] for p in pairs} | {p["low_id"] for p in pairs}
        for vid in need:
            inp = build_input_text(_variant_to_example(vid, variants[vid]))
            toks = tokenizer(inp, add_special_tokens=True).input_ids[:max_len]
            self._ids[vid] = toks

        self._gen = torch.Generator().manual_seed(seed)
        self._order: List[int] = []
        self._cursor = 0

    def __len__(self) -> int:
        return max(1, len(self.pairs) // self.batch_pairs)

    def _refill(self):
        perm = self._torch.randperm(len(self.pairs), generator=self._gen).tolist()
        self._order = perm
        self._cursor = 0

    def next_batch(self, device):
        """다음 배치를 device 텐서로. 쌍이 소진되면 자동 재셔플(무한 순환)."""
        torch = self._torch
        if self._cursor + self.batch_pairs > len(self._order):
            self._refill()
        sel = self._order[self._cursor:self._cursor + self.batch_pairs]
        self._cursor += self.batch_pairs

        uniq: Dict[str, int] = {}
        hi, lo, tr, gp = [], [], [], []
        for pi in sel:
            p = self.pairs[pi]
            for vid in (p["high_id"], p["low_id"]):
                if vid not in uniq:
                    uniq[vid] = len(uniq)
            hi.append(uniq[p["high_id"]])
            lo.append(uniq[p["low_id"]])
            tr.append(TRAIT_IDX[p["trait"]])
            gp.append(float(p["gap"]))

        # 고유 변형 M개를 우측 패딩으로 콜레이트 (학습/predict와 동일 위치 규약).
        order = sorted(uniq, key=lambda k: uniq[k])
        seqs = [self._ids[v] for v in order]
        maxlen = max(len(s) for s in seqs)
        input_ids, attn, pool_pos = [], [], []
        for s in seqs:
            pad = maxlen - len(s)
            input_ids.append(s + [self.pad_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
            pool_pos.append(len(s) - 1)

        L = lambda x: torch.tensor(x, dtype=torch.long, device=device)
        return {
            "input_ids": L(input_ids), "attention_mask": L(attn),
            "pool_pos": L(pool_pos),
            "high_idx": L(hi), "low_idx": L(lo), "trait_idx": L(tr),
            "gap": torch.tensor(gp, dtype=torch.float, device=device),
        }


def make_batcher(arch: Dict, fold: int, tokenizer, max_len: int, seed: int,
                 build_input_text, pad_id: int) -> Optional["PairBatcher"]:
    """
    arch["contrastive_dir"]가 있으면 배처를, 없으면 None을 반환한다(기능 off).
    호출측(multitask.fit)이 None이면 대조 손실 경로 전체를 건너뛴다.
    """
    out_dir = arch.get("contrastive_dir")
    if not out_dir:
        return None
    variants, pairs = load_pairs(
        out_dir, exclude_fold=fold,
        traits=arch.get("contrastive_traits"),
        max_gap=arch.get("contrastive_max_gap"),
    )
    if not pairs:
        print(f"    [contrastive] fold{fold}: 쌍 0건 (경로/누수필터 확인) → 비활성", flush=True)
        return None
    print(f"    [contrastive] fold{fold}: 변형 {len(variants)} · 쌍 {len(pairs)} 로드, "
          f"w={arch.get('contrastive_weight')} margin={arch.get('contrastive_margin')} "
          f"batch_pairs={arch.get('contrastive_batch_pairs')}", flush=True)
    return PairBatcher(variants, pairs, tokenizer, max_len,
                       int(arch.get("contrastive_batch_pairs", 16)), seed,
                       build_input_text, pad_id)
