# -*- coding: utf-8 -*-
"""
run_experiment.py — 5-fold 실험 런처 (tmux 안에서 실행 권장).

사용법:
    python3 run_experiment.py multitask
    python3 run_experiment.py autoregressive
    python3 run_experiment.py multitask --batch 4 --grad-accum 4
    python3 run_experiment.py multitask --folds 1,2,3,4   # 특정 fold만

두 아키텍처가 같은 하네스·같은 TrainConfig를 쓰도록 이 파일 하나로 통일한다
(복붙 실수로 대칭이 깨지는 것 방지). resume 기본 on이라 중단 후 재실행하면
끝난 fold는 건너뛰고 이어서 돈다.

실행 전 셸에서:
    export CUDA_VISIBLE_DEVICES=4                       # 실행 직전 nvidia-smi로 재확인
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

import argparse
import os

from awes_common import run_cv, TrainConfig
from awes_common.multitask import MultiTaskPipeline
from awes_common.autoregressive import AutoregressivePipeline

PIPELINES = {
    "multitask": MultiTaskPipeline,
    "autoregressive": AutoregressivePipeline,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=list(PIPELINES), help="실행할 아키텍처")
    ap.add_argument("--batch", type=int, default=None, help="per-device batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="gradient accumulation")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--backbone", type=str, default=None,
                    help="백본 HF id. 예: Qwen/Qwen3.5-9B (기본: Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--lora-targets", type=str, default=None,
                    help="LoRA 대상 모듈 콤마구분. 예: q_proj,k_proj,v_proj,o_proj "
                         "(MoE/다른 아키텍처는 모듈명이 달라 조정 필요)")
    ap.add_argument("--gen-batch", type=int, default=None, help="추론 생성 배치(arch.gen_batch_size)")
    ap.add_argument("--rank-weight", type=float, default=None,
                    help="멀티태스크 순위 손실 가중(arch.rank_weight, Spearman 특화). 0=off")
    ap.add_argument("--rank-hard-gap", type=float, default=None,
                    help="소격차 쌍만 rank 손실에 사용(arch.rank_hard_gap). 정규화 [0,1] 단위 "
                         "= 원점수/4. 예: 0.05 → |Δgold| ≤ 0.2점 쌍만. 0=전체(현행)")
    ap.add_argument("--rank-margin-mode", choices=["fixed", "gap"], default=None,
                    help="rank margin 해석(arch.rank_margin_mode). gap=min(margin,|Δgold|)로 "
                         "정답 격차 이상은 요구하지 않음. --rank-hard-gap과 함께 쓸 것")
    ap.add_argument("--init-from", type=str, default=None,
                    help="사전학습 산출물 디렉토리(adapter/+heads.pt)에서 LoRA·헤드 초기화"
                         "(arch.init_from). Stage A→B 연결. 예: runs/pt14b_aihub/pretrain")
    ap.add_argument("--init-no-head", action="store_true",
                    help="--init-from 시 LoRA만 이어받고 회귀 헤드는 새로 시작"
                         "(arch.init_head=False). 사전학습 헤드가 포화됐을 때")
    ap.add_argument("--gen-off", action="store_true",
                    help="생성 손실 완전 off(arch.gen_off) — 근거 라벨 없는 외부 사전학습용")
    ap.add_argument("--target-order", choices=["rs", "sr"], default=None,
                    help="AR 생성 순서(arch.target_order). rs=근거→점수(기본), "
                         "sr=점수→근거. 논문 트랙 Phase 2 축 — PAPER_TRACK.md 참조")
    ap.add_argument("--sparse-ce", action="store_true",
                    help="생성 CE를 라벨 위치에서만 계산(arch.sparse_ce) — 긴 입력/대배치용 "
                         "메모리 해금(~40GB→~3GB). 손실은 수학적으로 동일")
    ap.add_argument("--corr-weight", type=float, default=None,
                    help="멀티태스크 상관 손실 가중(arch.corr_weight, RMSE·Spearman 동시). 0=off")
    ap.add_argument("--gen-scores", action="store_true",
                    help="멀티태스크 생성 헤드가 점수까지 JSON 출력(시나리오 B 제출 대응)")
    ap.add_argument("--pooling", choices=["last", "mean", "attention"], default=None,
                    help="회귀 헤드 풀링(arch.pooling). last(기본)/mean/attention")
    ap.add_argument("--gen-uncertainty", choices=["regression", "kendall_cls", "classification"],
                    default=None, help="생성 항 불확실성 계수 형태(arch.gen_uncertainty, §1)")
    ap.add_argument("--attn-impl", choices=["eager", "sdpa", "flash_attention_2"], default=None,
                    help="어텐션 구현. flash_attention_2=저메모리→큰 배치 가능(도커 필요). "
                         "단 이 스택서 sdpa NaN 전례 있어 1ep 스모크로 skipped/bad_grad 확인 필수")
    ap.add_argument("--soft-weight", type=float, default=None,
                    help="soft-Spearman 손실 가중(arch.soft_weight). 미분가능 순위로 지표를 "
                         "직접 최적화. rank/corr와 다른 손실 계열이라 앙상블 다양성에도 기여. 0=off")
    ap.add_argument("--soft-tau", type=float, default=None,
                    help="soft rank 온도(arch.soft_tau, 기본 0.1). 작을수록 실제 순위에 근접")
    ap.add_argument("--contrastive-dir", type=str, default=None,
                    help="합성 대조쌍 경로(make_contrastive_pairs build --out). 지정 시 순위 손실 주입")
    ap.add_argument("--contrastive-weight", type=float, default=None,
                    help="대조 손실 가중(arch.contrastive_weight). 0=off")
    ap.add_argument("--contrastive-margin", type=float, default=None,
                    help="대조 margin(arch.contrastive_margin, 기본 0.05). gap배로 확대")
    ap.add_argument("--contrastive-batch-pairs", type=int, default=None,
                    help="optim step당 대조쌍 수(arch.contrastive_batch_pairs, 기본 16)")
    ap.add_argument("--contrastive-max-gap", type=int, default=None,
                    help="대조쌍 gap 상한 필터(arch.contrastive_max_gap). 미지정=전체")
    ap.add_argument("--contrastive-traits", type=str, default=None,
                    help="대조에 쓸 영역만 콤마로(예: expression 또는 content,expression)")
    ap.add_argument("--seed", type=int, default=None,
                    help="난수 시드(기본 42). 동일 config·다른 seed = seed 앙상블(E2)용")
    ap.add_argument("--no-checkpoint", action="store_true", help="gradient checkpointing 끄기")
    ap.add_argument("--folds", type=str, default=None, help="예: 1,2,3,4 (기본: 0~4 전체)")
    ap.add_argument("--no-resume", action="store_true", help="완료 fold도 재학습")
    ap.add_argument("--run-name", type=str, default=None,
                    help="결과 저장 이름(runs/<name>). 스윕에서 config별 분리 저장용. "
                         "기본: 아키텍처 이름(multitask/autoregressive)")
    args = ap.parse_args()

    # TrainConfig — 지정 안 한 값은 기본값(양쪽 공유). 대칭성 위해 두 아키텍처에 같은 값 줄 것.
    overrides = {}
    if args.batch is not None:
        overrides["batch_size"] = args.batch
    if args.grad_accum is not None:
        overrides["grad_accum"] = args.grad_accum
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.lora_r is not None:
        overrides["lora_r"] = args.lora_r
    if args.lora_alpha is not None:
        overrides["lora_alpha"] = args.lora_alpha
    if args.backbone is not None:
        overrides["backbone"] = args.backbone
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.lora_targets is not None:
        overrides["lora_targets"] = tuple(
            x.strip() for x in args.lora_targets.split(",") if x.strip())
    arch = {}
    if args.gen_batch is not None:
        arch["gen_batch_size"] = args.gen_batch
    if args.rank_weight is not None:
        arch["rank_weight"] = args.rank_weight
    if args.init_from:
        arch["init_from"] = args.init_from
    if args.init_no_head:
        arch["init_head"] = False
    if args.gen_off:
        arch["gen_off"] = True
    if args.target_order is not None:
        arch["target_order"] = args.target_order
    if args.sparse_ce:
        arch["sparse_ce"] = True
    if args.rank_hard_gap is not None:
        arch["rank_hard_gap"] = args.rank_hard_gap
    if args.rank_margin_mode is not None:
        arch["rank_margin_mode"] = args.rank_margin_mode
    if args.corr_weight is not None:
        arch["corr_weight"] = args.corr_weight
    if args.soft_weight is not None:
        arch["soft_weight"] = args.soft_weight
    if args.soft_tau is not None:
        arch["soft_tau"] = args.soft_tau
    if args.gen_scores:
        arch["gen_scores"] = True
    if args.pooling is not None:
        arch["pooling"] = args.pooling
    if args.gen_uncertainty is not None:
        arch["gen_uncertainty"] = args.gen_uncertainty
    if args.contrastive_dir is not None:
        arch["contrastive_dir"] = args.contrastive_dir
    if args.contrastive_weight is not None:
        arch["contrastive_weight"] = args.contrastive_weight
    if args.contrastive_margin is not None:
        arch["contrastive_margin"] = args.contrastive_margin
    if args.contrastive_batch_pairs is not None:
        arch["contrastive_batch_pairs"] = args.contrastive_batch_pairs
    if args.contrastive_max_gap is not None:
        arch["contrastive_max_gap"] = args.contrastive_max_gap
    if args.contrastive_traits is not None:
        arch["contrastive_traits"] = [t.strip() for t in args.contrastive_traits.split(",") if t.strip()]
    if arch:
        overrides["arch"] = arch
    if args.attn_impl is not None:
        overrides["attn_impl"] = args.attn_impl
    if args.no_checkpoint:
        overrides["grad_checkpointing"] = False
    cfg = TrainConfig(**overrides)

    folds = None
    if args.folds:
        folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]

    from awes_common import config as C
    eff_batch = cfg.batch_size * cfg.grad_accum
    print(f"=== {args.arch} {C.N_FOLDS}-fold ===")
    print(f"  backbone={cfg.backbone}  lora_r={cfg.lora_r} targets={list(cfg.lora_targets)}")
    print(f"  AWES_NFOLDS={os.environ.get('AWES_NFOLDS', '(미설정→기본 4)')}  N_FOLDS={C.N_FOLDS}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(미설정!)')}")
    print(f"  batch_size={cfg.batch_size} × grad_accum={cfg.grad_accum} "
          f"= 유효배치 {eff_batch}  epochs={cfg.epochs}  lr={cfg.lr}")
    print(f"  grad_checkpointing={cfg.grad_checkpointing}  max_len={cfg.max_len}")
    print(f"  resume={not args.no_resume}  folds={folds or 'all'}")

    # run_cv가 fold별 로그와 최종 CV 요약을 직접 출력한다 (중복 print 금지).
    run_cv(
        PIPELINES[args.arch], cfg,
        folds=folds,
        run_name=args.run_name,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

