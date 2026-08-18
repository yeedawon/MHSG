#!/usr/bin/env bash
# run_oneshot.sh — 일회성 인스턴스에서 Phase 1(게이트)을 끝까지 돌린다.
#
# 전제: 스토리지를 초기화한 개인 인스턴스. 팀 공유 캐시가 없고, 인스턴스가 죽으면
#       컨테이너 디스크의 산출물은 전부 사라진다. 그래서 이 스크립트는
#         · 학습 전에 데이터·디스크를 먼저 검증하고 (늦게 죽는 것보다 일찍 죽는 게 낫다)
#         · fold가 끝날 때마다 예측을 OUT으로 빼낸다 (중간에 죽어도 거기까지는 남는다)
#
# 사용:
#   BACKBONE=Qwen/Qwen3-1.7B \
#   AWES_ROOT=/path/to/data \
#   OUT=/path/to/persistent \
#     bash run_oneshot.sh
#
#   ARMS="par" bash run_oneshot.sh      # 한 arm만
#   DRY=1 bash run_oneshot.sh           # 계획만 출력
#
# 재개: 같은 명령을 다시 실행하면 완료된 fold는 건너뛴다(run_experiment.py 기본 동작).
#       단 인스턴스가 초기화됐다면 OUT에서 runs/를 먼저 되돌려 놓아야 한다.
set -euo pipefail

BACKBONE="${BACKBONE:?BACKBONE을 지정하세요. 예: BACKBONE=Qwen/Qwen3-1.7B}"
AWES_ROOT="${AWES_ROOT:?AWES_ROOT(데이터 경로)를 지정하세요}"
OUT="${OUT:?OUT(영속 저장 경로)을 지정하세요 — 인스턴스 디스크는 초기화되면 사라집니다}"
ARMS="${ARMS:-par par_nogen}"
# 스케일을 여러 개 돌릴 때 run 이름이 겹치면 앞 결과를 덮어쓴다. 백본 크기 등으로
# 태그를 붙여 분리한다. 예: TAG=9b → runs/9b_par, runs/9b_par_nogen
TAG="${TAG:-}"
DRY="${DRY:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# torch가 시스템 파이썬이 아니라 conda 등에 있는 이미지가 있다. 그런 환경에서
# `python3`를 그대로 쓰면 ModuleNotFoundError로 죽는다 — 인터프리터를 덮어쓸 수 있게 둔다.
#   예: PY=/opt/conda/bin/python bash run_oneshot.sh
PY="${PY:-python3}"

# 통제 변수 — arm 간 차이가 한 축뿐이어야 한다. 여기를 바꾸면 전 arm을 다시 돌려야 한다.
EPOCHS="${EPOCHS:-4}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-4}"
LORA_R="${LORA_R:-16}"
SEED="${SEED:-42}"
# max_len은 CLI 인자가 아니라 TrainConfig 기본값(2048)이다. 바꾸려면 코드를 고쳐야 하고,
# 그러면 전 arm을 다시 돌려야 한다.

export AWES_ROOT
# config.require_gpu_pinned()는 공유 서버용 가드다 — 미지정이면 남의 GPU를 물거나
# CPU로 떨어지는 사고를 막는다. VESSL은 할당된 GPU만 컨테이너에 보여 항상 0이므로
# 여기서는 0을 기본값으로 준다. 🔴 공유 서버에서 돌린다면 반드시 명시적으로 지정할 것.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$OUT/hf}"      # 캐시도 OUT에 두면 재실행 시 재다운로드를 피한다
# 🔴 산출물 위치를 명시한다. 미지정이면 config.runs_dir()가 $AWES_ROOT/runs를 쓰는데,
# AWES_ROOT는 데이터 경로라 결과가 데이터 폴더 안에 섞여 들어간다(2026-08-14 혼선).
export AWES_RUNS="${AWES_RUNS:-$OUT/runs}"

COMMON=(--backbone "$BACKBONE" --epochs "$EPOCHS" --lr "$LR"
        --batch "$BATCH" --grad-accum "$ACCUM"
        --lora-r "$LORA_R" --lora-alpha $((LORA_R * 2))
        --seed "$SEED" --sparse-ce)

echo "=================================================================="
echo " MHSG Phase 1 — 일회성 실행"
echo "   백본     $BACKBONE"
echo "   데이터   $AWES_ROOT"
echo "   반출     $OUT"
echo "   산출물   $AWES_RUNS"
echo "   캐시     $HF_HOME"
echo "   arm      $ARMS${TAG:+  (태그 $TAG)}"
echo "   GPU      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "   파이썬   $($PY -c 'import sys;print(sys.executable)' 2>/dev/null || echo "$PY (실행 불가)")"
echo "   통제     ep=$EPOCHS lr=$LR batch=$BATCH×$ACCUM r=$LORA_R seed=$SEED (max_len=2048 고정)"
echo "=================================================================="

# --- 1) 데이터 검증 — 여기서 막히면 학습 시간을 버리지 않는다 ------------------
echo "[1/4] 데이터 배치 검증"
if ! "$PY" "$REPO/check_data.py"; then
  echo "❌ 데이터 배치 문제 — 학습을 시작하지 않습니다."
  exit 1
fi

# --- 2) 디스크 — 백본을 받다 중간에 터지는 게 가장 흔한 실패다 ------------------
echo "[2/4] 디스크 확인"
mkdir -p "$OUT" "$HF_HOME"
avail_gb=$(df -Pk "$OUT" | awk 'NR==2 {print int($4/1024/1024)}')
echo "   $OUT 여유 ${avail_gb}GB"
if [ "${avail_gb:-0}" -lt 30 ]; then
  echo "   ⚠️ 30GB 미만입니다. 백본 가중치 + 어댑터 + 캐시가 들어가야 합니다."
  echo "      더 작은 백본을 쓰거나 여유 있는 경로를 OUT으로 주세요."
fi

# --- 3) 학습 — fold가 끝날 때마다 반출 ----------------------------------------
sync_out() {
  # AWES_RUNS가 이미 OUT 아래면 복사할 게 없다(rsync가 같은 경로면 no-op).
  # 다른 곳을 가리킬 때만 실제로 반출된다.
  [ "$AWES_RUNS" = "$OUT/runs" ] && return 0
  # 예측·config·로그만 뺀다. 어댑터는 크고, 재평가에는 예측만 있으면 된다.
  mkdir -p "$OUT/runs"
  rsync -a --prune-empty-dirs \
        --include='*/' \
        --include='predictions.jsonl' --include='config.json' --include='cv_result.json' \
        --exclude='*' \
        "$AWES_RUNS/" "$OUT/runs/" 2>/dev/null || true
}

# 넘길 플래그가 실제로 존재하는지 --help로 대조한다. 없는 인자를 주면 argparse가
# 학습 시작 직전에 죽는데, 8회 학습 큐에서는 그걸 늦게 발견하게 된다(2026-08-14 실사고).
help_txt="$("$PY" "$REPO/run_experiment.py" --help 2>&1 || true)"
bad=""
for tok in "${COMMON[@]}" --gen-off --reg-off --init-log-var-gen --target-order; do
  case "$tok" in
    --*) grep -q -- "$tok" <<< "$help_txt" || bad="$bad $tok" ;;
  esac
done
if [ -n "$bad" ]; then
  echo "❌ run_experiment.py가 모르는 인자:$bad"
  echo "   run_experiment.py --help 와 대조해 스크립트를 고치세요."
  exit 1
fi
echo "   ✅ 인자 유효성 확인"

echo "[3/4] 학습"
for arm in $ARMS; do
  # arm 이름이 곧 설정이다 — 이름만 보고 무슨 실험인지 알 수 있어야 한다.
  #   par        생성 헤드 on (가중 0.5)
  #   par_nogen  생성 손실 완전 off
  #   par_g<N>   생성 항 σ² 초기값 log=N → 비중 하향 (2≈0.068, 4≈0.009)
  #   par_genonly 회귀 손실 off → 근거 전용 어댑터 (분업 구조의 근거 절반)
  #   seq_rs/seq_sr  순차 생성 arm — 아키텍처가 autoregressive로 바뀐다
  #                  rs=근거→점수, sr=점수→근거(근거가 점수를 조건으로 받는다)
  arch=multitask
  case "$arm" in
    par_nogen)     extra=(--gen-off) ;;
    par_genonly)   extra=(--reg-off) ;;
    par_g*)        extra=(--init-log-var-gen "${arm#par_g}") ;;
    seq_rs|seq_sr) arch=autoregressive; extra=(--target-order "${arm#seq_}") ;;
    *)             extra=() ;;
  esac
  echo
  run="${TAG:+${TAG}_}$arm"
  echo "── arm: $arm  [$arch] ${extra[*]:-} → runs/$run ──"
  if [ "$DRY" = "1" ]; then
    echo "  [DRY] $PY run_experiment.py $arch ${COMMON[*]} ${extra[*]:-} --run-name $run"
    continue
  fi
  # 백그라운드로 띄우고 주기적으로 반출한다 — 중간에 죽어도 완료된 fold는 남는다.
  "$PY" "$REPO/run_experiment.py" "$arch" "${COMMON[@]}" "${extra[@]}" \
      --run-name "$run" 2>&1 | tee "$OUT/${run}.log" &
  train_pid=$!
  while kill -0 "$train_pid" 2>/dev/null; do
    sleep 120
    sync_out
  done
  wait "$train_pid" || { echo "❌ $run 실패 — $OUT/${run}.log 확인"; sync_out; exit 1; }
  sync_out
  echo "  ✅ $run 완료 · 반출됨"
done

[ "$DRY" = "1" ] && { echo "[DRY] 계획 출력만 하고 종료"; exit 0; }

# --- 4) 평가 ------------------------------------------------------------------
echo
echo "[4/4] 평가 (QWK 주 지표 · 페어드 부트스트랩)"
set -- $ARMS
if [ "$#" -ge 2 ]; then
  # par_genonly는 회귀를 학습하지 않아 점수가 의미 없다 — 채점 평가에서 제외한다.
  eval_args=""; for a in $ARMS; do
    [ "$a" = "par_genonly" ] && continue
    eval_args="$eval_args ${TAG:+${TAG}_}$a"
  done
  "$PY" "$REPO/paper_eval.py" $eval_args --ref "${TAG:+${TAG}_}par" --boot 2000 \
      | tee "$OUT/phase1_eval${TAG:+_$TAG}.txt"
  echo
  echo "판정: par vs par_nogen의 95% CI가 0을 포함하면 **여기서 멈춘다**."
  echo "      근거 생성 헤드가 점수에 기여하지 않는다는 뜻이므로 Phase 2로 가지 않는다."
else
  echo "  arm이 1개라 비교를 건너뜁니다."
fi

echo
echo "반출 완료: $OUT"
echo "  runs/<arm>/fold*/predictions.jsonl  — 재평가에 필요한 전부"
echo "  <arm>.log, phase1_eval.txt"
echo "⚠️ 인스턴스를 내리기 전에 $OUT 이 정말 영속 경로인지 확인하세요."
