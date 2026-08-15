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
DRY="${DRY:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 통제 변수 — arm 간 차이가 한 축뿐이어야 한다. 여기를 바꾸면 전 arm을 다시 돌려야 한다.
EPOCHS="${EPOCHS:-4}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-4}"
LORA_R="${LORA_R:-16}"
MAXLEN="${MAXLEN:-2048}"
SEED="${SEED:-42}"

export AWES_ROOT
export HF_HOME="${HF_HOME:-$OUT/hf}"      # 캐시도 OUT에 두면 재실행 시 재다운로드를 피한다

COMMON=(--backbone "$BACKBONE" --epochs "$EPOCHS" --lr "$LR"
        --batch "$BATCH" --grad-accum "$ACCUM"
        --lora-r "$LORA_R" --lora-alpha $((LORA_R * 2))
        --max-len "$MAXLEN" --seed "$SEED" --sparse-ce)

echo "=================================================================="
echo " MHSG Phase 1 — 일회성 실행"
echo "   백본     $BACKBONE"
echo "   데이터   $AWES_ROOT"
echo "   반출     $OUT"
echo "   캐시     $HF_HOME"
echo "   arm      $ARMS"
echo "   통제     ep=$EPOCHS lr=$LR batch=$BATCH×$ACCUM r=$LORA_R len=$MAXLEN seed=$SEED"
echo "=================================================================="

# --- 1) 데이터 검증 — 여기서 막히면 학습 시간을 버리지 않는다 ------------------
echo "[1/4] 데이터 배치 검증"
if ! python3 "$REPO/check_data.py"; then
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
  # 예측·config·로그만 뺀다. 어댑터는 크고, 재평가에는 예측만 있으면 된다.
  mkdir -p "$OUT/runs"
  rsync -a --prune-empty-dirs \
        --include='*/' \
        --include='predictions.jsonl' --include='config.json' --include='cv_result.json' \
        --exclude='*' \
        "$REPO/runs/" "$OUT/runs/" 2>/dev/null || true
}

echo "[3/4] 학습"
for arm in $ARMS; do
  extra=()
  [ "$arm" = "par_nogen" ] && extra=(--gen-off)   # 생성 손실 off = 게이트의 대조군
  echo
  echo "── arm: $arm ${extra[*]:-} ──"
  if [ "$DRY" = "1" ]; then
    echo "  [DRY] python3 run_experiment.py multitask ${COMMON[*]} ${extra[*]:-} --run-name $arm"
    continue
  fi
  # 백그라운드로 띄우고 주기적으로 반출한다 — 중간에 죽어도 완료된 fold는 남는다.
  python3 "$REPO/run_experiment.py" multitask "${COMMON[@]}" "${extra[@]}" \
      --run-name "$arm" 2>&1 | tee "$OUT/${arm}.log" &
  train_pid=$!
  while kill -0 "$train_pid" 2>/dev/null; do
    sleep 120
    sync_out
  done
  wait "$train_pid" || { echo "❌ $arm 실패 — $OUT/${arm}.log 확인"; sync_out; exit 1; }
  sync_out
  echo "  ✅ $arm 완료 · 반출됨"
done

[ "$DRY" = "1" ] && { echo "[DRY] 계획 출력만 하고 종료"; exit 0; }

# --- 4) 평가 ------------------------------------------------------------------
echo
echo "[4/4] 평가 (QWK 주 지표 · 페어드 부트스트랩)"
set -- $ARMS
if [ "$#" -ge 2 ]; then
  python3 "$REPO/paper_eval.py" $ARMS --ref par --boot 2000 | tee "$OUT/phase1_eval.txt"
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
