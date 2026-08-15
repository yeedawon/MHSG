# 논문 트랙 — 채점과 근거 생성의 결합 구조 (HCLT)

> 대회 트랙과 **분리**한다. 여기 실험은 리더보드 점수를 올리기 위한 것이 아니라
> 아키텍처 주장을 검증하기 위한 것이므로, 통제와 재현성이 성능보다 우선한다.
> 작성 2026-08-14.

---

## 0. 왜 다시 하는가 (기존 기록의 한계)

대회 트랙 저장소(비공개)의 실험 계보에 남은 2026-07-21 기록:

```
├──✗ AR(autoregressive) + distillation    0.7728 / 0.3993   ← 폐기
```

이 수치는 논문 근거로 쓸 수 없다. 교란이 셋이다.

1. **distillation이 붙어 있었다** — 아키텍처 단독 비교가 아니다
2. **백본 교체(7B→9B) 전, 6ep/r32 전**이다. 당시 멀티태스크 자신도 평균회귀가
   심했고(std비 0.23) 이후 크게 개선됐다 — 비교 시점의 기준선이 지금과 다르다
3. **지표 기준이 섞여 있다** (매크로 vs 공식 평균점수)

그리고 **`score→rationale` arm은 구현된 적이 없다.** 코드는 `rs` 한 방향뿐이었다
(`awes_common/autoregressive.py`의 `_SCHEMA_HINT`, `build_target_json`).
2026-08-14에 `target_order` 축을 추가해 `sr` arm을 만들었다.

---

## 1. 주장과 가설

**주장.** 점수와 근거를 *순차 조건부*로 잇는 대신 **공유 표현에서 병렬로 분기**하면,
점수가 생성 텍스트라는 이산 병목을 통과하지 않아도 된다.

> ⚠️ "사람은 점수와 근거를 동시에 떠올린다"는 유추는 **동기이지 근거가 아니다.**
> 논문에서는 설계 설명에만 쓰고, 주장은 아래 측정 가능한 형태로만 편다.

| | 가설 | 반증 조건 |
|---|---|---|
| **H1** | 근거 생성 헤드를 함께 학습하면 점수 성능이 오른다 | `gen_rationale` on/off 차이가 CI에 0 포함 |
| **H2** | 병렬(parallel) ≥ 순차(rs, sr) — 동일 백본·데이터·에폭에서 | 병렬이 어느 순차 arm에도 유의하게 앞서지 못함 |
| **H3** | 순차 arm의 손해는 **텍스트 병목**에서 온다 | 점수 해상도·파싱 실패·근거길이-오차 상관에서 차이 없음 |

**H1이 게이트다.** 음성이면 아키텍처 논문을 접고 결정규칙 논문으로 간다
(별도 트랙).

---

## 2. Arm 설계

| arm | 점수 경로 | 근거 경로 | 설정 |
|---|---|---|---|
| `par` | 회귀 헤드 (은닉표현 직결) | 생성 헤드 (공유 표현) | multitask, `gen_rationale=True` |
| `par-nogen` | 회귀 헤드 | 없음 | multitask, `gen_rationale=False` |
| `seq-rs` | 생성 텍스트에서 파싱 | 점수보다 먼저 생성 | AR, `arch.target_order="rs"` |
| `seq-sr` | 생성 텍스트에서 파싱 | 점수보다 나중에 생성 | AR, `arch.target_order="sr"` |

**Phase 1 (게이트, 2 arm):** `par` vs `par-nogen`
**Phase 2 (본 비교, 3 arm):** `par` vs `seq-rs` vs `seq-sr`

## 3. 통제 변수 — 전부 고정

백본 · LoRA rank/alpha/target · 에폭 · lr · 유효배치 · max_len · seed · fold 분할 ·
근거 데이터(`rationale_train_fold{k}.jsonl` 동일본).

- **에폭 정렬 필수.** 과거 혼합 비교가 에폭 교란으로 무효가 된 전례가 있다.
- **distillation 금지.** 07-21 기록이 그것 때문에 못 쓰게 됐다.
- **백본은 작게.** 리더보드용이 아니므로 9B를 쓸 이유가 없다. 결론이 뒤집힐 위험은
  §6의 스케일 점검으로 관리한다.

## 4. 지표

대회의 블렌드 score는 **쓰지 않는다**(외부 독자에게 의미 없음). AES 문헌 표준으로:

- **QWK** — gold를 사사오입해 산출. AES 비교 가능성의 핵심
- **Spearman ρ, RMSE** — 연속 gold 기준
- **영역별 + 3영역 평균** 둘 다
- **추론 비용** — arm별 생성 토큰 수. 병렬 arm은 점수에 생성이 0토큰이라
  이 열이 곧 실무적 함의다

판정은 **4-fold OOF n=2000**, 유의성은 **페어드 부트스트랩**(`paper_eval.py`).
단일 fold 판정 금지 — 과거 3번 뒤집혔다.

## 5. H3 기전 분석 (병렬이 이겼을 때만)

1. **점수 해상도** — 순차 arm이 실제로 뱉는 서로 다른 점수 값의 개수.
   토큰화된 숫자는 `3.25`처럼 여러 토큰으로 쪼개져 확률질량이 흩어진다
   (`awes_common/autoregressive.py` 모듈 주석에 이미 지적돼 있다)
2. **파싱 실패율·재시도 횟수** — 병렬 arm은 구조상 0
3. **근거 길이/품질 ↔ 점수 오차 상관** — `seq-rs`에서만 높게 나오면
   "점수가 근거에 오염된다"는 병목 가설의 직접 증거
4. **예측 분산 압축** — arm별로 예측σ/goldσ 비교

## 6. 결정 게이트

```
Phase 1 (par vs par-nogen, 2 arm × 4 fold)
   │
   ├─ CI가 0 포함 → 아키텍처 주장 폐기. 결정규칙 논문으로 전환 (여기서 멈춘다)
   │
   └─ 유의 → Phase 2 (3 arm × 4 fold)
         │
         ├─ 병렬이 순차 양쪽에 유의 → H3 기전 분석 → 스케일 점검(9B 1회) → 집필
         └─ 그 외 → 결과를 있는 그대로 보고(음성 결과도 기여). 주 기여를 결정규칙으로
```

## 7. 실행

공통 인자를 셸 변수로 묶어 **arm 간 차이가 한 축뿐임을 눈으로 보장**한다.

```bash
COMMON="--backbone <소형백본> --epochs 4 --lr 1e-4 --batch 4 --grad-accum 4 \
        --lora-r 16 --lora-alpha 32 --max-len 2048 --seed 42 --sparse-ce"

# Phase 1 — 게이트 (생성 손실 유무만 다르다)
python3 run_experiment.py multitask $COMMON --run-name par
python3 run_experiment.py multitask $COMMON --run-name par_nogen --gen-off

# Phase 2 — 순서 축 (생성 순서만 다르다)
python3 run_experiment.py autoregressive $COMMON --run-name seq_rs --target-order rs
python3 run_experiment.py autoregressive $COMMON --run-name seq_sr --target-order sr
```

- `--gen-off` = `arch.gen_off` (생성 손실 완전 off) → `par-nogen`
- `--target-order` = `arch.target_order` (2026-08-14 신설) → `seq-rs` / `seq-sr`
- 판정: `paper_eval.py`(QWK·ρ·RMSE + 페어드 부트스트랩)

## 8. 남는 리스크 (미리 적어둔다)

- **null 확률이 낮지 않다.** 헤드·손실 축 8~9개가 전부 ±0.001이었다. Phase 1이
  게이트인 이유다
- **단일 데이터셋·단일 언어.** 일반화 주장은 하지 않거나, 공개 AES 벤치마크
  재현을 추가해야 한다
- **신규성 미확인.** 공유 인코더 + trait별 헤드는 MTL/AES/다목적 리워드 모델에
  선례가 많다. `근거 생성 헤드와 회귀 헤드의 공동 학습`이 AES에 선례가 있는지
  **집필 전에 문헌 확인**이 필요하다
