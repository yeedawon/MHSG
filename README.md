# MHSG

한국어 논증적 글 자동 채점에서 **점수 예측과 근거 생성을 어떻게 결합할 것인가**를
통제 실험으로 비교하는 연구 코드입니다.

- 병렬(parallel): 공유 표현에서 회귀 헤드와 생성 헤드로 분기
- 순차(sequential): 근거→점수(`rs`) 또는 점수→근거(`sr`)를 한 생성열로 이어 붙임

가설·통제 변수·판정 기준은 [PAPER_TRACK.md](PAPER_TRACK.md)에 정리되어 있습니다.

> 이 저장소는 대회 제출 코드와 분리된 **연구 트랙**입니다. 리더보드 점수 최적화가
> 아니라 아키텍처 비교가 목적이므로, 통제와 재현성이 성능보다 우선합니다.

---

## 🔴 데이터는 포함되어 있지 않습니다

학습·평가에 쓰는 한국어 에세이 데이터(국립국어원 말평 배포본 및 NIKL 코퍼스)는
**이용약관 동의 후 제공되는 자료**이며 재배포할 수 없습니다. 원문에서 파생된 합성
근거도 같은 취급입니다. 따라서 이 저장소에는 코드만 있고, 데이터는 각자 확보해
로컬에 두어야 합니다.

`.gitignore`가 `data/`, `*.jsonl`, `runs/`를 차단합니다. **이 규칙을 우회해서
데이터를 커밋하지 마세요.**

### 데이터 배치

`AWES_ROOT` 아래에 다음 구조로 둡니다.

```
$AWES_ROOT/
├── AWES/data/
│   ├── train_with_folds.jsonl        # 본문 코퍼스 (id, prompt, essay, score)
│   ├── holdout_fold0.jsonl … fold3   # fold별 평가셋 (배타적 4분할)
│   └── official_val.jsonl            # (선택) 별도 검증셋
└── AWES/rationales/fold_train/
    └── rationale_train_fold0.jsonl … fold3   # fold별 학습셋 + 영역별 근거
```

**스키마**

```jsonc
// train_with_folds.jsonl / holdout_fold{k}.jsonl
{"id": "...", "prompt_num": "Q1", "prompt": "글의 주제",
 "essay": "학생 글 본문",
 "score": {"content": 3.25, "organization": 4.0, "expression": 2.5}}

// rationale_train_fold{k}.jsonl  (본문 없음 — id로 코퍼스와 조인한다)
{"id": "...",
 "content":      {"score": 3.25, "rationale": "근거 2~3문장"},
 "organization": {"score": 4.0,  "rationale": "..."},
 "expression":   {"score": 2.5,  "rationale": "..."}}
```

점수는 1.0~5.0 실수입니다(정수가 아닙니다). 근거는 fold와 무관하게 전역 1회
합성하고, **fold-k 학습셋에 holdout-k가 섞이지 않도록** 분리합니다.

### 배치 검증

학습 전에 반드시 돌리세요. 경로 오지정·id 불일치·누수를 잡습니다 — 셋 다 조용히
지나가면 지표를 통째로 무의미하게 만듭니다.

```bash
AWES_ROOT=/path/to/data python3 check_data.py
```

---

## 설치

```bash
pip install torch transformers peft accelerate
```

평가 스크립트(`paper_eval.py`, `check_data.py`)는 **표준 라이브러리만** 사용하므로
GPU 없는 환경에서도 돌아갑니다. 학습에만 위 의존성이 필요합니다.

## 실행

arm 간 차이가 한 축뿐임을 보장하기 위해 공통 인자를 묶어 씁니다.

```bash
export AWES_ROOT=/path/to/data
COMMON="--backbone <백본> --epochs 4 --lr 1e-4 --batch 4 --grad-accum 4 \
        --lora-r 16 --lora-alpha 32 --max-len 2048 --seed 42 --sparse-ce"

# Phase 1 — 게이트: 근거 생성 헤드가 점수 성능에 기여하는가
python3 run_experiment.py multitask $COMMON --run-name par
python3 run_experiment.py multitask $COMMON --run-name par_nogen --gen-off

# Phase 2 — 생성 순서 축
python3 run_experiment.py autoregressive $COMMON --run-name seq_rs --target-order rs
python3 run_experiment.py autoregressive $COMMON --run-name seq_sr --target-order sr
```

Phase 1이 음성(신뢰구간이 0을 포함)이면 거기서 멈춥니다 — 근거 생성 헤드가 점수에
기여하지 않는다면 아키텍처 주장이 성립하지 않습니다.

## 평가

AES 문헌과 비교 가능하도록 **QWK**를 주 지표로 두고, 연속 정답을 살리는 Spearman·
RMSE를 함께 봅니다. 유의성은 페어드 부트스트랩으로 판정합니다.

```bash
python3 paper_eval.py par par_nogen --ref par --boot 2000
python3 paper_eval.py par seq_rs seq_sr --ref par --metric qwk
```

판정은 **4-fold OOF 전체**로 합니다. 단일 fold 판정은 이 프로젝트에서 실제로 세 번
결론이 뒤집힌 적이 있습니다.

---

## 저장소 구조

```
awes_common/          공유 파이프라인 (데이터·지표·학습 루프)
  ├── multitask.py       병렬 arm — 회귀 헤드 + 생성 헤드
  ├── autoregressive.py  순차 arm — target_order로 rs/sr 전환
  ├── pipeline.py        fold 루프·TrainConfig
  └── data.py            fold 로딩 + 누수 재검증
run_experiment.py     학습 진입점
paper_eval.py         QWK·ρ·RMSE + 페어드 부트스트랩
check_data.py         데이터 배치 검증
PAPER_TRACK.md        실험 설계 (가설·통제·결정 게이트)
```

### 알아두면 좋은 것

- **패키지 이름이 `awes_common`인 이유**: 대회 트랙 저장소와 모듈을 공유하기 위해
  이름을 유지했습니다. 양쪽에서 버그 수정을 서로 가져오기 쉽습니다.
- **`contrastive.py`는 기본적으로 쓰이지 않습니다.** 기각된 축이지만
  `multitask.py`가 조건부로 참조하므로, 제거하면 클론이 import 단계에서 깨집니다.
  `contrastive_dir`을 지정하지 않으면 코드 경로 전체가 건너뛰어집니다.

## 라이선스

아직 정해지지 않았습니다. 공개 저장소에 라이선스 파일이 없으면 기본적으로 모든
권리가 유보되어 다른 사람이 합법적으로 사용·수정할 수 없습니다. 배포 의도에 맞는
라이선스를 추가하세요.
