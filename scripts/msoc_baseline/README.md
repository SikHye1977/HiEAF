# MSOC external baseline — HiEAF와 분리된 새 실험

**2026-11-26 Blackwell 이관:** 새 Windows 워크스테이션은 [BLACKWELL.md](BLACKWELL.md)의 설치·메타데이터·실행 절차를 따른다. `--data-config`로 데이터/메타데이터 루트를 지정하며, 과거 PC 절대 경로 대신 검증된 portable reference를 사용한다. 원래 HiEAF 가중치·캐시는 필요하지 않다. 공식 논문의 원래 분할/수치 재현이 아니라 같은 모델을 현재 고정 분할에 적용한 비교실험이다.

공식 소스: https://github.com/bok-bok/MSOC , commit `0014452f1efd1ca18b38357b0c25f0ff9df87c09`.
`upstream/`의 원본64파일은 수정하지 않고 보존하며 `upstream.lock.json`으로 검증한다. 실행 adapter는 원본 encoder를 격리하여 읽는다. HiEAF 코드·가중치·캐시·결과와 frozen manifest는 변경하지 않는다.

현재 Windows 데이터로 새로 학습하는 비교다. 원본 워크스테이션 미디어 해시를 복원하거나 역사적 학습 재현으로 취급하지 않는다. 현재 파일, 고정 ID/분할/라벨, 디코딩 및 새 cache/checkpoint 일관성을 검증한다.

## 빠른 기본 설정

| 항목 | 설정 |
|---|---|
| 모델 | **MSOC-ResNet**, 공식 ResNet variant, 전체 모델 from scratch |
| 실행 범위 | source별 학습 3회, 3 in-domain + 6 cross-dataset 평가 |
| Seed / epoch | 42 / 30 |
| Physical / effective batch | 64 / 64, accumulation 1; BN도 실제 64(마지막 batch 제외) |
| Optimizer | Adam, lr 0.0002, betas (0.5,0.9), weight_decay 0.0001 |
| Scheduler | ReduceLROnPlateau(source-dev loss), factor .5, patience 3, min_lr 1e-8 |
| Loss | 공식 4 OCSoftmax + AV cross-entropy 합 |
| Best | source-dev 연속 최종 score ROC-AUC 최대; 동률이면 앞 epoch |
| Precision | 학습 BF16 AMP, OC/CE FP32; dev/test FP32, TF32 off |
| 영상 | prefix 1.2초, 25fps × 30 RGB frames, 100×100 crop |
| 음성 | 16kHz 첫 채널, MFCC 120×13; 마지막 창 포함 1.215초 |
| Augmentation | train random spatial crop + horizontal flip; prefix 시간 구간 고정 |
| Sampling | joint (video,audio) 클래스 균형 복원추출, epoch당 원래 train 수만큼 draw |
| Workers | 전처리 thread4, DataLoader process2 |

단일 audio/video 및 AV 내부 audio/video의 네 인코더와 OCSoftmax, AV classifier, 공식 `mean` 추론을 유지했다. head만 학습하지 않는다. 이 구성에서는 외부 사전학습 가중치를 로드하지 않는다.

**해석 범위:** 빠른 MSOC-ResNet/single-prefix 비교다. SCNet-STIL 또는 전체 영상 multi-window 결과로 표시하지 않는다. HiEAF와 평가 ID는 같지만 모델 입력 범위·해상도는 다르다. AVDF1M의 국소 조작이 prefix 밖에 있을 수 있다. 이 결과 하나를 모든 MSOC 변형에 대한 우월성으로 해석하지 않는다. `--backbone scnet`도 구현했지만 GPU 속도/메모리는 아직 측정하지 않았다. target 성능에 따라 backbone을 선택하지 않는다.

## 데이터

| Dataset | Train | Dev | Test |
|---|---:|---:|---:|
| FAV | 14,724 | 3,147 | 3,126 |
| AVDF1M | 14,724 | 3,147 | 120,327 |
| PGF | 9,948 | 2,092 | 2,082 |

AVDF1M은 기존 `AVDF1M_fav_size_seed42` 목록을 그대로 사용한다. test는 줄이지 않는다. PGF modality 라벨은 기존 공식 JSON의 TTS/lip-sync 근거로 overlay하며 manifest를 바꾸지 않는다. AVDF1M test의 898개 클립은 일부 modality 라벨이 없지만 clip 라벨이 알려져 있으므로 clip 평가에 모두 포함한다. 평가에서는 modality loss를 계산하지 않는다.

프로젝트 Fake=1을 내부에서 공식 MSOC Real=1로 변환한다. 공식 추론식:

`p_real = (1[video_cos > .5] + 1[audio_cos > .5] + softmax(AV_logits)[real]) / 3`

`fake_score = 1-p_real`, `fake_prediction = fake_score >= .5`.

ROC-AUC는 연속 fake score, Fake F1은 Fake 양성, Real FPR은 Real 중 Fake 판정 비율이다. MSOC에는 자체 stored BN을 사용한다. HiEAF의 X3D 전용 A/B 개입은 적용하지 않는다.

## 실행

프로젝트 루트 PowerShell, `.venv/Scripts/python.exe` 사용. 현재 PC에는 공식 소스·추가 dependency 설치 완료.

```powershell
# 새 checkout에서만 필요
./scripts/msoc_baseline/setup.ps1

$exp = './scripts/msoc_baseline/experiment.py'
$out = './reports/msoc_resnet_v1'

./scripts/run.ps1 $exp preflight --output $out --device cuda:0
./scripts/run.ps1 $exp run --output $out --device cuda:0
```

전체 학습은 아직 시작하지 않았다. `run`은 세 학습과 아홉 평가 및 기존 결과 비교표를 수행한다. source 하나씩 실행하려면:

```powershell
./scripts/run.ps1 $exp run --sources FAV --output $out --device cuda:0
./scripts/run.ps1 $exp run --sources AVDF1M --output $out --device cuda:0
./scripts/run.ps1 $exp run --sources PGF --output $out --device cuda:0
```

기본 targets는 항상 세 데이터셋. source/target 선택 변경 시 완료 단계는 재사용한다. 모델·epoch·batch·seed·precision·코드·데이터 identity 변경은 새 output이 필요하다. worker 수와 저장 간격은 변경 가능하다. 실패나 mismatch를 해결하려고 과거 결과를 지우지 않는다.

## 중단·재개

- Ctrl+C 한 번: 현재 optimizer update를 끝내고 저장·중단. 동일 `run` 명령으로 재개한다.
- 강제 종료/전원 차단: 최근 checkpoint부터 재개(기본 180초 간격). 두 번째 Ctrl+C는 강제 종료이므로 미저장 작업을 재수행한다.
- latest에 model/BN/optimizer/scheduler/AMP scaler/Python·NumPy·Torch·CUDA RNG/epoch/draw cursor/best 상태를 저장한다. accumulation 중간은 저장하지 않는다.
- draw 위치로 crop/flip을 결정하고 DataLoader RNG는 model RNG와 분리한다. GPU kernel까지 bitwise 재현을 보장하는 설정은 아니다.
- 캐시는 sample별, 평가는 batch별 commit. 실패 샘플은 조용히 제외하지 않는다. writer lock으로 중복 실행을 막는다.
- 터미널에는 stage/epoch/count/%/rate/elapsed/현재 stage ETA를 표시한다. 추가 학습 ETA는 cache/dev/test 제외를 명시한다.

```powershell
# 별도 터미널에서 파일로 중단 요청
New-Item -ItemType File "$out/STOP_REQUESTED"
# 재개 전 요청 파일만 제거
Remove-Item -LiteralPath "$out/STOP_REQUESTED"
./scripts/run.ps1 $exp run --output $out --device cuda:0
./scripts/run.ps1 $exp status --output $out
```

## 캐시·시간

source와 무관한 영상/MFCC 입력을 한번 저장해 30epoch 및 세 source 평가에서 재사용한다. 전체 MP4 재인코딩이나 전체 WAV 추출을 하지 않는다.

- train: `reports/msoc_inputs_v1` (C: SSD), 예상35–45GiB.
- dev/test 기본값: `reports/msoc_eval_inputs_v1`, 예상70–100GiB. Blackwell에서는 `--eval-cache`로 여유 있는 NVMe 경로를 지정한다.
- 실제 압축률은 다르며 cohort 시작 전 보수적인 비압축 크기로 여유 공간 검사.
- `--train-cache` / `--eval-cache`로 경로 변경 가능. 기존 데이터를 복사하지 않으면 새 위치에서 다시 생성한다.
- `prepare`는 cache만 생성. `benchmark`는 임시 synthetic optimizer 검사+source train의 bounded decode만 수행한다.

RTX3090 실측: ResNet batch64 BF16 GPU 연산 약155.9 samples/s, FP32 약272.6 samples/s, peak allocated 8.02GiB. 실제 source-train36개 decode 약19.1 samples/s. 별도 입력-cache/Windows-worker/실제 미디어 FP32 검사도 통과했다(작은 warm cache에서 약225.7 samples/s).

이 수치들은 전체 처리량이 아니다. 초기 계획은 **세 source 학습+9평가 약8–16시간**, HDD/전처리 병목이 크면 하루 이상 가능하다. 첫 실제 epoch 후 갱신한다. synthetic GPU 값은 cache I/O, 체크포인트 저장, Windows worker 시작을 제외한다. 작은 cache 검사는 OS 캐시가 따뜻한 상태다. 이전 HiEAF 장시간 추출과 동일 작업량으로 비교하지 않는다.

## 결과

- `source_*/latest.pt`, `best.pt`, `HISTORY.json`, `TRAIN_COMPLETE.json`
- `source_*/test/{target}/scores.sqlite`, `predictions.csv`, `metrics.json`, `COMPLETE.json`
- `SUMMARY.csv/json`: MSOC 완료 결과
- `COMPARISON_WITH_HIEAF.csv/json`: 보존된54행+MSOC9행. manuscript A=historical B, B=historical C. 정확한 test ID/clip label 일치 검증. 점추정 비교이며 새로운 유의성 검정은 아니다.
- `PIPELINE_COMPLETE.json`: 요청한 조합 완료 시에만 생성.

```powershell
./scripts/run.ps1 $exp compare --output $out
./scripts/run.ps1 -m unittest scripts.msoc_baseline.test_pipeline -v
```

## 공식 코드 보완 내역

1. best checkpoint 실제 strict load, hash/identity 검증. 공식 `eval_all.py`의 checkpoint 변수만 만들고 로드하지 않는 경로를 사용하지 않는다.
2. dev AUC는 연속 score, metrics는 split마다 새로 계산한다. hard prediction AUC나 누적 metric 상태를 쓰지 않는다.
3. ResEncoder maxpool의 `no_grad`가 frontend gradient를 끊는 부분은 adapter override로 수정. 원본 보존.
4. optional SCNet의 RGB/time 축을 `permute` 후 reshape하여 보존.
5. 실제 상태 기반 resume, Windows 이름·atomic 저장·PermissionError retry·writer lock 추가.
6. 원본의 원해상도50% resize 대신 short-side128 표준화 후100 crop. timestamp25fps 재표집, first-video PTS 기준 audio 정렬. 짧은 video는 zero, 짧은 MFCC는 wrap. 23.976/29.97/30fps 입력도 같은 구간을 사용.
7. 공식30epoch/batch64/Adam/four-OC+CE/mean은 유지. 현재 고정 불균형 train에서는 joint-class 균형 복원추출. 공식 논문 원래 split 재현으로 표시하지 않는다.
8. BF16 AMP 및 cuDNN 활성화. target 학습/threshold tuning/HiEAF BN 개입 없음. Lightning/fairseq/wandb/global `models` import 의존성 제거.

정적 코드 문제가 저자의 실제 논문 실험에도 있었다고 단정하지 않는다. 원본 forward/loss/score 대응, frontend gradient, checkpoint resume, unknown modality 평가 및 cache 무결성을 테스트했다. 기록: `reports/msoc_implementation_validation_20261125`. 기존 결과를 수정하거나 새 성능 수치를 만들어 넣지 않았다.
