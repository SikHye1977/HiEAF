"""Human-readable run report generated from the saved measurements."""
def write_report(folder,summary,verification):
    m=summary['metrics'];cfg=summary['config']
    names=dict(artifact_or='Artifact-only / X3A OR',emotion_or='Emotion OR',
               hierarchical_or='동일 신규 head + 고정 최종 OR',trainable_final='학습형 최종 점수 융합',
               artifact_affine='Artifact-only 학습형 보정')
    lines=['# FAV → FAV 전체 학습형 최종 점수 융합 실험','',
           '기존 Artifact와 frozen emotion encoder는 유지하고, GRU + Temporal Attention 두 head를 전체 train에서 새로 학습했다.',
           '최종 융합은 계층 내부의 OR을 유지하고 마지막 OR만 3개 파라미터의 logistic score fusion으로 바꿨다.',
           '전체 고정 test 3,126개(정상60/조작3,066)를 평가했다. F1은 fake-positive binary F1이며 임계값은 0.5다.','',
           '## 결과','',
           '| 방법 | ROC-AUC (%) | Fake F1 (%) | Macro F1 (%) | BA (%) | Real FPR (%) | FP / FN |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in m.items():
        (tn,fp),(fn,tp)=r['confusion_matrix']
        lines.append(f'| {names[name]} | {100*r["roc_auc"]:.4f} | {100*r["f1"]:.4f} | {100*r["macro_f1"]:.4f} | '
                     f'{100*r["balanced_accuracy"]:.4f} | {100*r["real_fpr"]:.4f} | {fp} / {fn} |')
    lines+=['','| 학습형 최종 융합의 비교 대상 | AUC 차이 (%p) | F1 차이 (%p) |',
            '|---|---:|---:|']
    for name,d in summary['deltas_pp'].items():
        lines.append(f'| {names[name]} | {d["roc_auc"]:+.4f} | {d["f1"]:+.4f} |')
    lines+=['','검증된 sample별 판정 변화:','']
    for name,d in verification['paired'].items():
        lines.append(f'- {names[name]} 대비: 교정 {d["corrected"]}개 / 새 오분류 {d["regressed"]}개 / 판정 차이 {d["prediction_disagreements"]}개.')
    control=verification['paired']['artifact_affine']
    if control['prediction_disagreements']==0:
        lines+=['','학습형 최종 융합과 artifact-only 보정의 모든 이진 판정이 같다. F1 개선만으로 emotion의 추가 탐지 기여를 입증할 수 없다.']
    lines+=['','## 구조 및 학습','',
            '~~~mermaid','flowchart LR',
            '    V["128 frames / 256×256"] --> X["고정 X3D_m"]',
            '    W["16 kHz mono waveform"] --> AA["고정 AASIST"]',
            '    X --> AV["Av"]','    AA --> AS["Aa"]',
            '    AV --> A["Artifact OR: A"]','    AS --> A',
            '    V --> HSE["고정 HSEmotion"]',
            '    W --> E2V["고정 emotion2vec+seed"]',
            '    HSE --> VG["Projection + native emotion probabilities → GRU → Temporal Attention → FC"]',
            '    E2V --> AG["Projection + native emotion probabilities → GRU → Temporal Attention → FC"]',
            '    VG --> EV["Ev"]','    AG --> EA["Ea"]',
            '    EV --> E["Emotion OR: E"]','    EA --> E',
            '    A --> F["sigmoid(wA·logit(A) + wE·logit(E) + b)"]','    E --> F','~~~','',
            '- A = 1 − (1 − Av)(1 − Aa), E = 1 − (1 − Ev)(1 − Ea).',
            '- F = sigmoid(wA·logit(A) + wE·logit(E) + b). Logit 전 확률은 [1e-6,1−1e-6]로 clamp.',
            '- 가중치 두 개와 bias는 제한 없는 실수다. OR 또는 독립 사건 확률이라는 수학적 보장은 없으며 폴더명 trainable_or는 마지막 OR 대체를 뜻한다.',
            '- X3D_m/AASIST의 checkpoint·입력·artifact score는 기존 완료 결과와 동일하다. 기존 가중치의 전체 end-to-end 재학습은 하지 않았다.',
            '- X3D_m: 전체 영상에서 균일128프레임, 짧은변256/center crop256, mean0.45/std0.225.',
            '- AASIST: 16kHz mono, 284,672 samples까지 우측 zero padding, truncation 금지인 기존 artifact 점수 재사용.',
            '- HSEmotion: 같은128프레임·256×256 center crop, ImageNet 정규화. Audio emotion: 기존 3초 window/1초 hop, 짧으면 전체 waveform.',
            '- 시간순 embedding → 128차원 projection + 유효 token만 BatchNorm/ReLU/dropout → native 감정 확률 결합 → 단방향 GRU → masked Temporal Attention → FC.',
            '- Video GRU hidden64/1layer, Audio hidden128/2layers, dropout0.2. Video는 video label, audio는 audio label로 별도 감독.',
            f'- 새 emotion heads: 전체 train14,724개, {cfg["epochs"]} epochs, batch{cfg["batch_size"]}, AdamW lr{cfg["lr"]}, weight_decay{cfg["weight_decay"]}, BCE pos_weight{cfg["pos_weight"]}, gradient clip5.',
            f'- Seed: video{cfg["seed"]}/audio{cfg["seed"]+1}. Best head 선택: 전체 FAV dev3,147개의 modality ROC-AUC.',
            '- 최종 score fusion 및 artifact-affine 대조군: dev3,147개만 사용, 자연 클래스 분포의 BCE, L-BFGS-B, L2=0.01 (artifact identity prior). Test로 모델·임계값을 선택하지 않았다.',
            '- 고정 OR과 학습형 융합은 동일한 신규 heads의 동일 점수로 비교한다. 과거 OR 기록과 이번 신규 OR을 혼동하지 않는다.','',
            '## 선택된 epoch 및 소요시간','']
    for mod,s in summary['selection'].items():
        lines.append(f'- {mod}: best epoch{s["best_epoch"]}, dev AUC{s["dev_auc"]:.6f}, train{s["train_n"]}, 완료{s["epochs_completed"]}epochs.')
    lines.append(f'- 이번 호출 wall time: {summary["wall_seconds"]:.1f}초. 캐시 검사·head 학습·score fitting·평가 포함, 과거 raw feature 추출 제외.')
    lines.append(f'- 완료 epoch의 누적 학습/검증 시간: {summary["training_epoch_seconds"]:.1f}초. 중단으로 재실행된 미완료 epoch 시간은 미포함.')
    lines+=['','## 최종 융합 파라미터','',str(summary['fusion_weights']),'',
            '## 해석 범위','',
            '- Pilot의 learned fusion은 이미 full dev와 full-trained 기존 heads를 사용했다. 이번에는 동일 architecture의 heads를 전체 train/20epoch에서 새로 학습해 전체 파이프라인을 실행했다. 단순 데이터 확대에 따른 성능 상승을 전제하지 않는다.',
            '- 단일 seed, 과거에 이미 관찰한 FAV test다. 이번 결과를 독립적인 새 holdout이나 cross-dataset 개선 근거로 부르지 않는다.',
            '- Dev를 head 선택과 score fitting에 재사용했다. 독립 calibration 분할 검증이 아니며, full-data 실행이 이 한계를 없애지 않는다.',
            '- Real60/fake3,066의 심한 불균형 때문에 fake F1과 함께 macro F1·FPR·혼동행렬을 해석한다.',
            '- GRU는 입력 순서를 구조적으로 활용한다. 이번 실험의 성능만으로 감정 불일치의 인과적 학습이나 시간순서의 추가 이득을 주장하지 않는다.',
            '- 중단 시 마지막 완료 epoch부터 복구한다. 진행 중이던 epoch는 재실행한다. 변경된 설정은 새 출력 경로를 사용한다.',
            '- Frozen source 결과·manifest·ID·split·label은 수정하지 않았다. 완료 파일과 잠금 기록은 재실행 시 검증한다.','']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
