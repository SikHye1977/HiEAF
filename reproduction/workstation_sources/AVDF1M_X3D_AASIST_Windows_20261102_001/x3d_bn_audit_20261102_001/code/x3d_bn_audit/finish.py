"""Validate completed predictions, write an evidence report, and seal the bundle."""
import argparse
import csv
import json
from pathlib import Path
import time

from .core import metrics, sha256, write_json
from .evaluate import load_rows


def pct(x): return f'{100*x:.6f}%'


def finish(bundle):
    assert not (bundle/'SHA256SUMS.txt').exists(), 'Already sealed; do not overwrite'
    complete = json.loads((bundle/'metadata/EVALUATION_COMPLETED.json').read_text())
    assert complete['all_conditions_completed']
    summary = json.loads((bundle/'results/summary.json').read_text())
    audit = json.loads((bundle/'metadata/checkpoint_audit.json').read_text())
    config = json.loads((bundle/'metadata/config.json').read_text())
    baseline = json.loads((bundle/'results/A/baseline_reproduction.json').read_text())
    assert baseline['passed']
    all_rows = {c:load_rows(bundle/f'results/{c}/predictions.jsonl') for c in 'ABC'}
    for c, rows in all_rows.items():
        recalculated = metrics(rows)
        assert all(summary[c][key] == value for key,value in recalculated.items())
        assert len(rows) == len({r['sample_id'] for r in rows}) == 3147
        assert (summary[c]['video_real'], summary[c]['video_fake'], summary[c]['both_real']['count']) == (1573,1574,792)
        for rank in range(4):
            state = json.loads((bundle/f'results/{c}/state_preservation_rank{rank}.json').read_text())
            assert state['before'] == state['after'] and state['after']['state_sha256'] == audit['state_sha256']
    assert all(a['sample_id']==b['sample_id']==c['sample_id'] and
               a['input_tensor_sha256']==b['input_tensor_sha256']==c['input_tensor_sha256']
               for a,b,c in zip(all_rows['A'], all_rows['B'], all_rows['C']))
    originals = json.loads((bundle/'metadata/original_files_sha256.json').read_text())
    assert all(sha256(path)==expected for path,expected in originals.items())
    assert sha256(bundle/'weights/best_original.pt')==audit['original_checkpoint_sha256']
    assert sha256(bundle/'weights/best_state_dict.pth')==audit['original_export_sha256']
    paired = {}
    for first,second in [('A','B'),('B','C'),('A','C')]:
        before,after=all_rows[first],all_rows[second]
        paired[first+'_'+second]={
            'real_false_positives_corrected':sum(a['video_label']==0 and a['predicted_label']==1 and b['predicted_label']==0 for a,b in zip(before,after)),
            'real_false_positives_introduced':sum(a['video_label']==0 and a['predicted_label']==0 and b['predicted_label']==1 for a,b in zip(before,after)),
            'fake_true_positives_lost':sum(a['video_label']==1 and a['predicted_label']==1 and b['predicted_label']==0 for a,b in zip(before,after)),
            'fake_false_negatives_corrected':sum(a['video_label']==1 and a['predicted_label']==0 and b['predicted_label']==1 for a,b in zip(before,after)),
            'fpr_change_percentage_points':100*(summary[second]['real_fpr']-summary[first]['real_fpr']),
            'logit_auc_change_percentage_points':100*(summary[second]['roc_auc_logit']-summary[first]['roc_auc_logit'])}
    write_json(bundle/'results/paired_comparisons.json',paired)
    fields=['condition','roc_auc_logit','roc_auc_probability','roc_auc_probability_from_logit_float64',
            'precision','recall','f1','accuracy','real_fpr','TN','FP','FN','TP','both_real_fpr','both_real_FP','wall_seconds']
    with (bundle/'results/comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c,m in summary.items():
            writer.writerow({k:m[k] for k in fields if k in m}|m['confusion_counts']|
                            dict(both_real_fpr=m['both_real']['FPR'],both_real_FP=m['both_real']['FP']))
    report=[]
    def add(s=''): report.append(s)
    add('# AVDF1M X3D: 저장 BN 통계와 영상별 BN 통계 비교')
    add('\n## 직접 확인된 결과\n')
    add(f"같은 최적 가중치와 dev 3,147개에서 A의 Real FPR은 {pct(summary['A']['real_fpr'])}, "
        f"B는 {pct(summary['B']['real_fpr'])}, C는 {pct(summary['C']['real_fpr'])}였다. "
        f"BN 방식만 바꾼 B→C의 차이는 {paired['B_C']['fpr_change_percentage_points']:+.6f} percentage points이다.")
    add('A는 기존 최적 epoch의 예측 라벨과 혼동행렬을 재현했다. B/C는 이 재현 검사를 통과한 뒤 실행했다. '
        '표의 숫자만 표시용으로 반올림했으며 JSON/JSONL/CSV에는 Python float의 왕복 가능한 정밀도로 기록했다.')
    add('\n| 조건 | AUC(logit) | AUC(확률) | Fake Precision | Fake Recall | Fake F1 | Real FPR | 모두 Real FPR | 시간(초) |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for c,m in summary.items():
        add(f"| {c} | {pct(m['roc_auc_logit'])} | {pct(m['roc_auc_probability'])} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {pct(m['real_fpr'])} ({m['confusion_counts']['FP']}/1573) | {pct(m['both_real']['FPR'])} ({m['both_real']['FP']}/792) | {m['wall_seconds']:.3f} |")
    add('\n정답: `video_label` (Real=0, Fake=1). 영상 Real 1,573개, 영상 Fake 1,574개, 영상·음성 모두 Real 792개. '
        '임계값은 모든 조건에서 Fake 확률 ≥ 0.5이다. 혼동행렬은 행=실제 [Real,Fake], 열=예측 [Real,Fake] 순서이며 [[TN,FP],[FN,TP]]이다.')
    for c,m in summary.items(): add(f"\n- {c}: `{m['confusion_matrix']}`")
    add('\n시간은 동시 실행한 GPU 4개의 worker 중 가장 긴 실제 경과 시간이다. 파일·입력 해시, 디코딩, 전처리, '
        'GPU 추론과 예측 기록을 포함하며 모델 로드, 사전 smoke 검사와 최종 지표 집계는 제외한다. '
        '조건별 `timing_rank*.json`에 세부 시간이 있다.')
    add('\n## A 재현 검사 및 FP16 포화\n')
    add(f"기존 AUC(확률): {baseline['stored_probability_auc']:.17g}; 재평가: {baseline['new_probability_auc']:.17g}. "
        f"logit 완전 일치 {baseline['exact_logit_matches']}/3147, 확률 완전 일치 {baseline['exact_probability_matches']}/3147, "
        f"임계값 예측 불일치 {len(baseline['prediction_mismatches'])}개. "
        f"최대 절대 logit 차이 {baseline['max_abs_logit_difference']:.17g}, 확률 차이 {baseline['max_abs_probability_difference']:.17g}.")
    add('\n| 조건 | AUC(logit) | AUC(저장 확률) | AUC(float64 sigmoid(logit)) | Real p=0 / p=1 | Fake p=0 / p=1 | p≥.5 vs logit≥0 불일치 |')
    add('|---|---:|---:|---:|---:|---:|---:|')
    for c,m in summary.items():
        s=m['saturation']
        add(f"| {c} | {pct(m['roc_auc_logit'])} | {pct(m['roc_auc_probability'])} | {pct(m['roc_auc_probability_from_logit_float64'])} | {s['real']['probability_zero']} / {s['real']['probability_one']} | {s['fake']['probability_zero']} / {s['fake']['probability_one']} | {m['prediction_disagreements_probability_vs_logit']} |")
    add('\nA의 원시 출력과 sigmoid는 FP16이다. FP32로 변환해 파일에 저장해도 이미 생긴 포화·동점은 복원되지 않는다. '
        '`fake_probability_from_logit_float64`는 같은 logit에 float64 sigmoid를 다시 적용한 별도 진단값이다. '
        '이 값과 logit AUC의 비교는 sigmoid 양자화 영향을, A→B는 전체 순전파 정밀도 영향을 구분한다. '
        '확률 포화가 AUC에 영향을 줄 수 있다는 것과 0.5에서 높은 Real FPR이 발생한다는 것은 구분해야 한다. '
        '두 기준의 예측이 일치하면 sigmoid 포화만으로 해당 FPR을 설명할 수 없다.')
    add('\n## 점수 분포\n')
    for score in ['logit','fake_probability']:
        add(f'\n### {score}\n')
        add('| 조건 | 실제 클래스 | 최소 | 5% | 중앙값 | 95% | 최대 |')
        add('|---|---|---:|---:|---:|---:|---:|')
        for c,m in summary.items():
            for label in ['real','fake']:
                values=m['distribution'][label][score]
                add(f'| {c} | {label} | '+' | '.join(f'{values[k]:.9g}' for k in ['min','p05','median','p95','max'])+' |')
    add('\n분위수: NumPy 기본 linear quantile. 정확한 값은 각 조건 `metrics.json`에 기록했다.')
    add('\n## 평가 방식의 차이\n')
    add('- A: `model.eval()`, 저장된 BN running mean/variance, CUDA autocast FP16, FP16 sigmoid. 기존 평가를 재현한다.\n'
        '- B: `model.eval()`, 같은 저장 BN 통계, FP32 순전파와 sigmoid.\n'
        '- C: 먼저 `model.eval()` 후 BN 83개에만 `training=True`, `track_running_stats=False`. '
        'dropout 등 모든 비-BN 모듈은 eval, 학습된 affine weight/bias와 eps는 유지한다. '
        'PyTorch BN forward는 C에서 running mean/variance 대신 None을 전달하여 현재 입력의 통계만 사용한다. '
        '원래 버퍼는 등록된 상태로 남아 있고 갱신되지 않는다.\n'
        '- 모든 조건은 각 GPU에 batch size 1, 입력 한 영상의 128개 프레임이다. C의 BN 통계 축은 해당 층의 N,T,H,W이며 '
        'N=1이다. 다른 영상이나 GPU의 통계를 모으지 않는다. 4개 worker는 독립 shard `rank::4`를 평가한다.\n'
        '- 조건 및 GPU마다 원본 export를 다시 strict load했다. `torch.no_grad()`, 모든 파라미터 requires_grad=False, '
        'optimizer 생성·backward·update 없음. 영상별 상태를 다음 영상으로 누적하지 않는다.')
    add('\nC는 학습 완료된 네트워크의 정규화 규칙을 바꾸는 평가 방식이다. '
        '재학습이나 여러 dev 영상을 이용한 BN 통계 재추정이 아니며, 기존 표준 eval 결과와 구분하여 보고해야 한다.')
    add('\n## 체크포인트·입력·보존 검증\n')
    add(f"- 최적 epoch: zero-based {audit['best_epoch_zero_based']}, 실제 {audit['best_epoch_one_based']}번째 epoch.\n"
        f"- 원본 full checkpoint: `{audit['original_checkpoint_path']}`\n"
        f"- SHA-256: `{audit['original_checkpoint_sha256']}`\n"
        f"- 원본 export: `{audit['original_export_path']}`\n"
        f"- SHA-256: `{audit['original_export_sha256']}`\n"
        f"- 원본 best.pt의 model과 export: {audit['state_tensor_count']}개 state tensor 모두 torch.equal. "
        f"파라미터 tensor {audit['parameter_tensor_count']}개 ({audit['parameter_elements']:,} elements), "
        f"버퍼 {audit['buffer_tensor_count']}개. BN 83개의 affine, running_mean, running_var, num_batches_tracked 모두 포함.\n"
        f"- 모든 조건·GPU의 평가 전후 state tensor {audit['state_tensor_count']}개가 원본과 완전히 동일. "
        '상세 증거: `state_preservation_rank*.json`, `checkpoint_audit.json`.\n'
        f"- 원본 dev SHA-256: `{config['manifest_sha256']}`\n"
        f"- 전달용 portable dev SHA-256: `{config['portable_manifest_sha256']}`\n"
        '- 3,147개 원본 media SHA-256를 매 조건 확인했고, 전처리 입력 tensor SHA-256도 세 조건 모두 일치했다.\n'
        f'- 원본 체크포인트·dev 목록·기존 X3D 결과 총 {len(originals)}개 파일 해시가 평가 전후 일치했다. '
        '기존 파일에 대한 쓰기나 원본 체크포인트 저장은 수행하지 않았다.')
    add('\n모델: PyTorchVideo `x3d_m`, 마지막 proj를 1-logit Linear로 교체하고 activation=Identity. '
        'full state를 로드하므로 평가 시 pretrained 다운로드는 없다. 상세 모듈 구성은 `metadata/model_architecture.txt`. '
        '전처리는 학습 당시 `data_models.read_video`를 그대로 사용했다: PyAV 두 번 디코딩, '
        '전체 프레임에서 linspace 128개 균일 인덱스, RGB, OpenCV INTER_AREA로 짧은 변 256, '
        '가운데 256×256 crop, /255, mean .45 / std .225, C×T×H×W. '
        '실제 batch shape=1×3×128×256×256. 변경된 augmentation이나 temporal crop은 없다. '
        'PyTorch/CUDA/cuDNN와 TF32 플래그 등 실제 환경은 `metadata/runtime_*.json`에 있다.')
    add('\n## 사실과 미검증 가설\n')
    add(f"이 체크포인트에서 BN 규칙만 바꾼 B→C로 Real 오탐 {paired['B_C']['real_false_positives_corrected']}개가 해소되고, "
        f"새 오탐 {paired['B_C']['real_false_positives_introduced']}개가 생겼다. "
        f"Fake 정탐→미탐 {paired['B_C']['fake_true_positives_lost']}개, 미탐→정탐 {paired['B_C']['fake_false_negatives_corrected']}개였다. "
        '이는 동일 입력·가중치에서 정규화 방식 변경의 직접 측정 결과이다. '
        '효과의 방향과 크기는 위 표를 따른다.')
    if summary['C']['real_fpr'] < summary['B']['real_fpr']:
        add('\n영상별 BN을 사용하면 FPR이 감소하는 현상은 이번 워크스테이션 체크포인트에서도 확인됐다. '
            '이 실험은 저장 BN 통계를 사용하는 방식이 높은 FPR에 기여한다는 근거를 제공하지만, '
            '저장 통계가 왜 그렇게 형성됐는지까지 확정하지는 않는다.')
    else:
        add('\n이번 체크포인트에서는 영상별 BN 사용으로 FPR이 감소한다는 현상이 재현되지 않았다. '
            'Windows 별도 체크포인트의 관찰을 이 체크포인트에 그대로 적용할 수 없다.')
    add('\n미검증 가설: 학습 표본의 클래스·영상 분포와 dev 분포 차이, local BN와 rank0 통계 저장 방식, '
        '영상별 특징 통계 변화, 학습된 출력의 0.5 임계값에서의 보정 문제 등이 기여했을 수 있다. '
        '학습 중 각 rank의 activation 통계 비교·BN 형성 과정 재실험·독립 test 검증은 이번에 수행하지 않았다. '
        '따라서 BN 버퍼 손상, FAV와의 FPR 차이의 유일한 원인, 모든 데이터셋에서의 일반적 개선을 주장하지 않는다. '
        '이번 결과는 체크포인트 선택에도 사용된 동일 dev에서의 원인 점검이며 독립 test 성능 추정이 아니다. '
        'AASIST 재학습이나 cross-dataset 평가는 수행하지 않았다.')
    add('\n## 전달 폴더와 재실행\n')
    add('`weights/`: 원본 best.pt 복사 및 state_dict, `code/`: 실제 audit 코드와 원래 학습/평가 코드 snapshot, '
        '`dependencies/`: 당시 PyTorchVideo 소스, `manifests/`: 원본·portable dev 목록, '
        '`reference/`: 기존 최적 결과/학습 설정, `results/{A,B,C}/predictions.jsonl`: 각 영상의 예측, '
        '`results/comparison.csv`, `results/summary.json`: 비교 수치. '
        '데이터 영상 자체는 용량 때문에 복제하지 않았고, 경로·미디어 해시를 manifest에 포함했다.')
    command=json.loads((bundle/'metadata/evaluation_command.json').read_text())
    add('\n실제 실행 명령과 환경:\n\n```text\n'+command['command_line']+'\nCWD: '+command['cwd']+
        '\nENV: '+json.dumps(command['environment'])+'\n```')
    add('\n같은 경로에서 재실행해 결과를 덮어쓰면 안 된다. 새 output을 지정해 `x3d_bn_audit.launch`를 실행한다. '
        '정확한 원래 launch 명령은 `metadata/evaluation_command.json`의 launch_command에 있다. '
        '데이터를 다른 위치로 옮겼다면 새 launch의 --dataset을 수정한다. 기존 학습 실행 명령은 '
        '`reference/training_launch.json`, 원래 dev 평가 루프는 `code/avdf1m_fav_reproduction/train.py`에 포함했다.')
    add('\n모든 전달 파일은 `SHA256SUMS.txt`로 확인할 수 있다 (`sha256sum -c SHA256SUMS.txt`). '
        '이 해시 목록 자체만 자기참조를 피하려고 제외했다. `DELIVERY.json`은 봉인 직전 파일 수·용량을 기록한다.')
    (bundle/'REPORT.md').write_text('\n'.join(report)+'\n')
    files=sorted(p for p in bundle.rglob('*') if p.is_file())
    write_json(bundle/'DELIVERY.json',dict(completed=time.time(),folder=str(bundle),
               files_before_sealing=len(files),bytes_before_sealing=sum(p.stat().st_size for p in files),
               note='Counts precede DELIVERY.json and SHA256SUMS.txt; final size printed by finish',
               all_requested_conditions_complete=True,all_original_files_preserved=True))
    files=sorted(p for p in bundle.rglob('*') if p.is_file())
    (bundle/'SHA256SUMS.txt').write_text(''.join(f'{sha256(p)}  {p.relative_to(bundle)}\n' for p in files))
    final_files=sorted(p for p in bundle.rglob('*') if p.is_file())
    print(json.dumps(dict(event='BUNDLE_COMPLETE',folder=str(bundle),files=len(final_files),
                          total_bytes=sum(p.stat().st_size for p in final_files)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    finish(parser.parse_args().bundle.resolve())
