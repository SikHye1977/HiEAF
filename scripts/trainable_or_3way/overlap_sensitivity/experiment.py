"""CPU-only score sensitivity: preflight, run/resume, status, verify. No fitting/inference."""
import argparse
import csv
import io
import json
import sys
from pathlib import Path
import numpy as np
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.trainable_or_3way.reinforcement.common import (
    ROOT, require, read, resolve, sha, digest, runtime, immutable, atomic_json, atomic_text,
    writer_lock, completed, finish, Progress)
from scripts.trainable_or_3way.overlap_sensitivity.inputs import (
    FORMAT, DEFAULTS, OUTPUT, METHODS, NAMES, COHORTS, prepare, check_output, implementation)
from scripts.trainable_or_3way.overlap_sensitivity.statistics import (
    METRICS, points, bootstrap, summarize, interval)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run", "status", "verify"):
        q = sub.add_parser(command)
        q.add_argument("--output", default=OUTPUT)
        if command in ("preflight", "run"):
            for key, value in DEFAULTS.items():
                q.add_argument("--"+key, default=value)
            q.add_argument("--replicates", type=int, default=5000)
            q.add_argument("--seed", type=int, default=1701)
            q.add_argument("--checkpoint-every", type=int, default=25)
            q.add_argument("--max-attempts", type=int, default=50000)
    return p


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def conclusions(point, summary):
    result = {}
    full = summary["comparisons"]["full"]["M4-M1"]["roc_auc"]["delta"]
    for name in COHORTS[1:]:
        c = summary["comparisons"][name]["M4-M1"]
        delta = c["roc_auc"]["delta"]
        if full > 0 and delta > 0:
            direction = "positive_delta_retained"
        elif full < 0 and delta < 0:
            direction = "negative_delta_retained"
        elif full*delta < 0:
            direction = "direction_reversed"
        else:
            direction = "zero_or_changed_from_zero"
        ci = c["roc_auc"]["family_ci"] if name == COHORTS[1] else c["roc_auc"]["ci95"]
        support = "positive" if ci[0] > 0 else "negative" if ci[1] < 0 else "includes_zero"
        result[name] = dict(auc_direction=direction, auc_interval_evidence=support,
                           ci_level=.9875 if name == COHORTS[1] else .95,
                           fake_f1_increased=c["fake_f1"]["delta"] > 0,
                           real_fpr_increased=c["real_fpr"]["delta"] > 0,
                           proposed_auc_below_half=point[name]["M4"]["roc_auc"] < .5)
    return result


def export(output, jobs, descriptions):
    rows, contrasts, verdicts = [], [], []
    for name, job in jobs.items():
        folder = output/name
        point = read(folder/"METRICS.json")
        boot = read(folder/"BOOTSTRAP.json")
        inference = read(folder/"INTERPRETATION.json")
        fields = {k: job[k] for k in ("source", "target", "condition")}
        for cohort in COHORTS:
            count = descriptions[name][cohort]
            for method in METHODS:
                r = dict(**fields, cohort=cohort, method=method, model=NAMES[method], **count)
                r.update({k: point[cohort][method][k] for k in METRICS})
                rows.append(r)
            for right in ("M1", "M0"):
                r = dict(**fields, cohort=cohort, comparison="M4-"+right, **count)
                for metric, entry in boot["comparisons"][cohort]["M4-"+right].items():
                    r[metric+"_delta"] = entry["delta"]
                    for key in ("ci95", "family_ci", "change_ci95"):
                        for i, bound in enumerate(("low", "high")):
                            r[metric+"_"+key+"_"+bound] = None if entry[key] is None else entry[key][i]
                    r[metric+"_delta_change_from_full"] = entry["delta_change_from_full"]
                contrasts.append(r)
        verdicts.append(dict(**fields, result=inference))
    atomic_json(output/"SUMMARY.json", dict(rows=rows, comparisons=contrasts, interpretation=verdicts))
    atomic_text(output/"SUMMARY.csv", csv_text(rows))
    atomic_text(output/"COMPARISONS.csv", csv_text(contrasts))
    lines = ["# 공통 인물 제외 민감도 분석", "",
             "기존 저장 예측·융합 계수·임계값(0.5)을 고정하고 FAV↔AVDF1M B/C 네 조건을 분석했다.",
             "주 대조는 Proposed(M4) − an artifact-only baseline with learned score adjustment(M1)이다.",
             "전체/명시적 관련 인물 제외/파일명 후보까지 제외를 같은 모델로 비교한다. 표의 수치 단위는 % 및 percentage points(pp)다.", "",
             "| Source → Target | 조건 | 표본 | n (Real/Fake) | M1 AUC | M4 AUC | ΔAUC | ΔFake F1 | ΔReal FPR ↓ |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in contrasts:
        if r["comparison"] != "M4-M1":
            continue
        s = [x for x in rows if all(x[k] == r[k] for k in ("source", "target", "condition", "cohort"))]
        values = {x["method"]: x["roc_auc"]*100 for x in s}
        lines.append(f"| {r['source']} → {r['target']} | {r['condition']} | {r['cohort']} | "
                     f"{r['n']} ({r['real']}/{r['fake']}) | {values['M1']:.2f} | {values['M4']:.2f} | "
                     f"{100*r['roc_auc_delta']:+.2f} | {100*r['fake_f1_delta']:+.2f} | {100*r['real_fpr_delta']:+.2f} |")
    lines += ["", "주 필터의 AUC 비교에는 네 조건의 Bonferroni 보정 98.75% 개별 구간(가족 수준 95%)을 적용했다.",
              "보조 지표·파일명 후보 필터·전체 대비 Δ 변화는 탐색적 95% 구간이다. 자세한 구간은 COMPARISONS.csv에 있다.", ""]
    labels = dict(positive_delta_retained="AUC 개선 방향 유지", negative_delta_retained="AUC 저하 방향 유지",
                  direction_reversed="AUC 차이의 부호 반전", zero_or_changed_from_zero="0 또는 0과의 변화")
    evidence = dict(positive="구간 전체가 0보다 큼", negative="구간 전체가 0보다 작음", includes_zero="구간이 0을 포함")
    for v in verdicts:
        r = v["result"]["metadata_unshared"]
        text = f"- {v['source']}→{v['target']} {v['condition']}: {labels[r['auc_direction']]}; {evidence[r['auc_interval_evidence']]} ."
        if r["fake_f1_increased"] and r["real_fpr_increased"]:
            text += " Fake F1 향상과 Real FPR 증가가 함께 나타난다."
        if r["proposed_auc_below_half"]:
            text += " 제안 방법의 절대 AUC는 0.5 미만이다."
        lines.append(text)
    lines += ["", "해석 범위", "",
              "- 양의 Δ가 남는 것과 통계 구간이 양수인 것을 구분한다. 모든 전이 방향에서 개선된다는 결론을 자동으로 내리지 않는다.",
              "- 제외 전후에는 표본 구성과 Fake 비율이 달라진다. 같은 표본의 M4−M1 차이가 주 비교이며 Δ 변화 자체는 인물 중복의 인과 효과가 아니다.",
              "- source train+dev의 명시적 source/target 인물을 모두 사용한다. 파일명 후보는 의미가 미확인인 추가 제외 기준이다.",
              "- 이 분석은 알려진 공통 인물에 대한 사후 민감도 분석이다. FAV 내부 split의 관련 인물 교차, 공식 분할과의 차이, 원본 녹화물/미확인 인물 관계를 수정한 재학습은 아니다.",
              "- PGF는 검증된 공통 인물 매핑이 없어 이번 분석에 포함하지 않는다. 이 결과로 여섯 방향 모두에서 중복 영향이 배제됐다고 표현하지 않는다.",
              "- bootstrap 단위는 기존 manifest의 FAV source 인물 또는 AVDF1M 인물이다. 가중치·학습 seed 변동 및 모든 donor 관계의 독립성을 평가하는 구간은 아니다.",
              "- 기존 전체 결과는 보존된다. SUMMARY.csv는 0~1 단위이고 이 보고서만 %/pp로 표시한다.", ""]
    atomic_text(output/"REPORT.md", "\n".join(lines))


def execute(output, prepared, config):
    run = dict(format=FORMAT, inputs=prepared["pins"].files, paths={k: str(p) for k, p in prepared["paths"].items()},
               implementation=implementation(), runtime=runtime(), bootstrap=config,
               protocol=dict(methods=METHODS, cohorts=COHORTS, threshold=.5,
                             source_exclusion="union of explicit persons across source train + dev",
                             candidate_exclusion="also union filename-only candidates in both source and target",
                             primary="M4-M1 AUC in metadata_unshared, four B/C cross-direction contrasts",
                             primary_family_size=4, individual_ci=.9875,
                             no_fit=True, no_inference=True),
               cohort_counts=prepared["descriptions"])
    # JSON normalization avoids tuple/list mismatch on a resumed run.
    run = json.loads(json.dumps(run))
    identity = digest(run)
    with writer_lock(output):
        immutable(output/"RUN.json", run)
        if completed(output, identity):
            prepared["pins"].verify()
            print("[COMPLETE] Verified existing sensitivity analysis; nothing rerun", flush=True)
            return
        try:
            atomic_json(output/"STATUS.json", dict(status="RUNNING", identity=identity))
            mask_files = []
            for name, plan in prepared["plans"].items():
                rows = [dict(sample_id=r["sample_id"], label=r["label"],
                             keep_explicit=int(plan["masks"][1, i]),
                             keep_with_filename_candidates=int(plan["masks"][2, i]),
                             explicit_overlap=";".join(plan["overlap"][i]),
                             candidate_overlap=";".join(plan["candidates"][i])) for i, r in enumerate(plan["rows"])]
                relative = "masks/"+name+".csv"
                content = csv_text(rows)
                if (output/relative).exists():
                    require((output/relative).read_text(encoding="utf-8") == content,
                            "Existing mask list changed")
                else:
                    atomic_text(output/relative, content)
                mask_files.append(relative)
            total = len(prepared["jobs"])
            verified = {name for name in prepared["jobs"] if completed(output/name, digest(dict(run=identity, job=name)))}
            done = len(verified)
            progress = Progress("SENSITIVITY jobs (job-count ETA, sizes differ)", total, output, resumed=done)
            for name, job in prepared["jobs"].items():
                folder = output/name
                job_identity = digest(dict(run=identity, job=name))
                if name in verified:
                    print("[RESUME] verified completed " + name, flush=True)
                    continue
                else:
                    print("[STAGE] " + name + ": fixed-score metrics and paired bootstrap", flush=True)
                    point = points(job["data"], job["masks"], job["recorded"])
                    atomic_json(folder/"METRICS.json", point)
                    values, counters = bootstrap(job["data"], job["masks"], folder, job_identity, **config)
                    result = summarize(point, values, counters)
                    atomic_json(folder/"BOOTSTRAP.json", result)
                    atomic_json(folder/"INTERPRETATION.json", conclusions(point, result))
                    prepared["pins"].verify()
                    finish(folder, job_identity, ["METRICS.json", "BOOTSTRAP.json", "INTERPRETATION.json", "bootstrap_state.npz"])
                done += 1
                progress.update(done, detail=name)
            export(output, prepared["jobs"], prepared["descriptions"])
            prepared["pins"].verify()
            atomic_json(output/"PRESERVED_INPUTS.json", dict(status="PASS", files=prepared["pins"].files))
            files = ["RUN.json", "SUMMARY.json", "SUMMARY.csv", "COMPARISONS.csv", "REPORT.md", "PRESERVED_INPUTS.json", *mask_files]
            for name in prepared["jobs"]:
                files.extend(name+"/"+f for f in ["COMPLETED.json", "METRICS.json", "BOOTSTRAP.json", "INTERPRETATION.json", "bootstrap_state.npz"])
            finish(output, identity, files)
            atomic_json(output/"STATUS.json", dict(status="COMPLETE", identity=identity))
            print("[COMPLETE] " + str(output/"REPORT.md"), flush=True)
        except BaseException as exc:
            atomic_json(output/"STATUS.json", dict(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                                                  identity=identity, error=str(exc)))
            raise


def run(args):
    output = resolve(args.output)
    if args.command == "status":
        if not output.exists():
            print("[NOT STARTED] " + str(output))
            return
        for path in [output/"STATUS.json", output/"progress.json", *sorted(output.glob("*/progress.json"))]:
            if path.is_file():
                print(str(path) + "\n" + json.dumps(read(path), ensure_ascii=False, indent=2))
        return
    if args.command == "verify":
        r = read(output/"RUN.json")
        require(r.get("format") == FORMAT, "Not an overlap sensitivity run")
        require(completed(output, digest(r)), "Analysis not complete")
        for path, expected in r["inputs"].items():
            require(sha(path) == expected, "Historical input changed: " + path)
        require(r["implementation"] == implementation(), "Analysis code changed since run")
        print("[VERIFY] PASS: output receipts, code and preserved consumed inputs")
        return
    config = {k: getattr(args, k) for k in ("replicates", "seed", "checkpoint_every", "max_attempts")}
    require(config["replicates"] >= 2 and config["max_attempts"] >= config["replicates"] and
            config["checkpoint_every"] > 0 and config["seed"] >= 0, "Invalid analysis settings")
    paths = {k: getattr(args, k) for k in DEFAULTS}
    output = check_output(output, [resolve(p) for p in paths.values()])
    print("[PREFLIGHT] Reading only committed predictions and identity-audit metadata", flush=True)
    prepared = prepare(paths)
    print(json.dumps(prepared["descriptions"], ensure_ascii=False, indent=2))
    print("[PREFLIGHT] PASS: no metric estimation/bootstrap/model inference performed by preflight", flush=True)
    if args.command == "run":
        execute(output, prepared, config)


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except KeyboardInterrupt:
        print("[STOPPED] Partial bootstrap retained. Resume with the identical run command.", flush=True)
        sys.exit(130)
    except Exception as exc:
        print("[FAILED] " + str(exc) + "; no successful completion for failed stage", flush=True)
        raise
