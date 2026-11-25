"""Fixed study scope, separate destinations, and audited reuse helpers."""
import csv
import io
from scripts.trainable_or_3way.reinforcement_temporal.common import *

FORMAT = "six_way_figures_v1"
BASE = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports/six_way_figures_v1"
TEMPORAL = ROOT / "reports/fav_avdf1m_reinforcement_temporal_v1"
ORDER = ROOT / "reports/fav_avdf1m_reinforcement_p3a_v2"
PGF = ROOT / "reports/trainable_or_pgf_optimized_v1/source_PGF/seed_42"
FAV_PGF = ROOT / "reports/fav_pgf_common_bc_v1"
WS = ROOT / "reports/workstation_trainable_final_fusion_v1/source_AVDF1M/seed_42"
DATASETS = ("FAV", "AVDF1M", "PGF")
DIRECTIONS = tuple((s, t) for s in DATASETS for t in DATASETS if s != t)
NEW_DIRECTIONS = tuple((s, t) for s, t in DIRECTIONS if "PGF" in (s, t))
SEEDS = (42, 43, 44)
CONDITIONS = ("B", "C")
REPLICATES = 5000


def implementation():
    result = code_identity()
    for p in BASE.glob("*.py"):
        if not p.name.startswith("test_"):
            result[p.relative_to(ROOT).as_posix()] = sha(p)
    for rel in ("scripts/trainable_or_3way/data.py", "scripts/trainable_or_3way/pgf_source/common.py",
                "scripts/trainable_or_3way/reinforcement/inputs.py"):
        result[rel] = sha(ROOT / rel)
    return result


def destination(path):
    path = resolve(path)
    require(path.is_relative_to(ROOT / "reports") and path != ROOT / "reports",
            "Use a new dedicated subfolder of reports")
    safe_output(path, [TEMPORAL, ORDER, PGF, FAV_PGF, WS, ROOT / "output",
        ROOT / "reports/fav_avdf1m_reinforcement_v1", ROOT / "reports/fav_avdf1m_reinforcement_p2_v1",
        ROOT / "reports/hierarchical_score_fusion_gru_temporal_attention_fav_v1",
        ROOT / "reports/trainable_or_3way_fav_size_matched_v1", ROOT / "reports/trainable_or_pgf_v1",
        Path("D:/datasets")])
    for parent in path.parents:
        if parent == ROOT / "reports":
            break
        require(not (parent / "RUN.json").exists() and not (parent / "COMPLETED.json").exists(),
                "Output is inside another experiment")
    if path.exists():
        if (path / "RUN.json").exists():
            require(read(path / "RUN.json").get("format") == FORMAT, "Output belongs to another workflow")
        else:
            require(not any(p.name != ".writer.lock" for p in path.iterdir()), "Output is not empty")
    return path


def csv_write(path, rows):
    require(rows, "Empty result table")
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    atomic_text(path, buf.getvalue())


def align(data, cohort):
    """Align on IDs, verify labels and clusters; only evidence-backed unknown labels may be filled."""
    ids = data["sample_id"].tolist()
    wanted = cohort["sample_id"].tolist()
    require(len(set(ids)) == len(ids) and len(set(wanted)) == len(wanted) and set(ids) == set(wanted),
            "Score/cohort IDs differ or contain duplicates")
    lookup = {s: i for i, s in enumerate(ids)}
    order = np.array([lookup[s] for s in wanted])
    result = {k: np.asarray(v)[order] for k, v in data.items()}
    known = result["labels"] >= 0
    require(result["labels"].shape == cohort["labels"].shape
            and np.array_equal(result["labels"][known], cohort["labels"][known]), "Known labels disagree")
    if "cluster" in result:
        require(np.array_equal(result["cluster"], cohort["cluster"]), "Cluster identity differs")
    if "recorded_group" in result:
        require(np.array_equal(result["recorded_group"], cohort["cluster"]), "Stored groups differ")
    result.update({k: np.asarray(cohort[k]).copy() for k in ("sample_id", "labels", "cluster")})
    return result


class Evidence:
    """Hash only read inputs; never open historical files for writing."""
    def __init__(self):
        self.files = {}

    def file(self, path, expected=None):
        path = resolve(path)
        actual = sha(path)
        require(expected is None or actual == expected, "Historical input changed: " + str(path))
        if str(path) in self.files:
            require(self.files[str(path)] == actual, "Input changed during inspection: " + str(path))
        self.files[str(path)] = actual
        return path

    def committed(self, folder, name):
        folder = resolve(folder)
        self.file(folder / "COMPLETED.json")
        record = read(folder / "COMPLETED.json")
        mapping = {k.replace("\\", "/"): v for k, v in record["files"].items()}
        require(name.replace("\\", "/") in mapping, "File not bound by receipt: " + str(folder / name))
        return self.file(child(folder, name), mapping[name.replace("\\", "/")])

    def job(self, folder):
        folder = resolve(folder)
        self.file(folder / "COMPLETED.json")
        for name, expected in read(folder / "COMPLETED.json")["files"].items():
            self.file(child(folder, name), expected)
        return folder

    def verify(self):
        bar = Progress("VERIFY preserved inputs", len(self.files))
        for i, (p, expected) in enumerate(self.files.items(), 1):
            require(sha(p) == expected, "Preserved input changed: " + p)
            bar.update(i)


def import_job(source, target, evidence):
    """Copy small completed result files; no copying mutable training/recovery files."""
    source, target = Path(source), Path(target)
    evidence.job(source)
    receipt = read(source / "COMPLETED.json")
    identity = digest(dict(origin=str(source.resolve()), receipt=sha(source / "COMPLETED.json")))
    if completed(target, identity):
        return target
    from scripts.trainable_or_3way.reinforcement.common import atomic_bytes
    for name in receipt["files"]:
        atomic_bytes(child(target, name), child(source, name).read_bytes())
    atomic_json(target / "REUSE.json", dict(origin=str(source), receipt_sha256=sha(source / "COMPLETED.json"),
                                          fitting=False, inference=False, resampling=False))
    finish(target, identity, list(receipt["files"]) + ["REUSE.json"])
    return target
