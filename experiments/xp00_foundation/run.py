#!/usr/bin/env python3
"""XP0 — Data, splits, calibration set, harness.

Nothing here is a result about fire detection. This experiment exists to make
every *later* result trustworthy: it freezes the data, proves the class mapping
rather than assuming it, and drives a COCO-pretrained YOLOv8n end to end through
the shared harness so that when XP1's real numbers arrive, the only new thing in
the pipeline is the model.

Stages
    1. scan      — read D-Fire, verify the class ids against published box counts
    2. splits    — freeze train/val/test + the small-plume test view
    3. calib     — freeze the 500-image INT8 calibration set
    4. smoke     — COCO-pretrained YOLOv8n through the full harness, on-device
    5. repro     — run accuracy twice, assert the prediction fingerprints match

Usage
    python run.py                    # everything
    python run.py --stages splits    # just re-freeze the splits
    python run.py --quick            # smaller smoke test, for wiring checks only
                                     # (NOT a protocol-compliant measurement)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lib import data as dataset          # noqa: E402
from lib import evaluator                # noqa: E402

HERE = Path(__file__).resolve().parent
DFIRE = dataset.DATA / "dfire"
RAW = REPO / "results" / "raw"


def log(msg: str) -> None:
    print(f"[xp00] {msg}", flush=True)


def _quick_subset(samples: list, n: int = 200) -> list:
    """A representative slice for wiring checks.

    Strided, not ``[:n]``: D-Fire's filenames are grouped by source, so the first
    200 test images are all background and carry no ground truth at all — which
    makes mAP undefined and the check meaningless. Striding is deterministic, so
    the quick path stays reproducible.
    """
    if len(samples) <= n:
        return samples
    step = len(samples) // n
    return samples[::step][:n]


# --------------------------------------------------------------------------

def stage_scan() -> dict:
    """Read the dataset and establish what the class ids actually mean."""
    log("scanning D-Fire …")
    train_pool = dataset.scan_dataset(DFIRE / "train" / "images", DFIRE / "train" / "labels")
    test_pool = dataset.scan_dataset(DFIRE / "test" / "images", DFIRE / "test" / "labels")
    log(f"train pool {len(train_pool)} · test pool {len(test_pool)}")

    check = dataset.verify_class_map(train_pool + test_pool)
    log(f"class map: observed {check['observed_by_id']} -> derived {check['derived_map']}")
    if not check["consistent"]:
        raise SystemExit(
            f"CLASS_NAMES={dataset.CLASS_NAMES} contradicts the data: the larger class is "
            f"{check['derived_map']}. Fix lib/data.CLASS_NAMES before going further — every "
            f"per-class number in the series depends on this mapping."
        )
    log("class map corroborated by the published box counts ✓")
    return {"train_pool": train_pool, "test_pool": test_pool, "class_check": check}


def stage_splits(scan: dict) -> dict:
    log("freezing splits …")
    manifest = dataset.build_splits(scan["train_pool"], scan["test_pool"])
    log(f"train {manifest['counts']['train']} · val {manifest['counts']['val']} · "
        f"test {manifest['counts']['test']} · small-plume test {manifest['test_small_plume']}")
    log(f"checksums {manifest['checksums']}")
    return manifest


def stage_calib() -> dict:
    log("building the INT8 calibration set (sweeping train-set luminance stats) …")
    train = dataset.load_samples("train")
    t0 = time.perf_counter()
    stats = {}
    for i, s in enumerate(train, 1):
        stats[s.rel] = dataset.image_stats(s.image)
        if i % 2000 == 0:
            log(f"  … {i}/{len(train)} images scanned")
    log(f"  luminance sweep took {time.perf_counter() - t0:.1f}s")
    meta = dataset.build_calibration_set(train, stats=stats)
    log(f"calibration set: {meta['composition']}")
    return meta


def stage_smoke(quick: bool) -> dict:
    """COCO-pretrained YOLOv8n through the whole harness.

    The accuracy number here is expected to be near zero and that is the *point*:
    COCO has no fire or smoke class, so this measures plumbing, not detection.
    A non-trivial mAP would mean the class mapping or the GT wiring is wrong.
    """
    from lib.detectors import UltralyticsDetector

    weights = REPO / "weights" / "yolov8n.pt"
    weights.parent.mkdir(exist_ok=True)
    det = UltralyticsDetector(weights, input_res=evaluator.DEFAULT_INPUT_RES, half=True,
                              name="yolov8n_coco_pretrained")

    test = dataset.load_samples("test")
    acc_images = _quick_subset(test) if quick else test
    log(f"accuracy pass over {len(acc_images)} test images …")
    t0 = time.perf_counter()
    acc = evaluator.evaluate_accuracy(det, acc_images)
    log(f"  mAP50 {acc['map50']} · mAP50-95 {acc['map5095']} · "
        f"fingerprint {acc['fingerprint']} ({time.perf_counter() - t0:.1f}s)")

    log("timing pass …")
    timing_kwargs = dict(runs=1, warmup=10, frames=50, allow_short=True) if quick else {}
    jetson = evaluator.measure_latency(det, [s.image for s in test[:64]], **timing_kwargs)
    jetson |= evaluator.measure_memory()
    log(f"  {jetson['latency_ms_median']} ms median · {jetson['fps_batch1']} fps · "
        f"{jetson.get('power_w_mean')} W · {jetson.get('temp_c_peak')} °C")

    record = evaluator.results_record(
        model_id="yolov8n_coco_pretrained",
        fmt="pt",
        params_m=det.params_m(),
        size_disk_mb=det.size_disk_mb(),
        accuracy=acc,
        jetson=jetson,
        notes="XP0 dry run: COCO-pretrained weights, NOT fine-tuned on D-Fire. "
              "Near-zero mAP is the expected and correct outcome — COCO has no fire or "
              "smoke class. This record exists to prove the harness works end to end.",
    )
    if quick:
        record["notes"] += " QUICK MODE: reduced frame counts, NOT protocol-compliant."
    # Hand the exact image list to the repro stage. Re-deriving it there once let
    # the two runs score different subsets and look like nondeterminism.
    return {"record": record, "detector": det, "acc": acc, "quick": quick,
            "acc_images": acc_images}


def stage_repro(smoke: dict) -> dict:
    """Accuracy is deterministic given fixed weights and fixed data, so instead of
    error bars we assert bit-reproducibility (PLAN.md §0)."""
    log("re-running accuracy to assert bit-reproducibility …")
    det = smoke["detector"]
    second = evaluator.evaluate_accuracy(det, smoke["acc_images"])
    first_fp, second_fp = smoke["acc"]["fingerprint"], second["fingerprint"]
    ok = first_fp == second_fp
    log(f"  run 1 {first_fp} · run 2 {second_fp} · {'MATCH ✓' if ok else 'MISMATCH ✗'}")
    if not ok:
        raise SystemExit(
            "predictions are not bit-reproducible across two identical runs — the harness "
            "has a nondeterminism bug and no accuracy number in this repo can be trusted "
            "until it is found.")
    return {"bit_reproducible": ok, "fingerprint": first_fp}


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*",
                    default=["scan", "splits", "calib", "smoke", "repro"],
                    choices=["scan", "splits", "calib", "smoke", "repro"])
    ap.add_argument("--quick", action="store_true",
                    help="reduced frame counts — wiring check only, not a valid measurement")
    args = ap.parse_args()

    out: dict = {"protocol_version": evaluator.PROTOCOL_VERSION, "stages_run": args.stages}
    scan = smoke = None

    if "scan" in args.stages:
        scan = stage_scan()
        out["class_check"] = scan["class_check"]
    if "splits" in args.stages:
        if scan is None:
            scan = stage_scan()
        out["splits"] = stage_splits(scan)
    if "calib" in args.stages:
        out["calibration"] = stage_calib()
    if "smoke" in args.stages:
        smoke = stage_smoke(args.quick)
        out["dry_run"] = smoke["record"]
    if "repro" in args.stages:
        if smoke is None:
            raise SystemExit("--stages repro requires smoke in the same invocation")
        out["reproducibility"] = stage_repro(smoke)

    if "smoke" in args.stages:
        evaluator.write_results(smoke["record"], HERE / "results.json")
        evaluator.write_results(smoke["record"], RAW / "xp00_yolov8n_coco_dryrun.json")
        log(f"wrote {HERE / 'results.json'}")

    (HERE / "xp00_summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    log("done.")


if __name__ == "__main__":
    main()
