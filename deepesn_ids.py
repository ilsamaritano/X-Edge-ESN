import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

from esn import *
from readout import *


BASE_SEED = 1337


# =============================================================================
# LOAD DATA
# =============================================================================

def load(split, data_dir):
    return np.load(f"{data_dir}/{split}.npy")


def prepare_data(data_dir, classes_to_remove=None):
    X_train = torch.from_numpy(load("X_train_seq", data_dir)).float()
    X_val   = torch.from_numpy(load("X_val_seq", data_dir)).float()
    X_test  = torch.from_numpy(load("X_test_seq", data_dir)).float()

    Y_train = torch.from_numpy(load("y_train_seq", data_dir)).long()
    Y_val   = torch.from_numpy(load("y_val_seq", data_dir)).long()
    Y_test  = torch.from_numpy(load("y_test_seq", data_dir)).long()

    if classes_to_remove is None:
        classes_to_remove = [13, 8, 9, 14]

    classes_to_remove = torch.tensor(classes_to_remove, dtype=torch.long)

    tr_mask   = ~torch.isin(Y_train, classes_to_remove)
    vl_mask   = ~torch.isin(Y_val, classes_to_remove)
    test_mask = ~torch.isin(Y_test, classes_to_remove)

    X_train = X_train[tr_mask]
    X_val   = X_val[vl_mask]
    X_test  = X_test[test_mask]

    Y_train = Y_train[tr_mask]
    Y_val   = Y_val[vl_mask]
    Y_test  = Y_test[test_mask]

    classi_uniche = torch.unique(Y_train)

    max_val = max(
        Y_train.max().item(),
        Y_val.max().item(),
        Y_test.max().item()
    ) + 1

    remap = torch.full((max_val,), -1, dtype=torch.long)

    for nuovo_idx, vecchio_val in enumerate(classi_uniche):
        remap[vecchio_val] = nuovo_idx

    Y_train = remap[Y_train]
    Y_val   = remap[Y_val]
    Y_test  = remap[Y_test]

    assert (Y_train >= 0).all()
    assert (Y_val >= 0).all()
    assert (Y_test >= 0).all()

    num_classes = len(classi_uniche)
    class_names = [f"class_{i}" for i in range(num_classes)]

    print("Original remaining classes:", classi_uniche.tolist())
    print("Unique remapped train labels:", torch.unique(Y_train).tolist())
    print("Train:", X_train.shape, Y_train.shape)
    print("Val:  ", X_val.shape, Y_val.shape)
    print("Test: ", X_test.shape, Y_test.shape)
    print("Num classes:", num_classes)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, num_classes, class_names


# =============================================================================
# TIMER
# =============================================================================

class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_time = time.time() - self.start_time


# =============================================================================
# DATASET
# =============================================================================

class PrecomputedSequenceDataset(Dataset):
    """
    Dataset per sequenze già nel formato [N, T, F].
    Nessun transpose, nessuna mask, nessun padding.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# RESERVOIR STATES
# =============================================================================

@torch.no_grad()
def compute_reservoir_states(model, dataloader, device):
    """
    Input:
      x [B, T, F]
      y [B]

    Output:
      features [N, D]
    """

    all_features = []
    comp_time = 0.0

    model.eval()

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)

        with Timer() as timer:
            reservoir_features, _ = model(x)

        comp_time += timer.elapsed_time
        all_features.append(reservoir_features.cpu())

    return torch.cat(all_features, dim=0), comp_time


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_sequence_level(model, loader, device, num_classes, class_names):
    model.eval()
    model.readout.eval()

    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        states, _ = model(x)
        scores = model.readout(states)
        preds = scores.argmax(dim=-1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return report


def format_report(report, class_names):
    lines = [
        f"{'Class':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>9}",
        "-" * 46
    ]

    for name in class_names:
        r = report[name]
        lines.append(
            f"{name:<12} {r['precision']:>7.3f} {r['recall']:>7.3f} "
            f"{r['f1-score']:>7.3f} {int(r['support']):>9,}"
        )

    lines.append("-" * 46)

    ma = report["macro avg"]
    lines.append(
        f"{'macro avg':<12} {ma['precision']:>7.3f} {ma['recall']:>7.3f} "
        f"{ma['f1-score']:>7.3f} {int(ma['support']):>9,}"
    )

    return "\n".join(lines)


# =============================================================================
# INFERENCE TIMING
# =============================================================================

@torch.no_grad()
def measure_esn_inference(
    model,
    loader,
    device,
    n_warmup: int = 5,
):
    model.eval()
    model.readout.eval()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    print("[Latency ESN] Warmup...")

    for i, (x, y) in enumerate(loader):
        if i >= n_warmup:
            break

        x = x.to(device, non_blocking=True)

        if hasattr(model, "predict"):
            _ = model.predict(x)
        else:
            states, _ = model(x)
            _ = model.readout(states)

    sync()

    total_samples = 0
    latency_per_sample = []
    batch_latency_ms = []
    reservoir_times = []
    readout_times = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        B = x.shape[0]

        sync()
        t0 = time.perf_counter()
        states, _ = model(x)
        sync()
        t_res_ms = (time.perf_counter() - t0) * 1000.0

        sync()
        t0 = time.perf_counter()
        _ = model.readout(states)
        sync()
        t_read_ms = (time.perf_counter() - t0) * 1000.0

        t_total_ms = t_res_ms + t_read_ms

        batch_latency_ms.append(t_total_ms)
        latency_per_sample.append(t_total_ms / B)
        reservoir_times.append(t_res_ms / B)
        readout_times.append(t_read_ms / B)

        total_samples += B

    batch_latency_ms = np.array(batch_latency_ms, dtype=np.float64)
    latency_per_sample = np.array(latency_per_sample, dtype=np.float64)
    reservoir_times = np.array(reservoir_times, dtype=np.float64)
    readout_times = np.array(readout_times, dtype=np.float64)

    sync()
    t_start = time.perf_counter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        states, _ = model(x)
        _ = model.readout(states)

    sync()
    elapsed_s = time.perf_counter() - t_start

    throughput_sps = total_samples / elapsed_s if elapsed_s > 0 else 0.0

    return {
        "latency_batch_mean_ms":  float(np.mean(batch_latency_ms)),
        "latency_batch_std_ms":   float(np.std(batch_latency_ms)),
        "latency_sample_mean_ms": float(np.mean(latency_per_sample)),
        "latency_sample_std_ms":  float(np.std(latency_per_sample)),
        "reservoir_ms":           float(np.mean(reservoir_times)),
        "readout_ms":             float(np.mean(readout_times)),
        "throughput_sps":         float(throughput_sps),
        "elapsed_s":              float(elapsed_s),
        "total_samples":          int(total_samples),
    }


# =============================================================================
# STATS
# =============================================================================

def aggregate_mean_std(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "values": arr.tolist(),
    }


def summarize_reinitializations(all_reinit_results):
    summary_stats = {}

    summary_stats["val_err"] = aggregate_mean_std(
        [r["val_err"] for r in all_reinit_results]
    )

    summary_stats["val_f1"] = aggregate_mean_std(
        [r["val_f1"] for r in all_reinit_results]
    )

    summary_stats["test_f1"] = aggregate_mean_std(
        [r["test_f1"] for r in all_reinit_results]
    )

    summary_stats["training_time_s"] = aggregate_mean_std(
        [r["training_time_s"] for r in all_reinit_results]
    )

    summary_stats["train_state_extraction_s"] = aggregate_mean_std(
        [r["training_breakdown_s"]["train_state_extraction_s"] for r in all_reinit_results]
    )

    summary_stats["val_state_extraction_s"] = aggregate_mean_std(
        [r["training_breakdown_s"]["val_state_extraction_s"] for r in all_reinit_results]
    )

    summary_stats["readout_fit_s"] = aggregate_mean_std(
        [r["training_breakdown_s"]["readout_fit_s"] for r in all_reinit_results]
    )

    timing_keys = [
        "latency_batch_mean_ms",
        "latency_batch_std_ms",
        "latency_sample_mean_ms",
        "latency_sample_std_ms",
        "reservoir_ms",
        "readout_ms",
        "throughput_sps",
        "elapsed_s",
        "total_samples",
    ]

    for key in timing_keys:
        vals = [
            r["timing"][key]
            for r in all_reinit_results
            if r["timing"].get(key) is not None
        ]

        if len(vals) > 0:
            summary_stats[key] = aggregate_mean_std(vals)

    return summary_stats


# =============================================================================
# MAIN TOOLKIT
# =============================================================================

def run_esn_toolkit(
    data_dir: str,
    output: str,
    device_str: str,
    esn_units: int = 256,
    esn_layers: int = 1,
    spectral_radius: float = 0.3,
    input_scaling: float = 1.0,
    bias_scaling: float = 0.0,
    leaky: float = 0.9,
    spectral_radius_hidden: float = 0.7,
    bias_scaling_hidden: float = 0.5,
    input_scaling_hidden: float = 0.5,
    leaky_hidden: float = 0.9,
    readout_reg_min_exp: int = -8,
    readout_reg_max_exp: int = 5,
    readout_reg_steps: int = 15,
    batch_size: int = 64,
    fit_batch_size: int = 50_000,
    num_workers: int = 0,
    num_reinit: int = 10,
    use_parallel: bool = True,
    use_compile: bool = False,
    last_layer: bool = False,
    sequences: bool = False,
    mean: bool = False,
    classes_to_remove=None,
):
    X_train, X_val, X_test, Y_train, Y_val, Y_test, NUM_CLASSES, CLASS_NAMES = prepare_data(
        data_dir=data_dir,
        classes_to_remove=classes_to_remove,
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "esn_train.log"

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    device = torch.device(device_str)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    input_size = X_train.shape[-1]

    log(f"[Device] {device}")
    log(f"[Data] train={list(X_train.shape)} val={list(X_val.shape)} test={list(X_test.shape)}")
    log(f"[Data] num_classes={NUM_CLASSES} input_size={input_size}")

    y_train_oh = F.one_hot(Y_train, num_classes=NUM_CLASSES).float()
    y_val_oh   = F.one_hot(Y_val, num_classes=NUM_CLASSES).float()
    y_test_oh  = F.one_hot(Y_test, num_classes=NUM_CLASSES).float()

    train_loader = DataLoader(
        PrecomputedSequenceDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        PrecomputedSequenceDataset(X_val, Y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        PrecomputedSequenceDataset(X_test, Y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    readout_regularizer = np.logspace(
        readout_reg_min_exp,
        readout_reg_max_exp,
        readout_reg_steps,
    )

    best_reinit_err = float("inf")
    best_reinit_ckpt = None
    all_reinit_results = []

    for reinit in range(1, num_reinit + 1):
        seed = BASE_SEED + reinit

        torch.manual_seed(seed)
        np.random.seed(seed)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        log(f"\n{'=' * 60}")
        log(f"[Reinit {reinit}/{num_reinit}] seed={seed}")

        model = DeepESN(
            input_size=input_size,
            units=esn_units,
            num_layers=esn_layers,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling,
            spectral_radius=spectral_radius,
            spectral_radius_hidden=spectral_radius_hidden,
            bias_scaling_hidden=bias_scaling_hidden,
            input_scaling_hidden=input_scaling_hidden,
            leaky_hidden=leaky_hidden,
            leaky=leaky,
            readout_regularizer=readout_regularizer,
            type="bi",
            task="classification",
            score="f1",
            last_layer=last_layer,
            sequences=sequences,
            mean=mean,
            use_parallel=use_parallel,
        ).to(device)

        if use_compile:
            try:
                model = torch.compile(model)
                log("[Compile] torch.compile enabled.")
            except Exception as e:
                log(f"[Compile] torch.compile failed, continuing without compile. Error: {e}")

        out_size = esn_units * 2 * esn_layers

        n_params = sum(p.numel() for p in model.parameters())
        log(f"[Model] ESN out_size={out_size} params={n_params:,}")
        log(
            "[Config] "
            f"units={esn_units}, layers={esn_layers}, "
            f"sr={spectral_radius}, sr_hidden={spectral_radius_hidden}, "
            f"input_scaling={input_scaling}, input_scaling_hidden={input_scaling_hidden}, "
            f"bias_scaling={bias_scaling}, bias_scaling_hidden={bias_scaling_hidden}, "
            f"leaky={leaky}, leaky_hidden={leaky_hidden}, "
            f"use_parallel={use_parallel}, last_layer={last_layer}, "
            f"sequences={sequences}, mean={mean}"
        )

        # =====================================================================
        # TRAINING
        # =====================================================================

        reinit_train_t0 = time.time()

        log("[Step 1/3] Estrazione stati training...")
        t0 = time.time()
        X_train_states, ct_train = compute_reservoir_states(model, train_loader, device)
        train_extract_time = time.time() - t0

        log(
            f"  X_train_states: {list(X_train_states.shape)} "
            f"in {train_extract_time:.2f}s "
            f"(reservoir: {ct_train:.2f}s)"
        )

        log("[Step 2/3] Estrazione stati validation...")
        t0 = time.time()
        X_val_states, ct_val = compute_reservoir_states(model, val_loader, device)
        val_extract_time = time.time() - t0

        log(
            f"  X_val_states:   {list(X_val_states.shape)} "
            f"in {val_extract_time:.2f}s "
            f"(reservoir: {ct_val:.2f}s)"
        )

        log("[Readout] Fit con selezione lambda tramite model.fit(...)")

        def make_batches():
            for i in range(0, len(X_train_states), fit_batch_size):
                yield (
                    X_train_states[i:i + fit_batch_size].to(device),
                    y_train_oh[i:i + fit_batch_size].to(device),
                )

        best_err, readout_fit_time, readout_fit_time_ms = model.fit(
            train=X_train_states,
            labels=y_train_oh,
            num_targets=NUM_CLASSES,
            validation_data=(X_val_states.to(device), y_val_oh.to(device)),
            batches=make_batches(),
            verbose=True,
            device=device,
        )

        log(
            f"  Readout fit in {readout_fit_time:.2f}s "
            f"best_val_err={best_err:.4f} "
            f"val_f1={1.0 - best_err:.4f}"
        )

        reinit_training_time = time.time() - reinit_train_t0

        log(
            f"  [Training time reinit {reinit}] "
            f"total={reinit_training_time:.2f}s "
            f"(train_states={train_extract_time:.2f}s, "
            f"val_states={val_extract_time:.2f}s, "
            f"readout_fit={readout_fit_time:.2f}s)"
        )

        log("[Step 3/3] Estrazione stati test...")
        t0 = time.time()
        X_test_states, ct_test = compute_reservoir_states(model, test_loader, device)
        test_extract_time = time.time() - t0

        log(
            f"  X_test_states:  {list(X_test_states.shape)} "
            f"in {test_extract_time:.2f}s "
            f"(reservoir: {ct_test:.2f}s)"
        )

        # =====================================================================
        # CHECKPOINT
        # =====================================================================

        ckpt_path = output_dir / f"esn_reinit{reinit}.pt"

        torch.save(
            {
                "model_state": model.state_dict(),
                "readout_state": model.readout.state_dict(),
                "val_err": float(best_err),
                "val_f1": float(1.0 - best_err),
                "training_time_s": float(reinit_training_time),
                "training_breakdown_s": {
                    "train_state_extraction_s": float(train_extract_time),
                    "val_state_extraction_s": float(val_extract_time),
                    "readout_fit_s": float(readout_fit_time),
                    "test_state_extraction_s": float(test_extract_time),
                },
                "config": {
                    "data_dir": data_dir,
                    "input_size": int(input_size),
                    "num_classes": int(NUM_CLASSES),
                    "esn_units": esn_units,
                    "esn_layers": esn_layers,
                    "spectral_radius": spectral_radius,
                    "spectral_radius_hidden": spectral_radius_hidden,
                    "input_scaling": input_scaling,
                    "input_scaling_hidden": input_scaling_hidden,
                    "bias_scaling": bias_scaling,
                    "bias_scaling_hidden": bias_scaling_hidden,
                    "leaky": leaky,
                    "leaky_hidden": leaky_hidden,
                    "batch_size": batch_size,
                    "fit_batch_size": fit_batch_size,
                    "device": str(device),
                    "use_parallel": use_parallel,
                    "last_layer": last_layer,
                    "sequences": sequences,
                    "mean": mean,
                },
            },
            ckpt_path,
        )

        if best_err < best_reinit_err:
            best_reinit_err = best_err
            best_reinit_ckpt = ckpt_path
            log("  *** Nuovo miglior reinit! ***")

        # =====================================================================
        # EVALUATION
        # =====================================================================

        log("[Eval] Sequence-level sul test set...")
        report = evaluate_sequence_level(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES,
        )

        test_f1 = report["macro avg"]["f1-score"]

        log(format_report(report, CLASS_NAMES))
        log(f"[Reinit {reinit}] Test macro F1: {test_f1:.4f}")

        # =====================================================================
        # TIMING
        # =====================================================================

        log("[Timing] Inference latency ESN...")
        lat = measure_esn_inference(model, test_loader, device)

        log(
            f"[Reinit {reinit}] Batch latency      : "
            f"{lat['latency_batch_mean_ms']:.3f} ± "
            f"{lat['latency_batch_std_ms']:.3f} ms"
        )

        log(
            f"[Reinit {reinit}] Per-sample latency : "
            f"{lat['latency_sample_mean_ms']:.3f} ± "
            f"{lat['latency_sample_std_ms']:.3f} ms"
        )

        log(f"[Reinit {reinit}] Reservoir time     : {lat['reservoir_ms']:.3f} ms/sample")
        log(f"[Reinit {reinit}] Readout time       : {lat['readout_ms']:.3f} ms/sample")
        log(f"[Reinit {reinit}] Throughput         : {lat['throughput_sps']:.2f} samples/s")
        log(f"[Reinit {reinit}] Inference elapsed  : {lat['elapsed_s']:.2f}s")

        report_json = {
            "val_err": float(best_err),
            "val_f1": float(1.0 - best_err),
            "test_f1": float(test_f1),
            "classification_report": report,
            "timing": lat,
            "training_time_s": float(reinit_training_time),
            "training_breakdown_s": {
                "train_state_extraction_s": float(train_extract_time),
                "val_state_extraction_s": float(val_extract_time),
                "readout_fit_s": float(readout_fit_time),
                "test_state_extraction_s": float(test_extract_time),
            },
        }

        with open(output_dir / f"report_reinit{reinit}.json", "w") as f:
            json.dump(report_json, f, indent=2)

        all_reinit_results.append(
            {
                "reinit": int(reinit),
                "seed": int(seed),
                "ckpt": str(ckpt_path),
                "val_err": float(best_err),
                "val_f1": float(1.0 - best_err),
                "test_f1": float(test_f1),
                "training_time_s": float(reinit_training_time),
                "training_breakdown_s": {
                    "train_state_extraction_s": float(train_extract_time),
                    "val_state_extraction_s": float(val_extract_time),
                    "readout_fit_s": float(readout_fit_time),
                    "test_state_extraction_s": float(test_extract_time),
                },
                "timing": {
                    "latency_batch_mean_ms": float(lat["latency_batch_mean_ms"]),
                    "latency_batch_std_ms": float(lat["latency_batch_std_ms"]),
                    "latency_sample_mean_ms": float(lat["latency_sample_mean_ms"]),
                    "latency_sample_std_ms": float(lat["latency_sample_std_ms"]),
                    "reservoir_ms": float(lat["reservoir_ms"]),
                    "readout_ms": float(lat["readout_ms"]),
                    "throughput_sps": float(lat["throughput_sps"]),
                    "elapsed_s": float(lat["elapsed_s"]),
                    "total_samples": int(lat["total_samples"]),
                },
            }
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary_stats = summarize_reinitializations(all_reinit_results)

    log(f"\n[Summary over {num_reinit} reinitializations]")
    log(f"  Val error              : {summary_stats['val_err']['mean']:.4f} ± {summary_stats['val_err']['std']:.4f}")
    log(f"  Val F1                 : {summary_stats['val_f1']['mean']:.4f} ± {summary_stats['val_f1']['std']:.4f}")
    log(f"  Test macro F1          : {summary_stats['test_f1']['mean']:.4f} ± {summary_stats['test_f1']['std']:.4f}")
    log(f"  Training time          : {summary_stats['training_time_s']['mean']:.2f} ± {summary_stats['training_time_s']['std']:.2f} s")
    log(f"  Train state extraction : {summary_stats['train_state_extraction_s']['mean']:.2f} ± {summary_stats['train_state_extraction_s']['std']:.2f} s")
    log(f"  Val state extraction   : {summary_stats['val_state_extraction_s']['mean']:.2f} ± {summary_stats['val_state_extraction_s']['std']:.2f} s")
    log(f"  Readout fit            : {summary_stats['readout_fit_s']['mean']:.2f} ± {summary_stats['readout_fit_s']['std']:.2f} s")

    if "elapsed_s" in summary_stats:
        log(f"  Inference elapsed      : {summary_stats['elapsed_s']['mean']:.2f} ± {summary_stats['elapsed_s']['std']:.2f} s")

    if "latency_batch_mean_ms" in summary_stats:
        log(f"  Batch latency          : {summary_stats['latency_batch_mean_ms']['mean']:.3f} ± {summary_stats['latency_batch_mean_ms']['std']:.3f} ms")

    if "latency_sample_mean_ms" in summary_stats:
        log(f"  Per-sample latency     : {summary_stats['latency_sample_mean_ms']['mean']:.3f} ± {summary_stats['latency_sample_mean_ms']['std']:.3f} ms")

    if "reservoir_ms" in summary_stats:
        log(f"  Reservoir time/sample  : {summary_stats['reservoir_ms']['mean']:.3f} ± {summary_stats['reservoir_ms']['std']:.3f} ms")

    if "readout_ms" in summary_stats:
        log(f"  Readout time/sample    : {summary_stats['readout_ms']['mean']:.3f} ± {summary_stats['readout_ms']['std']:.3f} ms")

    if "throughput_sps" in summary_stats:
        log(f"  Throughput             : {summary_stats['throughput_sps']['mean']:.2f} ± {summary_stats['throughput_sps']['std']:.2f} samples/s")

    summary = {
        "best_reinit_ckpt": str(best_reinit_ckpt) if best_reinit_ckpt is not None else None,
        "best_val_err": float(best_reinit_err),
        "best_val_f1": float(1.0 - best_reinit_err),
        "num_reinit": int(num_reinit),
        "aggregated_stats": summary_stats,
        "reinitializations": all_reinit_results,
        "config": {
            "data_dir": data_dir,
            "input_size": int(input_size),
            "num_classes": int(NUM_CLASSES),
            "class_names": CLASS_NAMES,
            "esn_units": esn_units,
            "esn_layers": esn_layers,
            "spectral_radius": spectral_radius,
            "spectral_radius_hidden": spectral_radius_hidden,
            "input_scaling": input_scaling,
            "input_scaling_hidden": input_scaling_hidden,
            "bias_scaling": bias_scaling,
            "bias_scaling_hidden": bias_scaling_hidden,
            "leaky": leaky,
            "leaky_hidden": leaky_hidden,
            "readout_reg_min_exp": readout_reg_min_exp,
            "readout_reg_max_exp": readout_reg_max_exp,
            "readout_reg_steps": readout_reg_steps,
            "batch_size": batch_size,
            "fit_batch_size": fit_batch_size,
            "num_workers": num_workers,
            "device": str(device),
            "use_parallel": use_parallel,
            "use_compile": use_compile,
            "last_layer": last_layer,
            "sequences": sequences,
            "mean": mean,
            "classes_to_remove": classes_to_remove,
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n[Done] Miglior reinit: {best_reinit_ckpt} val_err={best_reinit_err:.4f}")

    print(
        "\nVal F1 mean ± std:",
        f"{summary['aggregated_stats']['val_f1']['mean']:.4f} ± "
        f"{summary['aggregated_stats']['val_f1']['std']:.4f}"
    )

    print(
        "Test F1 mean ± std:",
        f"{summary['aggregated_stats']['test_f1']['mean']:.4f} ± "
        f"{summary['aggregated_stats']['test_f1']['std']:.4f}"
    )

    print(
        "Training time mean ± std:",
        f"{summary['aggregated_stats']['training_time_s']['mean']:.2f} ± "
        f"{summary['aggregated_stats']['training_time_s']['std']:.2f} s"
    )

    print(
        "Per-sample latency mean ± std:",
        f"{summary['aggregated_stats']['latency_sample_mean_ms']['mean']:.3f} ± "
        f"{summary['aggregated_stats']['latency_sample_mean_ms']['std']:.3f} ms"
    )

    print(
        "Throughput mean ± std:",
        f"{summary['aggregated_stats']['throughput_sps']['mean']:.2f} ± "
        f"{summary['aggregated_stats']['throughput_sps']['std']:.2f} samples/s"
    )

    return best_reinit_ckpt, summary


# =============================================================================
# CLI
# =============================================================================

def parse_classes_to_remove(value):
    if value is None or value.strip() == "":
        return []

    return [int(v.strip()) for v in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Deep Bidirectional ESN IDS experiment over multiple random reinitializations."
    )

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--esn-units", type=int, default=256)
    parser.add_argument("--esn-layers", type=int, default=1)

    parser.add_argument("--spectral-radius", type=float, default=0.3)
    parser.add_argument("--spectral-radius-hidden", type=float, default=0.7)

    parser.add_argument("--input-scaling", type=float, default=1.0)
    parser.add_argument("--input-scaling-hidden", type=float, default=0.5)

    parser.add_argument("--bias-scaling", type=float, default=0.0)
    parser.add_argument("--bias-scaling-hidden", type=float, default=0.5)

    parser.add_argument("--leaky", type=float, default=0.9)
    parser.add_argument("--leaky-hidden", type=float, default=0.9)

    parser.add_argument("--readout-reg-min-exp", type=int, default=-8)
    parser.add_argument("--readout-reg-max-exp", type=int, default=5)
    parser.add_argument("--readout-reg-steps", type=int, default=15)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fit-batch-size", type=int, default=50_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-reinit", type=int, default=10)

    parser.add_argument(
        "--classes-to-remove",
        type=str,
        default="13,8,9,14",
        help="Comma-separated original labels to remove. Use empty string '' to remove none.",
    )

    parser.add_argument(
        "--use-parallel",
        action="store_true",
        help="Enable parallel ESN implementation if supported by DeepESN.",
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel ESN implementation.",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Try to enable torch.compile(model).",
    )

    parser.add_argument(
        "--last-layer",
        action="store_true",
        help="Use only last ESN layer if supported by DeepESN.",
    )

    parser.add_argument(
        "--sequences",
        action="store_true",
        help="Use sequence output if supported by DeepESN. Default is False, i.e. last state.",
    )

    parser.add_argument(
        "--mean",
        action="store_true",
        help="Use mean temporal aggregation if supported by DeepESN.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_parallel:
        use_parallel = False
    else:
        use_parallel = args.use_parallel

    classes_to_remove = parse_classes_to_remove(args.classes_to_remove)

    run_esn_toolkit(
        data_dir=args.data_dir,
        output=args.output,
        device_str=args.device,
        esn_units=args.esn_units,
        esn_layers=args.esn_layers,
        spectral_radius=args.spectral_radius,
        spectral_radius_hidden=args.spectral_radius_hidden,
        input_scaling=args.input_scaling,
        input_scaling_hidden=args.input_scaling_hidden,
        bias_scaling=args.bias_scaling,
        bias_scaling_hidden=args.bias_scaling_hidden,
        leaky=args.leaky,
        leaky_hidden=args.leaky_hidden,
        readout_reg_min_exp=args.readout_reg_min_exp,
        readout_reg_max_exp=args.readout_reg_max_exp,
        readout_reg_steps=args.readout_reg_steps,
        batch_size=args.batch_size,
        fit_batch_size=args.fit_batch_size,
        num_workers=args.num_workers,
        num_reinit=args.num_reinit,
        use_parallel=use_parallel,
        use_compile=args.compile,
        last_layer=args.last_layer,
        sequences=args.sequences,
        mean=args.mean,
        classes_to_remove=classes_to_remove,
    )


if __name__ == "__main__":
    main()
