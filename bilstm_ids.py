import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, classification_report


BASE_SEED = 1337


# =============================================================================
# LOAD DATA
# =============================================================================

def load(split, data_dir):
    return np.load(f"{data_dir}/{split}.npy")


def prepare_data(data_dir):
    X_train = torch.from_numpy(load("X_train_seq", data_dir)).float()
    X_val   = torch.from_numpy(load("X_val_seq", data_dir)).float()
    X_test  = torch.from_numpy(load("X_test_seq", data_dir)).float()

    Y_train = torch.from_numpy(load("y_train_seq", data_dir)).long()
    Y_val   = torch.from_numpy(load("y_val_seq", data_dir)).long()
    Y_test  = torch.from_numpy(load("y_test_seq", data_dir)).long()

    classi_da_rimuovere = torch.tensor([13, 8, 9, 14])

    tr_mask   = ~torch.isin(Y_train, classi_da_rimuovere)
    vl_mask   = ~torch.isin(Y_val, classi_da_rimuovere)
    test_mask = ~torch.isin(Y_test, classi_da_rimuovere)

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

    print("Unique remapped train labels:", torch.unique(Y_train))
    print("Train:", X_train.shape, Y_train.shape)
    print("Val:  ", X_val.shape, Y_val.shape)
    print("Test: ", X_test.shape, Y_test.shape)

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
# MODEL
# =============================================================================

class BiLSTMLastState(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_classes: int = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if num_classes is None:
            raise ValueError("num_classes must be specified")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)
        return logits


# =============================================================================
# TRAIN / EVAL
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_true = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_true.append(yb.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_true  = torch.cat(all_true).numpy()

    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    return f1, all_true, all_preds


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.stop = False

    def step(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return False


# =============================================================================
# INFERENCE TIMING
# =============================================================================

@torch.no_grad()
def measure_bilstm_inference(model, loader, device, n_warmup: int = 5):
    model.eval()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    print("[Latency BiLSTM] Warmup...")
    for i, (xb, yb) in enumerate(loader):
        if i >= n_warmup:
            break
        xb = xb.to(device, non_blocking=True)
        _ = model(xb)
    sync()

    total_samples = 0
    batch_latency_ms = []
    latency_per_sample = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        B = xb.shape[0]

        sync()
        t0 = time.perf_counter()
        _ = model(xb)
        sync()
        t_ms = (time.perf_counter() - t0) * 1000.0

        batch_latency_ms.append(t_ms)
        latency_per_sample.append(t_ms / B)
        total_samples += B

    batch_latency_ms = np.array(batch_latency_ms, dtype=np.float64)
    latency_per_sample = np.array(latency_per_sample, dtype=np.float64)

    sync()
    t0 = time.perf_counter()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        _ = model(xb)
    sync()
    elapsed_s = time.perf_counter() - t0

    throughput_sps = total_samples / elapsed_s if elapsed_s > 0 else 0.0

    return {
        "latency_batch_mean_ms": float(np.mean(batch_latency_ms)),
        "latency_batch_std_ms": float(np.std(batch_latency_ms)),
        "latency_sample_mean_ms": float(np.mean(latency_per_sample)),
        "latency_sample_std_ms": float(np.std(latency_per_sample)),
        "throughput_sps": float(throughput_sps),
        "elapsed_s": float(elapsed_s),
        "total_samples": int(total_samples),
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


def summarize_runs(all_run_results):
    summary_stats = {}

    summary_stats["best_val_f1"] = aggregate_mean_std([r["best_val_f1"] for r in all_run_results])
    summary_stats["test_f1"] = aggregate_mean_std([r["test_f1"] for r in all_run_results])
    summary_stats["training_time_s"] = aggregate_mean_std([r["training_time_s"] for r in all_run_results])
    summary_stats["epochs_trained"] = aggregate_mean_std([r["epochs_trained"] for r in all_run_results])

    timing_keys = [
        "latency_batch_mean_ms",
        "latency_batch_std_ms",
        "latency_sample_mean_ms",
        "latency_sample_std_ms",
        "throughput_sps",
        "elapsed_s",
        "total_samples",
    ]

    for key in timing_keys:
        summary_stats[key] = aggregate_mean_std([r["timing"][key] for r in all_run_results])

    return summary_stats


# =============================================================================
# MAIN
# =============================================================================

def run_bilstm_toolkit(
    data_dir: str,
    output: str,
    device_str: str,
    hidden_size: int = 256,
    num_layers: int = 1,
    dropout: float = 0.3,
    batch_size: int = 256,
    num_workers: int = 0,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
    num_reinit: int = 10,
):
    X_train, X_val, X_test, Y_train, Y_val, Y_test, NUM_CLASSES, CLASS_NAMES = prepare_data(data_dir)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "bilstm_train.log"

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    device = torch.device(device_str)
    input_size = X_train.shape[-1]

    log(f"[Device] {device}")
    log(f"[Data] train={list(X_train.shape)} val={list(X_val.shape)} test={list(X_test.shape)}")
    log(f"[Data] num_classes={NUM_CLASSES} input_size={input_size}")

    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val, Y_val)
    test_ds  = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    all_run_results = []
    best_global_val_f1 = -1.0
    best_global_ckpt = None

    for run in range(1, num_reinit + 1):
        seed = BASE_SEED + run
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        log(f"\n{'=' * 60}")
        log(f"[Run {run}/{num_reinit}] seed={seed}")

        model = BiLSTMLastState(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=NUM_CLASSES,
            dropout=dropout,
        ).to(device)

        torch.compile(model)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"[Model] trainable params: {n_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        early_stop = EarlyStopping(patience=patience)

        best_val_f1 = -1.0
        best_state = None
        epochs_trained = 0

        train_start = time.time()

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_f1, _, _ = evaluate(model, val_loader, device)
            scheduler.step(val_f1)

            log(f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_f1={val_f1:.4f}")

            improved = early_stop.step(val_f1)
            if improved:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_trained = epoch

            if early_stop.stop:
                log("Early stopping triggered.")
                break

        training_time_s = time.time() - train_start

        if best_state is not None:
            model.load_state_dict(best_state)

        test_f1, y_true, y_pred = evaluate(model, test_loader, device)
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        log(f"[Run {run}] Best val F1 : {best_val_f1:.4f}")
        log(f"[Run {run}] Test macro F1: {test_f1:.4f}")
        log(f"[Run {run}] Training time: {training_time_s:.2f}s")
        log(classification_report(y_true, y_pred, zero_division=0))

        timing = measure_bilstm_inference(model, test_loader, device)

        log(f"[Run {run}] Batch latency      : {timing['latency_batch_mean_ms']:.3f} ± {timing['latency_batch_std_ms']:.3f} ms")
        log(f"[Run {run}] Per-sample latency : {timing['latency_sample_mean_ms']:.3f} ± {timing['latency_sample_std_ms']:.3f} ms")
        log(f"[Run {run}] Throughput         : {timing['throughput_sps']:.2f} samples/s")
        log(f"[Run {run}] Inference elapsed  : {timing['elapsed_s']:.2f}s")

        ckpt_path = output_dir / f"bilstm_run{run}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "best_val_f1": best_val_f1,
            "test_f1": test_f1,
            "training_time_s": training_time_s,
            "epochs_trained": epochs_trained,
            "timing": timing,
        }, ckpt_path)

        if best_val_f1 > best_global_val_f1:
            best_global_val_f1 = best_val_f1
            best_global_ckpt = ckpt_path

        with open(output_dir / f"report_run{run}.json", "w") as f:
            json.dump({
                "best_val_f1": best_val_f1,
                "test_f1": test_f1,
                "training_time_s": training_time_s,
                "epochs_trained": epochs_trained,
                "timing": timing,
                "classification_report": report,
            }, f, indent=2)

        all_run_results.append({
            "run": run,
            "seed": seed,
            "ckpt": str(ckpt_path),
            "best_val_f1": float(best_val_f1),
            "test_f1": float(test_f1),
            "training_time_s": float(training_time_s),
            "epochs_trained": int(epochs_trained),
            "timing": timing,
        })

    summary_stats = summarize_runs(all_run_results)

    log(f"\n[Summary over {num_reinit} runs]")
    log(f"  Best val F1           : {summary_stats['best_val_f1']['mean']:.4f} ± {summary_stats['best_val_f1']['std']:.4f}")
    log(f"  Test macro F1         : {summary_stats['test_f1']['mean']:.4f} ± {summary_stats['test_f1']['std']:.4f}")
    log(f"  Training time         : {summary_stats['training_time_s']['mean']:.2f} ± {summary_stats['training_time_s']['std']:.2f} s")
    log(f"  Epochs trained        : {summary_stats['epochs_trained']['mean']:.2f} ± {summary_stats['epochs_trained']['std']:.2f}")
    log(f"  Batch latency         : {summary_stats['latency_batch_mean_ms']['mean']:.3f} ± {summary_stats['latency_batch_mean_ms']['std']:.3f} ms")
    log(f"  Per-sample latency    : {summary_stats['latency_sample_mean_ms']['mean']:.3f} ± {summary_stats['latency_sample_mean_ms']['std']:.3f} ms")
    log(f"  Throughput            : {summary_stats['throughput_sps']['mean']:.2f} ± {summary_stats['throughput_sps']['std']:.2f} samples/s")
    log(f"  Inference elapsed     : {summary_stats['elapsed_s']['mean']:.2f} ± {summary_stats['elapsed_s']['std']:.2f} s")

    summary = {
        "best_global_ckpt": str(best_global_ckpt) if best_global_ckpt is not None else None,
        "best_global_val_f1": float(best_global_val_f1),
        "num_runs": int(num_reinit),
        "aggregated_stats": summary_stats,
        "runs": all_run_results,
        "config": {
            "data_dir": data_dir,
            "input_size": input_size,
            "num_classes": NUM_CLASSES,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "max_epochs": max_epochs,
            "patience": patience,
            "device": str(device),
        }
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        "\nBest val F1 mean ± std:",
        f"{summary['aggregated_stats']['best_val_f1']['mean']:.4f} ± "
        f"{summary['aggregated_stats']['best_val_f1']['std']:.4f}"
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

    return best_global_ckpt, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run BiLSTM IDS experiment over multiple random initializations.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-reinit", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    run_bilstm_toolkit(
        data_dir=args.data_dir,
        output=args.output,
        device_str=args.device,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        num_reinit=args.num_reinit,
    )


if __name__ == "__main__":
    main()