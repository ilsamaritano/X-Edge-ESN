#!/usr/bin/env python3
"""
X-Edge-ESN GUI
Interactive interface for BiLSTM/BiGRU IDS training and inference.
"""

import sys
import os
import json
import re
import queue
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

# ─── project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "preprocessed_sequences"
BILSTM_OUT   = PROJECT_ROOT / "ids_bilstm_output"
BIGRU_OUT    = PROJECT_ROOT / "ids_bigru_output"

# ─── colour palette (dark-ish) ────────────────────────────────────────────────
BG       = "#1e1e2e"
BG2      = "#2a2a3e"
BG3      = "#313148"
FG       = "#cdd6f4"
ACCENT   = "#89b4fa"
GREEN    = "#a6e3a1"
YELLOW   = "#f9e2af"
RED      = "#f38ba8"
MUTED    = "#6c7086"
ENTRY_BG = "#181825"

# ─── parse helpers ───────────────────────────────────────────────────────────
_RE_EPOCH    = re.compile(r"Epoch\s+(\d+)\s*\|\s*loss=([\d.]+)\s*\|\s*val_f1=([\d.]+)")
_RE_RUN_DONE = re.compile(r"\[Run (\d+)\].*?Best val F1\s*:\s*([\d.]+)")
_RE_TEST_F1  = re.compile(r"\[Run (\d+)\].*?Test macro F1:\s*([\d.]+)")
_RE_RUN_START= re.compile(r"\[Run (\d+)/(\d+)\]\s+seed=(\d+)")


# ══════════════════════════════════════════════════════════════════════════════
# Styled helpers
# ══════════════════════════════════════════════════════════════════════════════

def _apply_style(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=FG, fieldbackground=ENTRY_BG,
                    troughcolor=BG2, selectbackground=ACCENT, selectforeground=BG,
                    font=("Segoe UI", 10))
    style.configure("TNotebook", background=BG, tabmargins=[2, 5, 2, 0])
    style.configure("TNotebook.Tab", background=BG3, foreground=MUTED, padding=[12, 5])
    style.map("TNotebook.Tab", background=[("selected", BG2)], foreground=[("selected", FG)])
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=FG)
    style.configure("TLabelframe", background=BG, foreground=ACCENT)
    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT)
    style.configure("TButton", background=BG3, foreground=FG, relief="flat", padding=[8, 4])
    style.map("TButton", background=[("active", ACCENT)], foreground=[("active", BG)])
    style.configure("Accent.TButton", background=ACCENT, foreground=BG, font=("Segoe UI", 10, "bold"))
    style.map("Accent.TButton", background=[("active", GREEN)])
    style.configure("Danger.TButton", background=RED, foreground=BG)
    style.map("Danger.TButton", background=[("active", "#ff6b6b")])
    style.configure("TEntry", fieldbackground=ENTRY_BG, foreground=FG)
    style.configure("TSpinbox", fieldbackground=ENTRY_BG, foreground=FG,
                    arrowcolor=ACCENT, background=BG)
    style.configure("TCombobox", fieldbackground=ENTRY_BG, foreground=FG,
                    arrowcolor=ACCENT)
    style.map("TCombobox", fieldbackground=[("readonly", ENTRY_BG)])
    style.configure("TScrollbar", background=BG2, troughcolor=BG, arrowcolor=MUTED)
    style.configure("TProgressbar", troughcolor=BG2, background=ACCENT)
    style.configure("TTreeview", background=BG2, foreground=FG, fieldbackground=BG2,
                    rowheight=24)
    style.configure("TTreeview.Heading", background=BG3, foreground=ACCENT)
    style.map("TTreeview", background=[("selected", ACCENT)], foreground=[("selected", BG)])
    style.configure("Status.TLabel", background=BG3, foreground=MUTED, padding=[6, 3])


def _label(parent, text, bold=False, fg=None, size=10):
    font = ("Segoe UI", size, "bold" if bold else "normal")
    return ttk.Label(parent, text=text, foreground=fg or FG, font=font)


def _entry(parent, textvariable=None, width=20):
    e = ttk.Entry(parent, textvariable=textvariable, width=width)
    return e


def _spinbox(parent, from_, to, textvariable=None, width=8, increment=1):
    return ttk.Spinbox(parent, from_=from_, to=to, textvariable=textvariable,
                       width=width, increment=increment)


def _btn(parent, text, command, style="TButton", width=None):
    kw = {"style": style, "command": command}
    if width:
        kw["width"] = width
    return ttk.Button(parent, text=text, **kw)


def _scroll_text(parent, **kw):
    frame = ttk.Frame(parent)
    txt = tk.Text(frame, bg=ENTRY_BG, fg=FG, insertbackground=FG,
                  selectbackground=ACCENT, relief="flat", font=("Courier New", 9), **kw)
    sb = ttk.Scrollbar(frame, command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    frame.txt = txt
    return frame


def _path_row(parent, label, var, filetypes=None, directory=False):
    """One-liner path picker: label + entry + browse button."""
    row = ttk.Frame(parent)
    _label(row, label).pack(side=tk.LEFT)
    e = _entry(row, textvariable=var, width=35)
    e.pack(side=tk.LEFT, padx=(4, 2))
    def browse():
        if directory:
            p = filedialog.askdirectory(initialdir=var.get() or PROJECT_ROOT)
        else:
            p = filedialog.askopenfilename(
                initialdir=str(PROJECT_ROOT),
                filetypes=filetypes or [("All", "*.*")]
            )
        if p:
            var.set(p)
    _btn(row, "Browse…", browse).pack(side=tk.LEFT)
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Embedded matplotlib figure
# ══════════════════════════════════════════════════════════════════════════════

class LiveChart(ttk.Frame):
    """Two subplots: F1-score per epoch and loss per epoch."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.fig = Figure(figsize=(7, 3.6), facecolor=BG2)
        self.fig.subplots_adjust(hspace=0.45, left=0.09, right=0.97,
                                 top=0.90, bottom=0.12)
        self.ax_f1   = self.fig.add_subplot(1, 2, 1)
        self.ax_loss = self.fig.add_subplot(1, 2, 2)
        self._style_ax(self.ax_f1,   "Val F1 per Epoch",    "Val F1")
        self._style_ax(self.ax_loss, "Train Loss per Epoch", "Loss")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._run_f1_data   = {}  # run -> [f1, ...]
        self._run_loss_data = {}  # run -> [loss, ...]
        self._current_run   = None

    def _style_ax(self, ax, title, ylabel):
        ax.set_facecolor(BG)
        ax.set_title(title, color=FG, fontsize=9, pad=4)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
        ax.set_xlabel("Epoch", color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(BG3)

    def reset(self):
        self._run_f1_data   = {}
        self._run_loss_data = {}
        self._current_run   = None
        for ax in (self.ax_f1, self.ax_loss):
            ax.cla()
        self._style_ax(self.ax_f1,   "Val F1 per Epoch",    "Val F1")
        self._style_ax(self.ax_loss, "Train Loss per Epoch", "Loss")
        self.canvas.draw_idle()

    def set_run(self, run_id):
        self._current_run = run_id
        if run_id not in self._run_f1_data:
            self._run_f1_data[run_id]   = []
            self._run_loss_data[run_id] = []

    def add_epoch(self, run_id, loss, val_f1):
        self._run_f1_data.setdefault(run_id, []).append(val_f1)
        self._run_loss_data.setdefault(run_id, []).append(loss)
        self._redraw()

    def _redraw(self):
        cmap = matplotlib.colormaps.get_cmap("tab10")
        for ax in (self.ax_f1, self.ax_loss):
            ax.cla()
        self._style_ax(self.ax_f1,   "Val F1 per Epoch",    "Val F1")
        self._style_ax(self.ax_loss, "Train Loss per Epoch", "Loss")

        for i, (run_id, f1_vals) in enumerate(self._run_f1_data.items()):
            color = cmap(i % 10)
            lw = 2 if run_id == self._current_run else 0.8
            alpha = 1.0 if run_id == self._current_run else 0.45
            xs = list(range(1, len(f1_vals) + 1))
            self.ax_f1.plot(xs, f1_vals, color=color, lw=lw, alpha=alpha,
                            label=f"R{run_id}")
            loss_vals = self._run_loss_data.get(run_id, [])
            if loss_vals:
                self.ax_loss.plot(xs[:len(loss_vals)], loss_vals, color=color,
                                  lw=lw, alpha=alpha)
        if len(self._run_f1_data) <= 5:
            self.ax_f1.legend(fontsize=6, facecolor=BG2, edgecolor=BG3,
                              labelcolor=FG, loc="lower right")
        self.canvas.draw_idle()


class SummaryChart(ttk.Frame):
    """Bar chart of per-run val/test F1 with error bands."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.fig = Figure(figsize=(7, 3.2), facecolor=BG2)
        self.ax  = self.fig.add_subplot(1, 1, 1)
        self._style()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _style(self):
        self.ax.set_facecolor(BG)
        self.ax.set_title("F1 Scores per Run", color=FG, fontsize=9)
        self.ax.tick_params(colors=MUTED, labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(BG3)

    def plot(self, runs_data):
        """runs_data: list of dicts with 'run', 'best_val_f1', 'test_f1'."""
        self.ax.cla()
        self._style()
        if not runs_data:
            self.canvas.draw_idle()
            return
        runs   = [d["run"] for d in runs_data]
        val_f1 = [d["best_val_f1"] for d in runs_data]
        tst_f1 = [d["test_f1"] for d in runs_data]
        x = np.arange(len(runs))
        w = 0.35
        self.ax.bar(x - w/2, val_f1, w, color=ACCENT, alpha=0.85, label="Val F1")
        self.ax.bar(x + w/2, tst_f1, w, color=GREEN,  alpha=0.85, label="Test F1")
        self.ax.set_xticks(x)
        self.ax.set_xticklabels([f"R{r}" for r in runs], fontsize=7, color=MUTED)
        self.ax.set_ylim(0, 1.05)
        self.ax.legend(fontsize=7, facecolor=BG2, edgecolor=BG3, labelcolor=FG)
        self.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════════════
# Training worker (subprocess)
# ══════════════════════════════════════════════════════════════════════════════

class TrainingWorker:
    """Runs the training script in a subprocess; emits parsed events via queue."""

    def __init__(self, cmd, out_q):
        self._cmd   = cmd
        self._q     = out_q
        self._proc  = None
        self._stop  = threading.Event()

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _run(self):
        self._q.put(("log", f"$ {' '.join(self._cmd)}\n"))
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
            )
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                line = line.rstrip()
                self._q.put(("log", line))
                self._parse(line)
            self._proc.wait()
            rc = self._proc.returncode
            if self._stop.is_set():
                self._q.put(("done", "stopped"))
            elif rc == 0:
                self._q.put(("done", "ok"))
            else:
                self._q.put(("done", f"error (rc={rc})"))
        except Exception as exc:
            self._q.put(("log", f"[GUI error] {exc}"))
            self._q.put(("done", "error"))

    def _parse(self, line):
        m = _RE_RUN_START.search(line)
        if m:
            self._q.put(("run_start", int(m.group(1)), int(m.group(2))))
            return
        m = _RE_EPOCH.search(line)
        if m:
            self._q.put(("epoch", int(m.group(1)), float(m.group(2)), float(m.group(3))))
            return
        m = _RE_RUN_DONE.search(line)
        if m:
            self._q.put(("run_val_f1", int(m.group(1)), float(m.group(2))))
        m = _RE_TEST_F1.search(line)
        if m:
            self._q.put(("run_test_f1", int(m.group(1)), float(m.group(2))))


# ══════════════════════════════════════════════════════════════════════════════
# Inference worker (subprocess)
# ══════════════════════════════════════════════════════════════════════════════

class InferenceWorker:
    def __init__(self, cmd, out_q):
        self._cmd = cmd
        self._q   = out_q
        self._proc = None

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        self._q.put(("log", f"$ {' '.join(self._cmd)}\n"))
        try:
            self._proc = subprocess.Popen(
                self._cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(PROJECT_ROOT)
            )
            for line in self._proc.stdout:
                self._q.put(("log", line.rstrip()))
            self._proc.wait()
            rc = self._proc.returncode
            self._q.put(("done", "ok" if rc == 0 else f"error (rc={rc})"))
        except Exception as exc:
            self._q.put(("log", f"[GUI error] {exc}"))
            self._q.put(("done", "error"))


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING TAB
# ══════════════════════════════════════════════════════════════════════════════

class TrainingTab(ttk.Frame):

    def __init__(self, parent, log_cb, status_cb):
        super().__init__(parent)
        self._log    = log_cb
        self._status = status_cb
        self._q      = queue.Queue()
        self._worker = None
        self._run_results = {}   # run_id -> {val_f1, test_f1}
        self._cur_run = 0
        self._num_runs = 10
        self._cur_epoch = 0
        self._max_epochs = 100

        self._build()

    # ── layout ──────────────────────────────────────────────────────────────

    def _build(self):
        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(pane, width=280)
        pane.add(left, weight=0)
        self._build_config(left)

        right = ttk.Frame(pane)
        pane.add(right, weight=1)
        self._build_right(right)

    def _build_config(self, parent):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0, width=270)
        sb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _resize(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", _resize)
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        # ── model type ──
        grp = ttk.LabelFrame(inner, text="Model", padding=6)
        grp.pack(fill=tk.X, padx=4, pady=(6, 2))
        self._model_var = tk.StringVar(value="BiLSTM")
        for m in ("BiLSTM", "BiGRU"):
            ttk.Radiobutton(grp, text=m, value=m, variable=self._model_var).pack(
                side=tk.LEFT, padx=8)

        # ── architecture ──
        grp = ttk.LabelFrame(inner, text="Architecture", padding=6)
        grp.pack(fill=tk.X, padx=4, pady=2)
        self._hidden  = self._param_row(grp, "Hidden size",  256, 16, 4096)
        self._layers  = self._param_row(grp, "Num layers",   1,   1,  8)
        self._dropout = self._float_row(grp, "Dropout",      0.3, 0.0, 0.9, 0.05)

        # ── training ──
        grp = ttk.LabelFrame(inner, text="Training", padding=6)
        grp.pack(fill=tk.X, padx=4, pady=2)
        self._batch   = self._param_row(grp, "Batch size",   256, 1, 8192, 32)
        self._lr_var  = tk.StringVar(value="1e-3")
        self._lr_row(grp)
        self._epochs  = self._param_row(grp, "Max epochs",   100, 1, 2000)
        self._patience= self._param_row(grp, "Patience",     10,  1, 100)
        self._reinit  = self._param_row(grp, "Num reinit",   10,  1, 50)
        self._workers = self._param_row(grp, "Workers",      0,   0, 16)

        # ── paths ──
        grp = ttk.LabelFrame(inner, text="Paths", padding=6)
        grp.pack(fill=tk.X, padx=4, pady=2)
        self._data_dir = tk.StringVar(value=str(ARTIFACT_DIR))
        self._out_dir  = tk.StringVar(value=str(BILSTM_OUT))
        for lbl, var, d in [
            ("Data dir",   self._data_dir, True),
            ("Output dir", self._out_dir,  True),
        ]:
            r = ttk.Frame(grp)
            r.pack(fill=tk.X, pady=1)
            _label(r, lbl, fg=MUTED).pack(anchor="w")
            row2 = ttk.Frame(r)
            row2.pack(fill=tk.X)
            _entry(row2, textvariable=var, width=22).pack(side=tk.LEFT, fill=tk.X, expand=True)
            _btn(row2, "…", lambda v=var: self._browse_dir(v)).pack(side=tk.LEFT, padx=2)

        def _sync_out(*_):
            m = self._model_var.get()
            if m == "BiLSTM":
                self._out_dir.set(str(BILSTM_OUT))
            else:
                self._out_dir.set(str(BIGRU_OUT))
        self._model_var.trace_add("write", _sync_out)

        # ── device ──
        grp = ttk.LabelFrame(inner, text="Device", padding=6)
        grp.pack(fill=tk.X, padx=4, pady=2)
        self._device_var = tk.StringVar(value="cuda")
        cb = ttk.Combobox(grp, textvariable=self._device_var,
                          values=["cuda", "cuda:0", "cuda:1", "cpu"], width=14)
        cb.pack(anchor="w")

        # ── buttons ──
        btn_frame = ttk.Frame(inner)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)
        self._start_btn = _btn(btn_frame, "▶  Start Training", self._start,
                               style="Accent.TButton", width=20)
        self._start_btn.pack(fill=tk.X, pady=(0, 4))
        self._stop_btn = _btn(btn_frame, "■  Stop", self._stop,
                              style="Danger.TButton")
        self._stop_btn.pack(fill=tk.X)
        self._stop_btn.state(["disabled"])

    def _param_row(self, parent, label, default, lo, hi, inc=1):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        _label(row, label, fg=MUTED).pack(side=tk.LEFT)
        var = tk.IntVar(value=default)
        _spinbox(row, lo, hi, textvariable=var, width=7, increment=inc
                 ).pack(side=tk.RIGHT)
        return var

    def _float_row(self, parent, label, default, lo, hi, inc):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        _label(row, label, fg=MUTED).pack(side=tk.LEFT)
        var = tk.DoubleVar(value=default)
        sb = ttk.Spinbox(parent if False else row, from_=lo, to=hi,
                         textvariable=var, width=7, increment=inc,
                         format="%.2f")
        sb.pack(side=tk.RIGHT)
        return var

    def _lr_row(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        _label(row, "Learning rate", fg=MUTED).pack(side=tk.LEFT)
        _entry(row, textvariable=self._lr_var, width=9).pack(side=tk.RIGHT)

    # ── right panel ─────────────────────────────────────────────────────────

    def _build_right(self, parent):
        # progress
        prog_frame = ttk.LabelFrame(parent, text="Progress", padding=6)
        prog_frame.pack(fill=tk.X, padx=4, pady=(6, 2))

        r1 = ttk.Frame(prog_frame)
        r1.pack(fill=tk.X)
        _label(r1, "Run:", fg=MUTED).pack(side=tk.LEFT)
        self._run_lbl = _label(r1, "–", fg=ACCENT)
        self._run_lbl.pack(side=tk.LEFT, padx=4)
        self._run_bar = ttk.Progressbar(r1, length=300, mode="determinate")
        self._run_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        r2 = ttk.Frame(prog_frame)
        r2.pack(fill=tk.X, pady=(2, 0))
        _label(r2, "Epoch:", fg=MUTED).pack(side=tk.LEFT)
        self._ep_lbl = _label(r2, "–", fg=YELLOW)
        self._ep_lbl.pack(side=tk.LEFT, padx=4)
        self._ep_bar = ttk.Progressbar(r2, length=300, mode="determinate")
        self._ep_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # charts
        chart_frame = ttk.LabelFrame(parent, text="Live Training Curves", padding=4)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self._chart = LiveChart(chart_frame)
        self._chart.pack(fill=tk.BOTH, expand=True)

        # run table
        tbl_frame = ttk.LabelFrame(parent, text="Run Results", padding=4)
        tbl_frame.pack(fill=tk.X, padx=4, pady=(2, 6))
        cols = ("run", "seed", "val_f1", "test_f1", "epochs", "time_s")
        self._tbl = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=5)
        hdrs = {"run":"Run","seed":"Seed","val_f1":"Val F1","test_f1":"Test F1",
                "epochs":"Epochs","time_s":"Time (s)"}
        widths = {"run":40,"seed":60,"val_f1":70,"test_f1":70,"epochs":55,"time_s":70}
        for c in cols:
            self._tbl.heading(c, text=hdrs[c])
            self._tbl.column(c, width=widths[c], anchor="center")
        sb2 = ttk.Scrollbar(tbl_frame, command=self._tbl.yview)
        self._tbl.configure(yscrollcommand=sb2.set)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self._tbl.pack(fill=tk.X)

    # ── actions ─────────────────────────────────────────────────────────────

    def _browse_dir(self, var):
        p = filedialog.askdirectory(initialdir=var.get() or str(PROJECT_ROOT))
        if p:
            var.set(p)

    def _start(self):
        try:
            lr = float(self._lr_var.get())
        except ValueError:
            messagebox.showerror("Invalid LR", "Learning rate must be a float (e.g. 1e-3).")
            return

        model = self._model_var.get()
        script = "bilstm_ids.py" if model == "BiLSTM" else "bigru_ids.py"

        cmd = [
            sys.executable, script,
            "--data-dir",    self._data_dir.get(),
            "--output",      self._out_dir.get(),
            "--device",      self._device_var.get(),
            "--hidden-size", str(self._hidden.get()),
            "--num-layers",  str(self._layers.get()),
            "--dropout",     str(round(self._dropout.get(), 3)),
            "--batch-size",  str(self._batch.get()),
            "--num-workers", str(self._workers.get()),
            "--lr",          str(lr),
            "--max-epochs",  str(self._epochs.get()),
            "--patience",    str(self._patience.get()),
            "--num-reinit",  str(self._reinit.get()),
        ]

        self._run_results.clear()
        self._cur_run   = 0
        self._num_runs  = self._reinit.get()
        self._cur_epoch = 0
        self._max_epochs = self._epochs.get()
        self._chart.reset()
        for item in self._tbl.get_children():
            self._tbl.delete(item)
        self._run_bar["value"] = 0
        self._ep_bar["value"]  = 0
        self._run_lbl.config(text="–")
        self._ep_lbl.config(text="–")

        self._start_btn.state(["disabled"])
        self._stop_btn.state(["!disabled"])
        self._status(f"Training {model}…")

        self._worker = TrainingWorker(cmd, self._q)
        self._worker.start()
        self.after(100, self._poll)

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self._stop_btn.state(["disabled"])
        self._status("Stopping…")

    def _poll(self):
        try:
            while True:
                evt = self._q.get_nowait()
                self._handle(evt)
        except queue.Empty:
            pass
        # reschedule only if still running
        if self._worker and self._stop_btn.instate(["!disabled"]):
            self.after(100, self._poll)

    def _handle(self, evt):
        kind = evt[0]
        if kind == "log":
            self._log(evt[1])
        elif kind == "run_start":
            _, run_id, num_runs = evt
            self._cur_run  = run_id
            self._num_runs = num_runs
            self._cur_epoch = 0
            self._chart.set_run(run_id)
            pct = int(100 * (run_id - 1) / num_runs)
            self._run_bar["value"] = pct
            self._run_lbl.config(text=f"{run_id}/{num_runs}")
            self._ep_bar["value"] = 0
            self._ep_lbl.config(text="–")
        elif kind == "epoch":
            _, ep, loss, val_f1 = evt
            self._cur_epoch = ep
            self._chart.add_epoch(self._cur_run, loss, val_f1)
            pct = int(100 * ep / self._max_epochs)
            self._ep_bar["value"] = pct
            self._ep_lbl.config(text=f"{ep}/{self._max_epochs}")
        elif kind == "run_val_f1":
            _, run_id, vf1 = evt
            self._run_results.setdefault(run_id, {})["val_f1"] = vf1
            self._update_table_row(run_id)
        elif kind == "run_test_f1":
            _, run_id, tf1 = evt
            self._run_results.setdefault(run_id, {})["test_f1"] = tf1
            self._update_table_row(run_id)
            # update run progress
            pct = int(100 * run_id / self._num_runs)
            self._run_bar["value"] = pct
        elif kind == "done":
            status = evt[1]
            self._start_btn.state(["!disabled"])
            self._stop_btn.state(["disabled"])
            if status == "ok":
                self._run_bar["value"] = 100
                self._status(f"Training complete ✓  ({self._model_var.get()})")
            elif status == "stopped":
                self._status("Training stopped.")
            else:
                self._status(f"Training ended: {status}")
            self._worker = None

    def _update_table_row(self, run_id):
        d = self._run_results.get(run_id, {})
        val_f1  = f"{d.get('val_f1',  0):.4f}" if "val_f1"  in d else "–"
        test_f1 = f"{d.get('test_f1', 0):.4f}" if "test_f1" in d else "–"
        # find existing row
        for item in self._tbl.get_children():
            if self._tbl.set(item, "run") == str(run_id):
                self._tbl.set(item, "val_f1",  val_f1)
                self._tbl.set(item, "test_f1", test_f1)
                return
        self._tbl.insert("", "end", values=(run_id, "–", val_f1, test_f1, "–", "–"))
        self._tbl.yview_moveto(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE TAB
# ══════════════════════════════════════════════════════════════════════════════

class InferenceTab(ttk.Frame):

    def __init__(self, parent, log_cb, status_cb):
        super().__init__(parent)
        self._log    = log_cb
        self._status = status_cb
        self._q      = queue.Queue()
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ── left config ──
        cfg = ttk.LabelFrame(top, text="Inference Settings", padding=8, width=320)
        cfg.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        cfg.pack_propagate(False)

        _label(cfg, "Model type", bold=True).pack(anchor="w", pady=(0, 2))
        self._model_var = tk.StringVar(value="BiLSTM")
        for m in ("BiLSTM", "BiGRU"):
            ttk.Radiobutton(cfg, text=m, value=m, variable=self._model_var).pack(
                side=tk.LEFT, padx=4)

        ttk.Separator(cfg, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        self._ckpt_var    = tk.StringVar()
        self._data_var    = tk.StringVar(value=str(ARTIFACT_DIR))
        self._outlog_var  = tk.StringVar(value=str(PROJECT_ROOT / "inference_log.txt"))
        self._batch_var   = tk.IntVar(value=1)
        self._device_var  = tk.StringVar(value="cpu")

        for lbl, var, typ in [
            ("Checkpoint (.pt)", self._ckpt_var, "ckpt"),
            ("Data dir",         self._data_var, "dir"),
            ("Output log",       self._outlog_var, "save"),
        ]:
            _label(cfg, lbl, fg=MUTED).pack(anchor="w")
            row = ttk.Frame(cfg)
            row.pack(fill=tk.X, pady=(0, 4))
            _entry(row, textvariable=var, width=22).pack(side=tk.LEFT, fill=tk.X, expand=True)
            if typ == "dir":
                cmd = lambda v=var: v.set(
                    filedialog.askdirectory(initialdir=v.get() or str(PROJECT_ROOT)) or v.get())
            elif typ == "ckpt":
                cmd = lambda v=var: v.set(
                    filedialog.askopenfilename(
                        initialdir=str(PROJECT_ROOT),
                        filetypes=[("PyTorch checkpoint","*.pt"),("All","*.*")]
                    ) or v.get())
            else:
                cmd = lambda v=var: v.set(
                    filedialog.asksaveasfilename(
                        initialdir=str(PROJECT_ROOT),
                        defaultextension=".txt",
                        filetypes=[("Text log","*.txt"),("All","*.*")]
                    ) or v.get())
            _btn(row, "…", cmd).pack(side=tk.LEFT, padx=2)

        row = ttk.Frame(cfg)
        row.pack(fill=tk.X, pady=2)
        _label(row, "Batch size", fg=MUTED).pack(side=tk.LEFT)
        _spinbox(row, 1, 4096, textvariable=self._batch_var, width=7).pack(side=tk.RIGHT)

        row2 = ttk.Frame(cfg)
        row2.pack(fill=tk.X, pady=2)
        _label(row2, "Device", fg=MUTED).pack(side=tk.LEFT)
        cb = ttk.Combobox(row2, textvariable=self._device_var,
                          values=["cpu","cuda","cuda:0","cuda:1"], width=10)
        cb.pack(side=tk.RIGHT)

        ttk.Separator(cfg, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        self._inf_btn = _btn(cfg, "▶  Run Inference", self._run_inference,
                             style="Accent.TButton")
        self._inf_btn.pack(fill=tk.X)

        # ── right results ──
        res = ttk.Frame(top)
        res.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        met_frame = ttk.LabelFrame(res, text="Metrics", padding=6)
        met_frame.pack(fill=tk.X, pady=(0, 4))
        self._metrics_vars = {}
        metrics_grid = [
            ("Batch latency mean",  "bat_mean"),
            ("Batch latency std",   "bat_std"),
            ("Sample latency mean", "smp_mean"),
            ("Sample latency std",  "smp_std"),
            ("Throughput (sps)",    "throughput"),
            ("Total elapsed (s)",   "elapsed"),
        ]
        for i, (lbl, key) in enumerate(metrics_grid):
            r, c = divmod(i, 2)
            _label(met_frame, lbl + ":", fg=MUTED).grid(
                row=r, column=c*2, sticky="w", padx=(4 if c else 0, 2), pady=2)
            var = tk.StringVar(value="–")
            self._metrics_vars[key] = var
            _label(met_frame, "", fg=ACCENT).grid(
                row=r, column=c*2+1, sticky="w", padx=(0, 12))
            # bind to var dynamically
            lbl_w = _label(met_frame, "", fg=ACCENT)
            lbl_w.configure(textvariable=var)
            lbl_w.grid(row=r, column=c*2+1, sticky="w", padx=(0, 12))

        log_frame = ttk.LabelFrame(res, text="Output", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self._out_text = _scroll_text(log_frame, height=12)
        self._out_text.pack(fill=tk.BOTH, expand=True)
        _btn(log_frame, "Clear", lambda: self._out_text.txt.delete("1.0", tk.END)).pack(
            anchor="e", pady=(4, 0))

    def _run_inference(self):
        model = self._model_var.get()
        ckpt  = self._ckpt_var.get().strip()
        ddir  = self._data_var.get().strip()
        if not ckpt:
            messagebox.showwarning("Missing", "Please select a checkpoint file.")
            return
        if not ddir:
            messagebox.showwarning("Missing", "Please select the data directory.")
            return

        script = ("run_inference_bilstm.py" if model == "BiLSTM"
                  else "run_inference_bigru.py")
        cmd = [
            sys.executable, script,
            "--data-dir",        ddir,
            "--model-ckpt-path", ckpt,
            "--output-log",      self._outlog_var.get(),
            "--batch-size",      str(self._batch_var.get()),
        ]

        self._inf_btn.state(["disabled"])
        self._status(f"Running inference ({model})…")
        for key in self._metrics_vars:
            self._metrics_vars[key].set("–")
        self._out_text.txt.delete("1.0", tk.END)

        worker = InferenceWorker(cmd, self._q)
        worker.start()
        self.after(100, self._poll)

    def _poll(self):
        try:
            while True:
                evt = self._q.get_nowait()
                if evt[0] == "log":
                    line = evt[1]
                    self._out_text.txt.insert(tk.END, line + "\n")
                    self._out_text.txt.see(tk.END)
                    self._log(line)
                    self._parse_metrics(line)
                elif evt[0] == "done":
                    self._inf_btn.state(["!disabled"])
                    status = evt[1]
                    self._status("Inference done ✓" if status == "ok"
                                 else f"Inference ended: {status}")
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)

    _RE_LATENCY_BATCH  = re.compile(r"Batch latency.*?([\d.]+)\s*±\s*([\d.]+)\s*ms")
    _RE_LATENCY_SAMPLE = re.compile(r"Per-sample latency.*?([\d.]+)\s*±\s*([\d.]+)\s*ms")
    _RE_THROUGHPUT     = re.compile(r"Throughput.*?([\d.]+)\s*samples/s")
    _RE_ELAPSED        = re.compile(r"Inference elapsed.*?([\d.]+)s")

    def _parse_metrics(self, line):
        m = self._RE_LATENCY_BATCH.search(line)
        if m:
            self._metrics_vars["bat_mean"].set(f"{float(m.group(1)):.3f} ms")
            self._metrics_vars["bat_std"].set(f"± {float(m.group(2)):.3f} ms")
        m = self._RE_LATENCY_SAMPLE.search(line)
        if m:
            self._metrics_vars["smp_mean"].set(f"{float(m.group(1)):.4f} ms")
            self._metrics_vars["smp_std"].set(f"± {float(m.group(2)):.4f} ms")
        m = self._RE_THROUGHPUT.search(line)
        if m:
            self._metrics_vars["throughput"].set(f"{float(m.group(1)):,.0f}")
        m = self._RE_ELAPSED.search(line)
        if m:
            self._metrics_vars["elapsed"].set(f"{float(m.group(1)):.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA TAB
# ══════════════════════════════════════════════════════════════════════════════

class DataTab(ttk.Frame):

    def __init__(self, parent, log_cb, status_cb):
        super().__init__(parent)
        self._log    = log_cb
        self._status = status_cb
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        self._dir_var = tk.StringVar(value=str(ARTIFACT_DIR))
        _label(top, "Dataset directory:", bold=True).pack(side=tk.LEFT)
        _entry(top, textvariable=self._dir_var, width=45).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        _btn(top, "Browse…", self._browse).pack(side=tk.LEFT, padx=2)
        _btn(top, "Load", self._load, style="Accent.TButton").pack(
            side=tk.LEFT, padx=4)

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # metadata tab
        meta_frame = ttk.Frame(nb)
        nb.add(meta_frame, text="Metadata")
        self._meta_tree = ttk.Treeview(meta_frame, show="tree headings")
        self._meta_tree["columns"] = ("value",)
        self._meta_tree.heading("#0",     text="Key")
        self._meta_tree.heading("value",  text="Value")
        self._meta_tree.column("#0",      width=280)
        self._meta_tree.column("value",   width=400)
        sb = ttk.Scrollbar(meta_frame, command=self._meta_tree.yview)
        self._meta_tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._meta_tree.pack(fill=tk.BOTH, expand=True)

        # files tab
        files_frame = ttk.Frame(nb)
        nb.add(files_frame, text="Files")
        cols = ("name", "shape", "dtype", "size_mb")
        self._file_tbl = ttk.Treeview(files_frame, columns=cols, show="headings")
        for c, w, lbl in [("name",70,"File"),("shape",200,"Shape"),
                           ("dtype",80,"dtype"),("size_mb",80,"Size (MB)")]:
            self._file_tbl.heading(c, text=lbl)
            self._file_tbl.column(c, width=w)
        sb2 = ttk.Scrollbar(files_frame, command=self._file_tbl.yview)
        self._file_tbl.configure(yscrollcommand=sb2.set)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self._file_tbl.pack(fill=tk.BOTH, expand=True)

        # classes tab
        cls_frame = ttk.Frame(nb)
        nb.add(cls_frame, text="Classes")
        self._cls_txt = _scroll_text(cls_frame, height=20)
        self._cls_txt.pack(fill=tk.BOTH, expand=True)

    def _browse(self):
        p = filedialog.askdirectory(initialdir=self._dir_var.get() or str(PROJECT_ROOT))
        if p:
            self._dir_var.set(p)

    def _load(self):
        d = Path(self._dir_var.get())
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            messagebox.showerror("Not found", f"metadata.json not found in:\n{d}")
            return
        with open(meta_path) as f:
            meta = json.load(f)
        self._populate_meta(meta)
        self._populate_files(d, meta)
        self._populate_classes(meta)
        self._status(f"Dataset loaded from {d}")
        self._log(f"[Data] Loaded metadata from {meta_path}")

    def _populate_meta(self, meta):
        for item in self._meta_tree.get_children():
            self._meta_tree.delete(item)
        simple_keys = ["timesteps","stride","n_features","split_strategy"]
        for k in simple_keys:
            if k in meta:
                self._meta_tree.insert("","end",text=k,values=(str(meta[k]),))
        # shapes
        if "shapes" in meta:
            shapes_node = self._meta_tree.insert("","end",text="shapes",values=("",))
            for k, v in meta["shapes"].items():
                self._meta_tree.insert(shapes_node,"end",text=k,values=(str(v),))
        # class_weight_map
        if "class_weight_map" in meta:
            cw_node = self._meta_tree.insert("","end",text="class_weight_map",values=("",))
            for k, v in meta["class_weight_map"].items():
                self._meta_tree.insert(cw_node,"end",text=k,values=(f"{v:.4f}",))
        # feature names
        if "feature_names" in meta:
            fn_node = self._meta_tree.insert("","end",text=f"feature_names ({len(meta['feature_names'])})",values=("",))
            for fn in meta["feature_names"]:
                self._meta_tree.insert(fn_node,"end",text=fn,values=("",))

    def _populate_files(self, d, meta):
        for item in self._file_tbl.get_children():
            self._file_tbl.delete(item)
        for f in sorted(d.glob("*.npy")):
            try:
                arr = np.load(str(f), mmap_mode="r")
                size_mb = f.stat().st_size / 1e6
                self._file_tbl.insert("","end",
                    values=(f.name, str(arr.shape), str(arr.dtype), f"{size_mb:.1f}"))
            except Exception as e:
                self._file_tbl.insert("","end",values=(f.name,"error","",""))

    def _populate_classes(self, meta):
        self._cls_txt.txt.delete("1.0", tk.END)
        classes = meta.get("classes", [])
        cw = meta.get("class_weight_map", {})
        lines = ["  IDX  CLASS                          WEIGHT\n",
                 "  " + "─"*48 + "\n"]
        for i, cls in enumerate(classes):
            w = cw.get(str(i), cw.get(i, "–"))
            wstr = f"{w:.4f}" if isinstance(w, float) else str(w)
            lines.append(f"  {i:3d}  {cls:<30s}  {wstr}\n")
        self._cls_txt.txt.insert(tk.END, "".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TAB
# ══════════════════════════════════════════════════════════════════════════════

class ResultsTab(ttk.Frame):

    def __init__(self, parent, log_cb, status_cb):
        super().__init__(parent)
        self._log    = log_cb
        self._status = status_cb
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        self._summary_var = tk.StringVar()
        _label(top, "Summary file:", bold=True).pack(side=tk.LEFT)
        _entry(top, textvariable=self._summary_var, width=45).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        _btn(top, "Browse…", self._browse).pack(side=tk.LEFT)
        _btn(top, "Load", self._load, style="Accent.TButton").pack(
            side=tk.LEFT, padx=4)

        # quick-load buttons
        qf = ttk.Frame(self)
        qf.pack(fill=tk.X, padx=8)
        _label(qf, "Quick load:", fg=MUTED).pack(side=tk.LEFT)
        _btn(qf, "BiLSTM summary", lambda: self._quick(BILSTM_OUT/"summary.json")).pack(
            side=tk.LEFT, padx=4)
        _btn(qf, "BiGRU summary",  lambda: self._quick(BIGRU_OUT/"summary.json")).pack(
            side=tk.LEFT, padx=2)

        pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # agg stats
        agg_frame = ttk.LabelFrame(pane, text="Aggregate Statistics", padding=6)
        pane.add(agg_frame, weight=0)
        self._agg_vars = {}
        agg_keys = [
            ("best_val_f1",          "Best Val F1"),
            ("test_f1",              "Test F1"),
            ("training_time_s",      "Train time (s)"),
            ("epochs_trained",       "Epochs"),
            ("latency_batch_mean_ms","Batch latency (ms)"),
            ("latency_sample_mean_ms","Sample latency (ms)"),
            ("throughput_sps",       "Throughput (sps)"),
        ]
        for i, (key, lbl) in enumerate(agg_keys):
            r, c = divmod(i, 4)
            _label(agg_frame, lbl+":", fg=MUTED).grid(row=r, column=c*2,
                                                        sticky="w", padx=(4,2), pady=2)
            var = tk.StringVar(value="–")
            self._agg_vars[key] = var
            lw = _label(agg_frame, "")
            lw.configure(textvariable=var, foreground=ACCENT)
            lw.grid(row=r, column=c*2+1, sticky="w", padx=(0,16))

        bottom = ttk.Frame(pane)
        pane.add(bottom, weight=1)

        # per-run table
        tbl_frame = ttk.LabelFrame(bottom, text="Per-Run Results", padding=4)
        tbl_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,4))
        cols = ("run","seed","val_f1","test_f1","epochs","time_s","sps")
        self._res_tbl = ttk.Treeview(tbl_frame, columns=cols, show="headings")
        hdrs = {"run":"Run","seed":"Seed","val_f1":"Val F1","test_f1":"Test F1",
                "epochs":"Epochs","time_s":"Time (s)","sps":"Throughput"}
        ws   = {"run":40,"seed":70,"val_f1":70,"test_f1":70,
                "epochs":55,"time_s":70,"sps":90}
        for c in cols:
            self._res_tbl.heading(c, text=hdrs[c])
            self._res_tbl.column(c, width=ws[c], anchor="center")
        sb3 = ttk.Scrollbar(tbl_frame, command=self._res_tbl.yview)
        self._res_tbl.configure(yscrollcommand=sb3.set)
        sb3.pack(side=tk.RIGHT, fill=tk.Y)
        self._res_tbl.pack(fill=tk.BOTH, expand=True)

        # chart
        chart_frame = ttk.LabelFrame(bottom, text="F1 Scores per Run", padding=4)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._chart = SummaryChart(chart_frame)
        self._chart.pack(fill=tk.BOTH, expand=True)

    def _browse(self):
        p = filedialog.askopenfilename(
            initialdir=str(PROJECT_ROOT),
            filetypes=[("JSON","*.json"),("All","*.*")])
        if p:
            self._summary_var.set(p)

    def _quick(self, path):
        if path.exists():
            self._summary_var.set(str(path))
            self._load()
        else:
            messagebox.showwarning("Not found", f"File not found:\n{path}")

    def _load(self):
        p = Path(self._summary_var.get())
        if not p.exists():
            messagebox.showerror("Not found", f"File not found:\n{p}")
            return
        with open(p) as f:
            data = json.load(f)
        self._populate_agg(data.get("aggregated_stats", {}))
        self._populate_runs(data.get("runs", []))
        self._status(f"Results loaded from {p.name}")
        self._log(f"[Results] Loaded {p}")

    def _populate_agg(self, stats):
        for key, var in self._agg_vars.items():
            if key in stats:
                m = stats[key].get("mean", 0)
                s = stats[key].get("std",  0)
                var.set(f"{m:.4f} ± {s:.4f}")
            else:
                var.set("–")

    def _populate_runs(self, runs):
        for item in self._res_tbl.get_children():
            self._res_tbl.delete(item)
        for r in runs:
            t = r.get("timing", {})
            self._res_tbl.insert("", "end", values=(
                r.get("run","–"),
                r.get("seed","–"),
                f"{r.get('best_val_f1',0):.4f}",
                f"{r.get('test_f1',0):.4f}",
                r.get("epochs_trained","–"),
                f"{r.get('training_time_s',0):.1f}",
                f"{t.get('throughput_sps',0):,.0f}",
            ))
        self._chart.plot(runs)


# ══════════════════════════════════════════════════════════════════════════════
# LOGS TAB
# ══════════════════════════════════════════════════════════════════════════════

class LogsTab(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=6, pady=4)
        _label(top, "Console Log", bold=True).pack(side=tk.LEFT)
        _btn(top, "Clear", self._clear).pack(side=tk.RIGHT)
        _btn(top, "Save…", self._save).pack(side=tk.RIGHT, padx=4)
        self._txt_frame = _scroll_text(self, height=35)
        self._txt_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self._txt = self._txt_frame.txt
        self._txt.tag_config("ts",    foreground=MUTED)
        self._txt.tag_config("err",   foreground=RED)
        self._txt.tag_config("info",  foreground=ACCENT)
        self._txt.tag_config("epoch", foreground=GREEN)

    def append(self, line: str):
        ts = datetime.now().strftime("%H:%M:%S")
        tag = "epoch" if "Epoch" in line else \
              "err"   if any(w in line for w in ["Error","error","Traceback","failed"]) else \
              "info"  if line.startswith("[") else ""
        self._txt.insert(tk.END, f"[{ts}] ", "ts")
        self._txt.insert(tk.END, line + "\n", tag)
        self._txt.see(tk.END)

    def _clear(self):
        self._txt.delete("1.0", tk.END)

    def _save(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if p:
            with open(p, "w") as f:
                f.write(self._txt.get("1.0", tk.END))


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGER TAB
# ══════════════════════════════════════════════════════════════════════════════

class CheckpointsTab(ttk.Frame):
    """Browse, inspect, and compare saved model checkpoints."""

    def __init__(self, parent, status_cb):
        super().__init__(parent)
        self._status = status_cb
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)
        _label(top, "Search directory:", bold=True).pack(side=tk.LEFT)
        self._dir_var = tk.StringVar(value=str(PROJECT_ROOT))
        _entry(top, textvariable=self._dir_var, width=40).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        _btn(top, "Browse…", self._browse).pack(side=tk.LEFT)
        _btn(top, "Scan", self._scan, style="Accent.TButton").pack(
            side=tk.LEFT, padx=4)

        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # checkpoint list
        left = ttk.LabelFrame(pane, text="Checkpoints", padding=4)
        pane.add(left, weight=1)
        cols = ("name","val_f1","test_f1","epochs","time_s")
        self._ckpt_tbl = ttk.Treeview(left, columns=cols, show="headings",
                                       selectmode="browse")
        for c, w, lbl in [("name",160,"File"),("val_f1",75,"Val F1"),
                           ("test_f1",75,"Test F1"),("epochs",60,"Epochs"),
                           ("time_s",70,"Time (s)")]:
            self._ckpt_tbl.heading(c, text=lbl)
            self._ckpt_tbl.column(c, width=w)
        self._ckpt_tbl.bind("<<TreeviewSelect>>", self._on_select)
        sb = ttk.Scrollbar(left, command=self._ckpt_tbl.yview)
        self._ckpt_tbl.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._ckpt_tbl.pack(fill=tk.BOTH, expand=True)

        # details
        right = ttk.LabelFrame(pane, text="Checkpoint Details", padding=8)
        pane.add(right, weight=1)
        self._det_txt = _scroll_text(right, height=20)
        self._det_txt.pack(fill=tk.BOTH, expand=True)
        self._paths = {}   # iid -> full path

    def _browse(self):
        p = filedialog.askdirectory(initialdir=self._dir_var.get() or str(PROJECT_ROOT))
        if p:
            self._dir_var.set(p)

    def _scan(self):
        for item in self._ckpt_tbl.get_children():
            self._ckpt_tbl.delete(item)
        self._paths.clear()
        root = Path(self._dir_var.get())
        ckpts = sorted(root.rglob("*.pt"))
        if not ckpts:
            messagebox.showinfo("No checkpoints", f"No .pt files found in:\n{root}")
            return
        import torch
        for ck in ckpts:
            try:
                data = torch.load(str(ck), map_location="cpu", weights_only=False)
                vf1 = data.get("best_val_f1", float("nan"))
                tf1 = data.get("test_f1",    float("nan"))
                ep  = data.get("epochs_trained", "–")
                tt  = data.get("training_time_s", float("nan"))
                iid = self._ckpt_tbl.insert("", "end", values=(
                    ck.name,
                    f"{vf1:.4f}" if isinstance(vf1, float) else str(vf1),
                    f"{tf1:.4f}" if isinstance(tf1, float) else str(tf1),
                    ep,
                    f"{tt:.1f}"  if isinstance(tt, float) else str(tt),
                ))
                self._paths[iid] = ck
            except Exception as e:
                self._ckpt_tbl.insert("", "end", values=(ck.name, "error","","",""))
        self._status(f"Found {len(ckpts)} checkpoint(s) in {root}")

    def _on_select(self, _event):
        sel = self._ckpt_tbl.selection()
        if not sel:
            return
        iid  = sel[0]
        path = self._paths.get(iid)
        if not path:
            return
        import torch
        try:
            data = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as e:
            self._det_txt.txt.delete("1.0", tk.END)
            self._det_txt.txt.insert(tk.END, f"Error loading checkpoint:\n{e}")
            return
        lines = [f"File : {path}\n", f"Size : {path.stat().st_size/1e6:.2f} MB\n\n"]
        keys_to_show = ["best_val_f1","test_f1","training_time_s","epochs_trained"]
        for k in keys_to_show:
            if k in data:
                lines.append(f"{k}: {data[k]}\n")
        if "timing" in data:
            lines.append("\n--- Timing ---\n")
            for k, v in data["timing"].items():
                lines.append(f"  {k}: {v}\n")
        if "model_state" in data:
            ms = data["model_state"]
            n_params = sum(v.numel() for v in ms.values())
            lines.append(f"\n--- Model state ---\n")
            lines.append(f"  Parameters : {n_params:,}\n")
            lines.append(f"  Layers     : {len(ms)}\n")
            for name, tensor in list(ms.items())[:10]:
                lines.append(f"  {name:40s}  {list(tensor.shape)}\n")
            if len(ms) > 10:
                lines.append(f"  ... (+{len(ms)-10} more layers)\n")
        self._det_txt.txt.delete("1.0", tk.END)
        self._det_txt.txt.insert(tk.END, "".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("X-Edge-ESN  |  BiLSTM / BiGRU IDS")
        self.geometry("1200x820")
        self.minsize(900, 600)
        self.configure(bg=BG)
        _apply_style(self)
        self._build_menu()
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

    # ── menu ────────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = tk.Menu(self, bg=BG2, fg=FG, activebackground=ACCENT,
                     activeforeground=BG, relief="flat", tearoff=False)
        self.configure(menu=mb)

        fm = tk.Menu(mb, bg=BG2, fg=FG, activebackground=ACCENT,
                     activeforeground=BG, tearoff=False)
        mb.add_cascade(label="File", menu=fm)
        fm.add_command(label="Open project folder…",
                       command=lambda: os.startfile(str(PROJECT_ROOT))
                       if sys.platform == "win32"
                       else subprocess.Popen(["xdg-open", str(PROJECT_ROOT)]))
        fm.add_separator()
        fm.add_command(label="Quit", command=self.quit)

        hm = tk.Menu(mb, bg=BG2, fg=FG, activebackground=ACCENT,
                     activeforeground=BG, tearoff=False)
        mb.add_cascade(label="Help", menu=hm)
        hm.add_command(label="About", command=self._about)

    # ── header ──────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG3, height=48)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="X-Edge-ESN", bg=BG3, fg=ACCENT,
                 font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT, padx=14)
        tk.Label(hdr, text="BiLSTM / BiGRU Intrusion Detection System",
                 bg=BG3, fg=MUTED, font=("Segoe UI", 10)).pack(
                     side=tk.LEFT, padx=0)
        self._ts_lbl = tk.Label(hdr, text="", bg=BG3, fg=MUTED,
                                font=("Segoe UI", 9))
        self._ts_lbl.pack(side=tk.RIGHT, padx=14)
        self._tick()

    def _tick(self):
        self._ts_lbl.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._tick)

    # ── notebook ────────────────────────────────────────────────────────────

    def _build_notebook(self):
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self._logs_tab = LogsTab(self._nb)

        def log_cb(line):
            self._logs_tab.append(line)

        def status_cb(msg):
            self._statusvar.set(msg)

        self._train_tab   = TrainingTab(self._nb, log_cb, status_cb)
        self._infer_tab   = InferenceTab(self._nb, log_cb, status_cb)
        self._data_tab    = DataTab(self._nb, log_cb, status_cb)
        self._results_tab = ResultsTab(self._nb, log_cb, status_cb)
        self._ckpts_tab   = CheckpointsTab(self._nb, status_cb)

        for tab, label in [
            (self._train_tab,   "  Training  "),
            (self._infer_tab,   "  Inference  "),
            (self._data_tab,    "  Data  "),
            (self._results_tab, "  Results  "),
            (self._ckpts_tab,   "  Checkpoints  "),
            (self._logs_tab,    "  Logs  "),
        ]:
            self._nb.add(tab, text=label)

    # ── status bar ──────────────────────────────────────────────────────────

    def _build_statusbar(self):
        self._statusvar = tk.StringVar(value="Ready")
        bar = tk.Frame(self, bg=BG3, height=26)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self._statusvar, bg=BG3, fg=MUTED,
                 font=("Segoe UI", 9), anchor="w").pack(
                     side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        tk.Label(bar, text=f"Project: {PROJECT_ROOT}",
                 bg=BG3, fg=MUTED, font=("Segoe UI", 8)).pack(
                     side=tk.RIGHT, padx=10)

    def _about(self):
        messagebox.showinfo(
            "About X-Edge-ESN GUI",
            "X-Edge-ESN GUI\n\n"
            "Interactive interface for training and evaluating\n"
            "BiLSTM / BiGRU Intrusion Detection Systems\n"
            "on the CICIDS2017 dataset.\n\n"
            f"Project root:\n{PROJECT_ROOT}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
