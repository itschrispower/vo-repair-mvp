"""
src/gui_app.py  —  VO Repair desktop GUI

Inputs:
  AAF Full Bounce  the full Pro Tools session bounce WAV (aaf_reference.wav)
  VO BU            the raw VO backup recording (VOBU_48k.wav)
  AAF              the Pro Tools AAF export (.aaf)

The output folder auto-populates from the AAF location.
Each run writes into a timestamped subfolder (run_YYYYMMDD_HHMMSS/) so
previous results are never overwritten.

Engine stdout is streamed live into the log area. Progress is parsed from
the standard "  [  3/ 47] …" format printed by engine.py.
"""
from __future__ import annotations

import io
import json
import queue
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox

import run_job


# ── Colours & fonts ───────────────────────────────────────────────────────────
BG        = "#1e1e1e"
PANEL     = "#2a2a2a"
ACCENT    = "#4a9eff"
BTN_DIS   = "#3a3a3a"   # disabled button background
OK_COL    = "#4caf50"
REVIEW_COL = "#ff9800"
FAIL_COL  = "#f44336"
FORCE_COL = "#c792ea"   # soft purple — FORCED (low-confidence) clips
FG        = "#e0e0e0"
FG_DIM    = "#888888"
FONT_SM   = ("Helvetica Neue", 11)
MONO      = ("Menlo", 11)

# ── Engine-log parsers ────────────────────────────────────────────────────────
# Matches:  "Matching 47 clips against VOBU…"
_TOTAL_RE    = re.compile(r'Matching (\d+) clips')
# Matches:  "  [  3/ 47] clip_name …"
_PROGRESS_RE = re.compile(r'\[\s*(\d+)/\s*(\d+)\]')

# Status display helpers
_STATUS_ICONS  = {"ok": "✓", "review": "~", "fail": "✗", "forced": "⚡"}
_STATUS_COLORS = {"ok": OK_COL, "review": REVIEW_COL, "fail": FAIL_COL, "forced": FORCE_COL}
_STATUS_LABELS = {"ok": "OK", "review": "REVIEW", "fail": "FAIL", "forced": "FORCED"}


# ── Stdout capture ────────────────────────────────────────────────────────────

class _QueueWriter:
    """Redirects sys.stdout writes into a Queue for live GUI display."""

    def __init__(self, q: queue.Queue) -> None:
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self)           -> None: pass
    def isatty(self)          -> bool: return False
    def fileno(self):
        raise io.UnsupportedOperation("fileno")


# ── File-picker widget ────────────────────────────────────────────────────────

class FilePicker(tk.Frame):
    """
    Labelled row: [label][path display][Choose… button]

    initialdir_fn — optional callable() → str, returns the initial directory
                    for the file dialog.  Pass a lambda that reads from the
                    App's _last_dir so all pickers share the latest location.
    """

    def __init__(
        self,
        parent,
        label: str,
        mode: str = "file",
        filetypes=None,
        initialdir_fn=None,
        **kwargs,
    ):
        super().__init__(parent, bg=PANEL, **kwargs)
        self._mode = mode
        self._filetypes = filetypes or []
        self._initialdir_fn = initialdir_fn
        self._path = tk.StringVar()
        self._on_change_cb = None

        tk.Label(
            self, text=label, font=FONT_SM, bg=PANEL,
            fg=FG_DIM, width=17, anchor="w",
        ).pack(side="left", padx=(12, 4))

        self._path_lbl = tk.Label(
            self, textvariable=self._path, font=FONT_SM,
            bg=PANEL, fg=FG, anchor="w", width=34, relief="flat",
        )
        self._path_lbl.pack(side="left")

        tk.Button(
            self, text="Choose…", font=FONT_SM, bg="#555", fg="white",
            relief="flat", padx=10, cursor="hand2",
            command=self._pick,
        ).pack(side="right", padx=12, pady=6)

    def _pick(self) -> None:
        initialdir = self._initialdir_fn() if self._initialdir_fn else None
        if self._mode == "folder":
            path = filedialog.askdirectory(
                title="Select output folder",
                initialdir=initialdir,
            )
        else:
            path = filedialog.askopenfilename(
                title="Select file",
                filetypes=self._filetypes or [("All files", "*.*")],
                initialdir=initialdir,
            )
        if path:
            self._path.set(path)
            if self._on_change_cb:
                self._on_change_cb(path)

    def get(self) -> str:
        return self._path.get().strip()

    def set(self, value: str) -> None:
        self._path.set(value)

    def clear(self) -> None:
        self._path.set("")

    def is_set(self) -> bool:
        return bool(self.get())

    def on_change(self, callback) -> None:
        self._on_change_cb = callback


# ── Review window ─────────────────────────────────────────────────────────────

class ReviewWindow(tk.Toplevel):
    """
    Modal-ish window that lists every non-OK clip with:
      • Status badge (colour-coded) + clip name + confidence
      • First reason string
      • "Reviewed" checkbox
      • Notes text entry

    Bottom row: [Mark All Reviewed]  [Save Review]  [Close]

    Saving writes  run_dir/session_review.json.
    """

    def __init__(self, parent: tk.Tk, run_dir: Path, flagged: list[dict]) -> None:
        super().__init__(parent)
        self._run_dir = run_dir
        self._flagged = flagged
        self._reviewed_vars: list[tk.BooleanVar] = []
        self._notes_vars:    list[tk.StringVar]  = []
        self._session_var    = tk.BooleanVar(value=False)
        self._saved          = False

        self.title("Review Flagged Clips")
        self.configure(bg=BG)
        self.geometry("720x540")
        self.resizable(True, True)
        self.minsize(600, 400)

        # Keep on top of parent
        self.transient(parent)
        self.grab_set()

        self._build()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=16, pady=(14, 0))

        tk.Label(
            hdr, text="Review Flagged Clips",
            font=("Helvetica Neue", 16, "bold"),
            bg=BG, fg=FG,
        ).pack(side="left")

        # Counts badges in header
        counts: dict[str, int] = {}
        for c in self._flagged:
            st = c.get("status", "?").lower()
            counts[st] = counts.get(st, 0) + 1
        badge_parts = []
        for st in ("forced", "review", "fail"):
            n = counts.get(st, 0)
            if n:
                badge_parts.append(
                    f"  {_STATUS_ICONS.get(st,'?')} {n} {_STATUS_LABELS.get(st, st.upper())}"
                )
        if badge_parts:
            tk.Label(
                hdr, text="".join(badge_parts),
                font=FONT_SM, bg=BG, fg=FG_DIM,
            ).pack(side="left", padx=(12, 0))

        tk.Label(
            self,
            text="Mark each clip reviewed and add notes before delivery.",
            font=FONT_SM, bg=BG, fg=FG_DIM,
        ).pack(anchor="w", padx=16, pady=(2, 6))

        # ── Session-level "mark all" checkbox ────────────────────────────────
        sess_row = tk.Frame(self, bg=PANEL, padx=12, pady=6,
                            highlightthickness=1, highlightbackground="#444")
        sess_row.pack(fill="x", padx=16, pady=(0, 8))

        tk.Checkbutton(
            sess_row, text="  Mark entire session as reviewed (all clips signed off)",
            variable=self._session_var,
            bg=PANEL, fg=FG, selectcolor="#3a3a3a", activebackground=PANEL,
            font=FONT_SM, cursor="hand2",
        ).pack(side="left")

        # ── Scrollable clip list ──────────────────────────────────────────────
        list_outer = tk.Frame(self, bg=PANEL,
                              highlightthickness=1, highlightbackground="#444")
        list_outer.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        canvas = tk.Canvas(list_outer, bg=PANEL, bd=0, highlightthickness=0)
        vscroll = tk.Scrollbar(list_outer, orient="vertical", command=canvas.yview)
        self._scroll_frame = tk.Frame(canvas, bg=PANEL)

        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Mousewheel scroll on all platforms
        def _on_wheel(event):
            if sys.platform == "darwin":
                canvas.yview_scroll(-1 * event.delta, "units")
            else:
                canvas.yview_scroll(-1 * (event.delta // 120), "units")
        canvas.bind_all("<MouseWheel>", _on_wheel)
        canvas.bind_all("<Button-4>",   lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>",   lambda e: canvas.yview_scroll( 1, "units"))

        # Per-clip rows
        for i, clip in enumerate(self._flagged):
            self._build_clip_row(i, clip)

        # ── Bottom button bar ─────────────────────────────────────────────────
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(fill="x", padx=16, pady=(0, 14))

        tk.Button(
            btn_row, text="Mark All Reviewed",
            font=FONT_SM, bg="#444", fg=FG, relief="flat",
            padx=10, pady=5, cursor="hand2",
            command=self._mark_all,
        ).pack(side="left", padx=(0, 8))

        self._save_btn = tk.Button(
            btn_row, text="💾  Save Review",
            font=FONT_SM, bg=ACCENT, fg="white", relief="flat",
            padx=12, pady=5, cursor="hand2",
            command=self._save,
        )
        self._save_btn.pack(side="left", padx=(0, 8))

        tk.Button(
            btn_row, text="Close",
            font=FONT_SM, bg="#444", fg=FG, relief="flat",
            padx=10, pady=5, cursor="hand2",
            command=self._close,
        ).pack(side="left")

        # Saved indicator label (hidden until save)
        self._saved_lbl = tk.Label(
            btn_row, text="✓ Saved", font=FONT_SM, bg=BG, fg=OK_COL,
        )

    def _build_clip_row(self, idx: int, clip: dict) -> None:
        """Add one clip row to the scroll frame."""
        st      = clip.get("status", "?").lower()
        color   = _STATUS_COLORS.get(st, FG)
        icon    = _STATUS_ICONS.get(st, "?")
        label   = _STATUS_LABELS.get(st, st.upper())
        name    = clip.get("clip_name", "?")
        conf    = clip.get("confidence", 0.0)
        reasons = clip.get("reasons", [])

        # Outer row frame
        row = tk.Frame(self._scroll_frame, bg=PANEL, pady=8)
        row.pack(fill="x", padx=0, pady=0)

        # ── Status badge + name + conf ────────────────────────────────────────
        top = tk.Frame(row, bg=PANEL)
        top.pack(fill="x", padx=10)

        tk.Label(
            top, text=f"{icon} {label}",
            font=("Helvetica Neue", 10, "bold"),
            bg=PANEL, fg=color, width=10, anchor="w",
        ).pack(side="left")

        tk.Label(
            top, text=name,
            font=MONO, bg=PANEL, fg=FG, anchor="w",
        ).pack(side="left", padx=(4, 0))

        tk.Label(
            top, text=f"  conf={conf:.0%}",
            font=FONT_SM, bg=PANEL, fg=FG_DIM,
        ).pack(side="left")

        # ── First reason ──────────────────────────────────────────────────────
        if reasons:
            reason_text = reasons[0][:90]
            tk.Label(
                row, text=f"    ⚑  {reason_text}",
                font=("Helvetica Neue", 10), bg=PANEL, fg=FG_DIM, anchor="w",
            ).pack(fill="x", padx=10, pady=(2, 0))

        # ── Reviewed checkbox + notes entry ───────────────────────────────────
        ctrl = tk.Frame(row, bg=PANEL)
        ctrl.pack(fill="x", padx=10, pady=(4, 0))

        rev_var = tk.BooleanVar(value=False)
        self._reviewed_vars.append(rev_var)

        tk.Checkbutton(
            ctrl, text="Reviewed",
            variable=rev_var,
            bg=PANEL, fg=FG, selectcolor="#3a3a3a", activebackground=PANEL,
            font=FONT_SM, cursor="hand2",
        ).pack(side="left", padx=(0, 10))

        tk.Label(ctrl, text="Notes:", font=FONT_SM, bg=PANEL, fg=FG_DIM,
                 ).pack(side="left")

        notes_var = tk.StringVar()
        self._notes_vars.append(notes_var)

        tk.Entry(
            ctrl, textvariable=notes_var, font=FONT_SM,
            bg="#333", fg=FG, insertbackground=FG, relief="flat",
            width=42,
        ).pack(side="left", padx=(4, 0))

        # Separator (between clips, not after the last)
        sep_frame = tk.Frame(self._scroll_frame, bg="#3a3a3a", height=1)
        sep_frame.pack(fill="x")

    # ── Button actions ────────────────────────────────────────────────────────

    def _mark_all(self) -> None:
        for v in self._reviewed_vars:
            v.set(True)
        self._session_var.set(True)

    def _save(self) -> None:
        clips_out = []
        for clip, rev_var, notes_var in zip(
            self._flagged, self._reviewed_vars, self._notes_vars
        ):
            clips_out.append({
                "clip_name":  clip.get("clip_name", "?"),
                "status":     clip.get("status", "?"),
                "confidence": clip.get("confidence", 0.0),
                "reasons":    clip.get("reasons", []),
                "reviewed":   rev_var.get(),
                "notes":      notes_var.get().strip(),
            })

        payload = {
            "session_reviewed": self._session_var.get(),
            "timestamp":        datetime.now().isoformat(timespec="seconds"),
            "run_folder":       str(self._run_dir),
            "flagged_count":    len(self._flagged),
            "clips":            clips_out,
        }

        out_path = self._run_dir / "session_review.json"
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            messagebox.showerror("Save Failed", f"Could not write review file:\n{exc}")
            return

        self._saved = True
        self._save_btn.config(text="✓ Saved", bg="#3a6a4a", state="disabled")
        self._saved_lbl.pack(side="left", padx=(8, 0))

    def _close(self) -> None:
        # Unbind mousewheel before destroying to avoid errors
        try:
            self.unbind_all("<MouseWheel>")
            self.unbind_all("<Button-4>")
            self.unbind_all("<Button-5>")
        except Exception:
            pass
        self.grab_release()
        self.destroy()


# ── Main application window ───────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VO Repair")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.geometry("680x780")

        # State
        self._processing    = False
        self._log_q         = queue.Queue()
        self._last_run_dir: Path | None = None
        self._last_dir      = str(Path.home())   # shared across all pickers
        self._total_clips   = 0
        self._report_data:  list = []             # filled after each run

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Header ───────────────────────────────────────────────────────────
        tk.Label(
            self, text="VO Repair",
            font=("Helvetica Neue", 22, "bold"),
            bg=BG, fg=FG,
        ).pack(anchor="w", padx=20, pady=(16, 10))

        # ── Input panel ───────────────────────────────────────────────────────
        panel = tk.Frame(
            self, bg=PANEL, bd=0,
            highlightthickness=1, highlightbackground="#444",
        )
        panel.pack(fill="x", padx=20, pady=(0, 8))

        self._ref_wav = FilePicker(
            panel, "AAF Full Bounce",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialdir_fn=lambda: self._last_dir,
        )
        self._ref_wav.pack(fill="x")
        self._ref_wav.on_change(self._on_file_picked)
        self._sep(panel)

        self._vobu_wav = FilePicker(
            panel, "VO BU",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialdir_fn=lambda: self._last_dir,
        )
        self._vobu_wav.pack(fill="x")
        self._vobu_wav.on_change(self._on_file_picked)
        self._sep(panel)

        self._aaf = FilePicker(
            panel, "AAF",
            filetypes=[("AAF files", "*.aaf"), ("All files", "*.*")],
            initialdir_fn=lambda: self._last_dir,
        )
        self._aaf.pack(fill="x")
        self._aaf.on_change(self._on_aaf_picked)
        self._sep(panel)

        self._out_folder = FilePicker(
            panel, "Output Folder", mode="folder",
            initialdir_fn=lambda: self._last_dir,
        )
        self._out_folder.pack(fill="x")
        self._out_folder.on_change(self._on_file_picked)

        # ── Button row ────────────────────────────────────────────────────────
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=(8, 2))

        self._btn = tk.Button(
            btn_row, text="PROCESS",
            font=("Helvetica Neue", 14, "bold"),
            bg=BTN_DIS, fg="#666", relief="flat",
            padx=24, pady=7, cursor="hand2",
            state="disabled",
            command=self._run,
        )
        self._btn.pack(side="left", padx=(0, 8))

        self._reset_btn = tk.Button(
            btn_row, text="Clear / New Session",
            font=FONT_SM, bg="#444", fg=FG,
            relief="flat", padx=12, pady=7, cursor="hand2",
            command=self._reset,
        )
        self._reset_btn.pack(side="left")

        # ── Status line ───────────────────────────────────────────────────────
        self._status_var = tk.StringVar(
            value="Select AAF Full Bounce, VO BU, and AAF to enable processing."
        )
        tk.Label(
            self, textvariable=self._status_var, font=FONT_SM,
            bg=BG, fg=FG_DIM, wraplength=640, justify="left",
        ).pack(padx=20, pady=(4, 2), anchor="w")

        # ── Action row ────────────────────────────────────────────────────────
        action_row = tk.Frame(self, bg=BG)
        action_row.pack(fill="x", padx=20, pady=(2, 4))

        self._open_btn = tk.Button(
            action_row, text="📂  Open Output Folder",
            font=FONT_SM, bg="#444", fg="#aaa",
            relief="flat", padx=12, pady=5, cursor="hand2",
            state="disabled",
            command=self._open_output_folder,
        )
        self._open_btn.pack(side="left", padx=(0, 8))

        self._review_btn = tk.Button(
            action_row, text="✎  Review Flagged Clips",
            font=FONT_SM, bg="#444", fg="#aaa",
            relief="flat", padx=12, pady=5, cursor="hand2",
            state="disabled",
            command=self._open_review_window,
        )
        self._review_btn.pack(side="left")

        # ── Log / results area ────────────────────────────────────────────────
        self._results = scrolledtext.ScrolledText(
            self, font=MONO, bg=PANEL, fg=FG,
            relief="flat", padx=12, pady=12,
            state="disabled", height=20, width=76,
            insertbackground=FG,
        )
        self._results.pack(padx=20, pady=(4, 20), fill="both", expand=True)

        self._results.tag_config("ok",      foreground=OK_COL)
        self._results.tag_config("review",  foreground=REVIEW_COL)
        self._results.tag_config("fail",    foreground=FAIL_COL)
        self._results.tag_config("forced",  foreground=FORCE_COL)
        self._results.tag_config("head",    foreground=ACCENT,
                                 font=("Helvetica Neue", 12, "bold"))
        self._results.tag_config("subhead", foreground=ACCENT,
                                 font=("Helvetica Neue", 11, "bold"))
        self._results.tag_config("dim",     foreground=FG_DIM)
        self._results.tag_config("warn",    foreground=REVIEW_COL,
                                 font=("Helvetica Neue", 11, "bold"))
        self._results.tag_config("path",    foreground="#7ec8e3")
        self._results.tag_config("good",    foreground=OK_COL,
                                 font=("Helvetica Neue", 11, "bold"))

    def _sep(self, parent) -> None:
        tk.Frame(parent, bg="#444", height=1).pack(fill="x")

    # ── Input-change callbacks ────────────────────────────────────────────────

    def _on_file_picked(self, path: str) -> None:
        self._last_dir = str(Path(path).parent)
        self._update_process_btn()

    def _on_aaf_picked(self, path: str) -> None:
        self._last_dir = str(Path(path).parent)
        auto_out = Path(path).parent / "VO_Repair_Output"
        self._out_folder.set(str(auto_out))
        self._update_process_btn()

    def _update_process_btn(self, *_) -> None:
        all_set = (
            self._ref_wav.is_set()
            and self._vobu_wav.is_set()
            and self._aaf.is_set()
            and self._out_folder.is_set()
        )
        if not self._processing:
            if all_set:
                self._btn.config(state="normal", bg=ACCENT, fg="white")
            else:
                self._btn.config(state="disabled", bg=BTN_DIS, fg="#666")

    # ── Processing ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        if self._processing:
            return

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self._out_folder.get()) / f"run_{ts}"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._set_status(f"Cannot create output folder: {exc}")
            return

        self._last_run_dir = run_dir
        self._processing   = True
        self._total_clips  = 0
        self._report_data  = []

        self._btn.config(state="disabled", bg=BTN_DIS, fg="#666", text="Processing…")
        self._reset_btn.config(state="disabled")
        self._open_btn.config(state="disabled", bg="#444", fg="#aaa")
        self._review_btn.config(state="disabled", bg="#444", fg="#aaa")

        self._clear_results()
        self._write(f"Run folder: {run_dir}\n\n", "dim")
        self._set_status("Starting pipeline…")

        # Drain stale log items from a previous run
        while not self._log_q.empty():
            try:
                self._log_q.get_nowait()
            except queue.Empty:
                break

        threading.Thread(
            target=self._process_thread,
            args=(run_dir,),
            daemon=True,
        ).start()
        self.after(100, self._poll_log)

    def _process_thread(self, run_dir: Path) -> None:
        log_writer = _QueueWriter(self._log_q)
        old_stdout = sys.stdout
        sys.stdout  = log_writer
        error_msg   = None

        try:
            run_job.process_files(
                aaf_path      = Path(self._aaf.get()),
                ref_wav_path  = Path(self._ref_wav.get()),
                vobu_wav_path = Path(self._vobu_wav.get()),
                output_dir    = run_dir,
            )
        except Exception as exc:
            import traceback
            error_msg = f"{exc}\n\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout
            self._log_q.put(None)   # sentinel

        if error_msg:
            self.after(0, self._show_error, error_msg)
        else:
            self.after(0, self._show_results, run_dir)

    def _poll_log(self) -> None:
        try:
            while True:
                item = self._log_q.get_nowait()

                if item is None:
                    return  # sentinel — stop polling

                m = _TOTAL_RE.search(item)
                if m:
                    self._total_clips = int(m.group(1))
                    self._set_status(
                        f"Processing — {self._total_clips} clips to match…"
                    )

                m = _PROGRESS_RE.search(item)
                if m:
                    cur   = int(m.group(1))
                    total = int(m.group(2))
                    self._total_clips = total
                    pct = int(cur / total * 100) if total > 0 else 0
                    self._set_status(f"Clip {cur} of {total}  ({pct}%)")

                self._write(item, "dim")

        except queue.Empty:
            pass

        if self._processing:
            self.after(100, self._poll_log)

    # ── Result display ────────────────────────────────────────────────────────

    def _show_results(self, run_dir: Path) -> None:
        self._processing = False

        # Load JSON report
        reports = sorted(run_dir.glob("*_report.json"))
        data: list = []
        if reports:
            try:
                with open(reports[0], encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                self._write(f"\nCould not read report: {exc}\n", "warn")

        self._report_data = data

        # Tally counts
        counts: dict[str, int] = {}
        for item in data:
            st = item.get("status", "?").lower()
            counts[st] = counts.get(st, 0) + 1

        ok_n     = counts.get("ok",     0)
        rev_n    = counts.get("review", 0)
        fail_n   = counts.get("fail",   0)
        forced_n = counts.get("forced", 0)
        total_n  = len(data)

        wav_files    = sorted(run_dir.glob("*_final.wav"))
        aaf_files    = sorted(run_dir.glob("*_rebuilt.aaf"))
        report_files = sorted(run_dir.glob("*_report.json"))
        has_aaf      = bool(aaf_files)
        needs_review = (rev_n + fail_n + forced_n) > 0

        # ── Summary block ─────────────────────────────────────────────────────
        self._write("\n" + "─" * 44 + "\n", "dim")
        self._write("RESULTS\n\n", "head")

        # Counts row
        if total_n:
            self._write(f"  {total_n} clips  │  ", "dim")
            self._write(f"✓ {ok_n} OK", "ok")
            if rev_n:
                self._write(f"  ~ {rev_n} REVIEW", "review")
            if forced_n:
                self._write(f"  ⚡ {forced_n} FORCED", "forced")
            if fail_n:
                self._write(f"  ✗ {fail_n} FAIL", "fail")
            self._write("\n", "dim")

        # Delivery status line
        self._write("\n", "dim")
        if not needs_review and total_n > 0:
            self._write("  ✓  All clips matched — ready for delivery.\n", "good")
        elif forced_n:
            self._write(
                f"  ⚠  {forced_n} FORCED clip(s) written with low confidence."
                f"  Audit before delivery.\n",
                "warn",
            )
        elif rev_n or fail_n:
            self._write(
                f"  ~  {rev_n + fail_n} clip(s) flagged for review.\n", "warn"
            )

        # AAF status
        self._write("\n", "dim")
        if has_aaf:
            self._write(f"  AAF:    {aaf_files[0].name}\n", "path")
        else:
            self._write("  AAF:    not generated (see log for error)\n", "review")
        if wav_files:
            self._write(f"  WAV:    {wav_files[0].name}\n", "path")
        if report_files:
            self._write(f"  Report: {report_files[0].name}\n", "dim")
        self._write(f"  Folder: {run_dir}\n", "dim")

        # ── Per-clip details ──────────────────────────────────────────────────
        if data:
            self._write("\n─── PER-CLIP DETAILS ────────────────────────\n", "dim")

            for item in data:
                st     = item.get("status", "?").lower()
                conf   = item.get("confidence", 0.0)
                name   = item.get("clip_name", "?")
                start  = item.get("timeline_start_tc", "?")
                end_   = item.get("timeline_end_tc", "?")
                stab   = item.get("stability", 1.0)
                icon   = _STATUS_ICONS.get(st, "?")
                tag    = st if st in ("ok", "review", "fail", "forced") else "dim"

                flags = ""
                if not item.get("consistency_ok", True):
                    flags += " [!trend]"
                if item.get("drift_jump_s", 0.0) > 3.0:
                    flags += " [!jump]"

                self._write(
                    f"  {icon}  {name:<28}  {start}→{end_}"
                    f"  conf={conf:.0%}  stab={stab:.2f}{flags}\n",
                    tag,
                )

                if st != "ok":
                    reasons = item.get("reasons", [])
                    if reasons:
                        self._write(f"      ⚑ {reasons[0][:72]}\n", "dim")

        # ── Final status bar message ──────────────────────────────────────────
        if not needs_review and total_n > 0:
            msg = "Done — all clips matched confidently."
        elif forced_n:
            msg = f"Done with warnings — {forced_n} FORCED clip(s) need review before delivery."
        else:
            msg = "Done — some clips flagged. Check the report before delivery."

        if not has_aaf:
            msg += "  (AAF not generated — WAV available.)"

        if needs_review:
            msg += "  Use 'Review Flagged Clips' to sign off."

        self._set_status(msg)

        # ── Re-enable controls ────────────────────────────────────────────────
        self._btn.config(state="normal", bg=ACCENT, fg="white", text="PROCESS")
        self._reset_btn.config(state="normal")
        self._open_btn.config(state="normal", bg="#3a6a4a", fg="white")

        if needs_review:
            self._review_btn.config(
                state="normal",
                bg="#5a3a7a",   # distinct purple — draws attention
                fg="white",
                text="✎  Review Flagged Clips",
            )
        else:
            self._review_btn.config(
                state="disabled", bg="#444", fg="#aaa",
                text="✎  Review Flagged Clips",
            )

    def _show_error(self, msg: str) -> None:
        self._processing = False
        self._write("\n── PROCESSING ERROR ──────────────────────────\n", "fail")
        self._write(msg + "\n", "fail")
        self._set_status("Processing failed — see error above.")
        self._btn.config(state="normal", bg=ACCENT, fg="white", text="PROCESS")
        self._reset_btn.config(state="normal")

    # ── Action buttons ────────────────────────────────────────────────────────

    def _open_output_folder(self) -> None:
        if not self._last_run_dir or not self._last_run_dir.exists():
            self._set_status("No output folder to open yet.")
            return
        target = str(self._last_run_dir)
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", target])
            elif sys.platform.startswith("win"):
                subprocess.Popen(["explorer", target])
            else:
                subprocess.Popen(["xdg-open", target])
        except Exception as exc:
            self._set_status(f"Could not open folder: {exc}")

    def _open_review_window(self) -> None:
        if not self._last_run_dir:
            return
        flagged = [
            c for c in self._report_data
            if c.get("status", "ok").lower() != "ok"
        ]
        if not flagged:
            self._set_status("No flagged clips to review.")
            return
        ReviewWindow(self, self._last_run_dir, flagged)

    def _reset(self) -> None:
        if self._processing:
            return
        self._ref_wav.clear()
        self._vobu_wav.clear()
        self._aaf.clear()
        self._out_folder.clear()
        self._clear_results()
        self._set_status(
            "Select AAF Full Bounce, VO BU, and AAF to enable processing."
        )
        self._last_run_dir = None
        self._total_clips  = 0
        self._report_data  = []
        self._open_btn.config(state="disabled", bg="#444", fg="#aaa")
        self._review_btn.config(state="disabled", bg="#444", fg="#aaa")
        self._update_process_btn()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _write(self, text: str, tag: str = "") -> None:
        self._results.config(state="normal")
        if tag:
            self._results.insert("end", text, tag)
        else:
            self._results.insert("end", text)
        self._results.config(state="disabled")
        self._results.see("end")

    def _clear_results(self) -> None:
        self._results.config(state="normal")
        self._results.delete("1.0", "end")
        self._results.config(state="disabled")

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
