import sys
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext
from pathlib import Path

import run_job


# ── colours & fonts ──────────────────────────────────────────────────────────
BG       = "#1e1e1e"
PANEL    = "#2a2a2a"
ACCENT   = "#4a9eff"
OK_COL   = "#4caf50"
REVIEW   = "#ff9800"
FAIL_COL = "#f44336"
FG       = "#e0e0e0"
FG_DIM   = "#888888"
FONT     = ("Helvetica Neue", 13)
FONT_SM  = ("Helvetica Neue", 11)
MONO     = ("Menlo", 11)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VO Repair")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.geometry("620x520")

        self._job_path = tk.StringVar(value="")
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=20, pady=10)

        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", **pad)
        tk.Label(hdr, text="VO Repair", font=("Helvetica Neue", 22, "bold"),
                 bg=BG, fg=FG).pack(side="left")

        # Job folder picker
        picker = tk.Frame(self, bg=PANEL, bd=0, highlightthickness=1,
                          highlightbackground="#444")
        picker.pack(fill="x", padx=20, pady=(0, 10))

        tk.Label(picker, text="Job folder", font=FONT_SM,
                 bg=PANEL, fg=FG_DIM).pack(anchor="w", padx=12, pady=(10, 0))

        row = tk.Frame(picker, bg=PANEL)
        row.pack(fill="x", padx=12, pady=(4, 10))

        self._path_label = tk.Label(row, textvariable=self._job_path,
                                    font=FONT_SM, bg=PANEL, fg=FG,
                                    anchor="w", width=48, relief="flat")
        self._path_label.pack(side="left")

        tk.Button(row, text="Choose…", font=FONT_SM, bg="#333", fg=FG,
                  relief="flat", padx=10, cursor="hand2",
                  command=self._pick_folder).pack(side="right")

        # Process button
        self._btn = tk.Button(self, text="Process", font=("Helvetica Neue", 14, "bold"),
                              bg=ACCENT, fg="white", relief="flat",
                              padx=20, pady=8, cursor="hand2",
                              command=self._run)
        self._btn.pack(pady=(0, 14))

        # Status label
        self._status = tk.StringVar(value="Choose a job folder to begin.")
        tk.Label(self, textvariable=self._status, font=FONT_SM,
                 bg=BG, fg=FG_DIM).pack()

        # Results area
        self._results = scrolledtext.ScrolledText(
            self, font=MONO, bg=PANEL, fg=FG,
            relief="flat", padx=12, pady=12,
            state="disabled", height=16, width=72,
            insertbackground=FG
        )
        self._results.pack(padx=20, pady=(10, 20), fill="both", expand=True)

        # Tag colours for result lines
        self._results.tag_config("ok",     foreground=OK_COL)
        self._results.tag_config("review", foreground=REVIEW)
        self._results.tag_config("fail",   foreground=FAIL_COL)
        self._results.tag_config("head",   foreground=ACCENT,
                                 font=("Helvetica Neue", 12, "bold"))
        self._results.tag_config("dim",    foreground=FG_DIM)

    # ── actions ──────────────────────────────────────────────────────────────

    def _pick_folder(self):
        path = filedialog.askdirectory(title="Select VO job folder")
        if path:
            self._job_path.set(path)
            self._clear_results()
            self._set_status("Ready — click Process.")

    def _run(self):
        path = self._job_path.get().strip()
        if not path:
            self._set_status("Please choose a job folder first.")
            return

        self._btn.config(state="disabled")
        self._clear_results()
        self._set_status("Processing…")
        threading.Thread(target=self._process, args=(path,), daemon=True).start()

    def _process(self, path):
        try:
            run_job.process_job(Path(path))
            self.after(0, self._load_results, Path(path))
        except Exception as exc:
            self.after(0, self._show_error, str(exc))

    # ── result display ────────────────────────────────────────────────────────

    def _load_results(self, job_path: Path):
        import json

        # Find the report JSON written by engine.py
        rebuild_dir = job_path / "rebuild_audio"
        reports = sorted(rebuild_dir.glob("*_report.json")) if rebuild_dir.exists() else []

        # Fallback: look in OUTPUT
        if not reports:
            out_dir = job_path / "OUTPUT"
            reports = sorted(out_dir.glob("*_report.json")) if out_dir.exists() else []

        self._clear_results()
        self._write("RESULTS\n", "head")

        if reports:
            with open(reports[0]) as f:
                data = json.load(f)

            counts = {"ok": 0, "review": 0, "fail": 0}
            for item in data:
                status = item.get("status", "?").lower()
                counts[status] = counts.get(status, 0) + 1
                name  = item.get("clip_name", "?")
                conf  = item.get("confidence", 0)
                start = item.get("timeline_start_tc", "?")
                end   = item.get("timeline_end_tc", "?")
                tag   = status if status in ("ok", "review", "fail") else "dim"
                icon  = {"ok": "✓", "review": "~", "fail": "✗"}.get(status, "?")
                line  = f"  {icon}  Clip {name}   {start} → {end}   confidence {conf:.0%}\n"
                self._write(line, tag)

            self._write("\n", "dim")
            self._write(
                f"  ✓ {counts['ok']} OK   "
                f"~ {counts['review']} review   "
                f"✗ {counts['fail']} failed\n",
                "dim"
            )
            failed = counts["fail"] > 0
            self._set_status("Done — check results below." if not failed
                             else "Finished with failures — see results below.")
        else:
            # Fall back to summary .txt if JSON not found
            summaries = sorted(job_path.rglob("*_summary.txt"))
            if summaries:
                self._write(summaries[0].read_text(encoding="utf-8"), "dim")
            else:
                self._write("Job completed — no detailed report found.\n", "dim")
            self._set_status("Done.")

        self._btn.config(state="normal")

    def _show_error(self, msg: str):
        self._write(f"ERROR\n\n{msg}\n", "fail")
        self._set_status("Processing failed — see error above.")
        self._btn.config(state="normal")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _write(self, text: str, tag: str = ""):
        self._results.config(state="normal")
        if tag:
            self._results.insert("end", text, tag)
        else:
            self._results.insert("end", text)
        self._results.config(state="disabled")
        self._results.see("end")

    def _clear_results(self):
        self._results.config(state="normal")
        self._results.delete("1.0", "end")
        self._results.config(state="disabled")

    def _set_status(self, msg: str):
        self._status.set(msg)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
