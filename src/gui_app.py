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


class FilePicker(tk.Frame):
    """A labelled row with a path display and a Choose button."""

    def __init__(self, parent, label, mode="file", filetypes=None, **kwargs):
        super().__init__(parent, bg=PANEL, **kwargs)
        self._mode = mode
        self._filetypes = filetypes or []
        self._path = tk.StringVar()

        tk.Label(self, text=label, font=FONT_SM, bg=PANEL,
                 fg=FG_DIM, width=16, anchor="w").pack(side="left", padx=(12, 4))

        tk.Label(self, textvariable=self._path, font=FONT_SM,
                 bg=PANEL, fg=FG, anchor="w", width=38,
                 relief="flat").pack(side="left")

        tk.Button(self, text="Choose…", font=FONT_SM, bg="#333", fg=FG,
                  relief="flat", padx=10, cursor="hand2",
                  command=self._pick).pack(side="right", padx=12, pady=6)

    def _pick(self):
        if self._mode == "folder":
            path = filedialog.askdirectory(title="Select output folder")
        else:
            path = filedialog.askopenfilename(
                title=f"Select file",
                filetypes=self._filetypes or [("All files", "*.*")]
            )
        if path:
            self._path.set(path)

    def get(self):
        return self._path.get().strip()

    def is_set(self):
        return bool(self.get())


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VO Repair")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.geometry("640x560")
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        tk.Label(self, text="VO Repair",
                 font=("Helvetica Neue", 22, "bold"),
                 bg=BG, fg=FG).pack(anchor="w", padx=20, pady=(16, 10))

        # File pickers panel
        panel = tk.Frame(self, bg=PANEL, bd=0,
                         highlightthickness=1, highlightbackground="#444")
        panel.pack(fill="x", padx=20, pady=(0, 12))

        self._aaf = FilePicker(
            panel, "AAF File",
            filetypes=[("AAF files", "*.aaf"), ("All files", "*.*")]
        )
        self._aaf.pack(fill="x")
        self._sep(panel)

        self._ref_wav = FilePicker(
            panel, "Reference WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        self._ref_wav.pack(fill="x")
        self._sep(panel)

        self._vobu_wav = FilePicker(
            panel, "VO Backup WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        self._vobu_wav.pack(fill="x")
        self._sep(panel)

        self._out_folder = FilePicker(panel, "Output Folder", mode="folder")
        self._out_folder.pack(fill="x")

        # Process button
        self._btn = tk.Button(
            self, text="Process",
            font=("Helvetica Neue", 14, "bold"),
            bg=ACCENT, fg="white", relief="flat",
            padx=24, pady=8, cursor="hand2",
            command=self._run
        )
        self._btn.pack(pady=(4, 6))

        # Status
        self._status = tk.StringVar(value="Choose your files then click Process.")
        tk.Label(self, textvariable=self._status, font=FONT_SM,
                 bg=BG, fg=FG_DIM).pack()

        # Results area
        self._results = scrolledtext.ScrolledText(
            self, font=MONO, bg=PANEL, fg=FG,
            relief="flat", padx=12, pady=12,
            state="disabled", height=14, width=74,
            insertbackground=FG
        )
        self._results.pack(padx=20, pady=(10, 20), fill="both", expand=True)

        self._results.tag_config("ok",     foreground=OK_COL)
        self._results.tag_config("review", foreground=REVIEW)
        self._results.tag_config("fail",   foreground=FAIL_COL)
        self._results.tag_config("head",   foreground=ACCENT,
                                 font=("Helvetica Neue", 12, "bold"))
        self._results.tag_config("dim",    foreground=FG_DIM)

    def _sep(self, parent):
        tk.Frame(parent, bg="#444", height=1).pack(fill="x")

    # ── actions ──────────────────────────────────────────────────────────────

    def _run(self):
        missing = []
        if not self._aaf.is_set():       missing.append("AAF File")
        if not self._ref_wav.is_set():   missing.append("Reference WAV")
        if not self._vobu_wav.is_set():  missing.append("VO Backup WAV")
        if not self._out_folder.is_set(): missing.append("Output Folder")

        if missing:
            self._set_status(f"Please choose: {', '.join(missing)}")
            return

        self._btn.config(state="disabled")
        self._clear_results()
        self._set_status("Processing…")

        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        try:
            out = run_job.process_files(
                aaf_path=Path(self._aaf.get()),
                ref_wav_path=Path(self._ref_wav.get()),
                vobu_wav_path=Path(self._vobu_wav.get()),
                output_dir=Path(self._out_folder.get()),
            )
            self.after(0, self._load_results, Path(self._out_folder.get()))
        except Exception as exc:
            self.after(0, self._show_error, str(exc))

    # ── result display ────────────────────────────────────────────────────────

    def _load_results(self, output_dir: Path):
        import json

        reports = sorted(output_dir.glob("*_report.json"))

        self._clear_results()
        self._write("RESULTS\n\n", "head")

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
                self._write(
                    f"  {icon}  Clip {name}   {start} → {end}   "
                    f"confidence {conf:.0%}\n",
                    tag
                )

            self._write("\n", "dim")
            self._write(
                f"  ✓ {counts['ok']} OK     "
                f"~ {counts['review']} review     "
                f"✗ {counts['fail']} failed\n",
                "dim"
            )
            failed = counts["fail"] > 0
            self._set_status(
                "Done — check results below." if not failed
                else "Finished with failures — see results below."
            )
        else:
            # Plain text fallback
            summaries = sorted(output_dir.glob("*_summary.txt"))
            if summaries:
                self._write(summaries[0].read_text(encoding="utf-8"), "dim")
            else:
                self._write("Job completed.\n", "dim")
            self._set_status("Done.")

        self._btn.config(state="normal")

    def _show_error(self, msg: str):
        self._write(f"ERROR\n\n{msg}\n", "fail")
        self._set_status("Processing failed — see error above.")
        self._btn.config(state="normal")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _write(self, text, tag=""):
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

    def _set_status(self, msg):
        self._status.set(msg)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
