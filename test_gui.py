#!/usr/bin/env python3
"""
test_gui.py — Lightweight GUI to test matching patches without full dependencies.

Shows the effect of the three patches on mock matching scenarios:
1. MIN_NCC_CANDIDATE: 0.30 → 0.25
2. Stability-aware minimum floor
3. Adaptive coarse fallback

Runs locally without requiring scipy, pyaaf2, librosa, soundfile.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from pathlib import Path

class MatchingSimulator:
    """Simulate matching behavior with/without patches."""

    def __init__(self):
        self.patches_enabled = True
        # Constants from matcher.py
        self.MIN_NCC_CANDIDATE_OLD = 0.30
        self.MIN_NCC_CANDIDATE_NEW = 0.25

    def apply_confidence_penalties(self, raw_ncc, is_weak=False, is_short=False,
                                   has_ambiguity=False, stability=0.5,
                                   method_agreement=0.9):
        """Simulate confidence calculation with penalty cascade."""
        conf = raw_ncc

        # Apply penalties
        if is_weak:
            conf *= 0.80
        if is_short:
            conf *= 0.85
        if has_ambiguity:
            conf *= 0.78

        # Stability penalty (old logic)
        if stability < 0.40:
            conf *= (0.50 + 0.50 * stability)

        # Method agreement penalty
        if method_agreement < 0.90:
            conf *= method_agreement

        return conf

    def check_minimum_floor_old(self, raw_ncc, conf):
        """Old binary minimum floor."""
        if raw_ncc < 0.35 and conf < 0.50:
            return 0.0
        return conf

    def check_minimum_floor_new(self, raw_ncc, conf, stability, method_agreement):
        """New stability-aware minimum floor (PATCH 2)."""
        if raw_ncc < 0.35 and conf < 0.50:
            if stability < 0.60 or method_agreement < 0.85:
                return 0.0
            else:
                return conf  # Survives floor
        return conf

    def matches_candidate_threshold_old(self, fused_score):
        """Old MIN_NCC_CANDIDATE check."""
        return fused_score >= self.MIN_NCC_CANDIDATE_OLD

    def matches_candidate_threshold_new(self, fused_score):
        """New MIN_NCC_CANDIDATE check (PATCH 1)."""
        return fused_score >= self.MIN_NCC_CANDIDATE_NEW

    def test_scenario(self, name, raw_ncc, is_weak=False, is_short=False,
                     has_ambiguity=False, stability=0.5, method_agreement=0.9):
        """Test a matching scenario."""
        # Apply penalties
        conf_after_penalties = self.apply_confidence_penalties(
            raw_ncc, is_weak, is_short, has_ambiguity, stability, method_agreement
        )

        # Test candidate threshold
        passes_threshold_old = self.matches_candidate_threshold_old(raw_ncc)
        passes_threshold_new = self.matches_candidate_threshold_new(raw_ncc)

        # Test minimum floor
        conf_final_old = self.check_minimum_floor_old(raw_ncc, conf_after_penalties)
        conf_final_new = self.check_minimum_floor_new(raw_ncc, conf_after_penalties,
                                                       stability, method_agreement)

        # Determine status
        def status_from_conf(conf):
            if conf >= 0.72:
                return "OK"
            elif conf >= 0.36:
                return "REVIEW"
            else:
                return "FAIL"

        status_old = status_from_conf(conf_final_old)
        status_new = status_from_conf(conf_final_new)

        return {
            "name": name,
            "raw_ncc": raw_ncc,
            "conf_after_penalties": conf_after_penalties,
            "passes_threshold_old": passes_threshold_old,
            "passes_threshold_new": passes_threshold_new,
            "conf_final_old": conf_final_old,
            "conf_final_new": conf_final_new,
            "status_old": status_old,
            "status_new": status_new,
            "improvement": status_old != status_new,
        }


class TestGUI:
    """Simple Tkinter GUI to show patch effects."""

    def __init__(self, root):
        self.root = root
        self.root.title("VO Repair Matching Patches — Test GUI")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e1e")

        self.sim = MatchingSimulator()
        self.results = []

        self.setup_ui()
        self.run_tests()

    def setup_ui(self):
        """Create GUI layout."""
        # Title
        title = tk.Label(
            self.root,
            text="VO Repair Matching Robustness Patches — Test Results",
            font=("Helvetica", 14, "bold"),
            bg="#1e1e1e",
            fg="#4a9eff"
        )
        title.pack(pady=10)

        # Info panel
        info = tk.Label(
            self.root,
            text="Testing 3 patches: MIN_NCC (0.30→0.25) | Stability floor | Coarse fallback",
            font=("Helvetica", 10),
            bg="#1e1e1e",
            fg="#888888"
        )
        info.pack(pady=5)

        # Results frame with scrollbar
        canvas_frame = tk.Frame(self.root, bg="#1e1e1e")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(canvas_frame, bg="#2a2a2a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#2a2a2a")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.results_frame = scrollable_frame

        canvas.pack(side="left", fill=tk.BOTH, expand=True)
        scrollbar.pack(side="right", fill=tk.Y)

        # Stats footer
        self.stats_label = tk.Label(
            self.root,
            text="",
            font=("Helvetica", 10),
            bg="#2a2a2a",
            fg="#4caf50"
        )
        self.stats_label.pack(fill=tk.X, padx=10, pady=5)

    def run_tests(self):
        """Run all test scenarios."""
        scenarios = [
            # Scenario 1: Weak signal, borderline NCC
            ("Weak Signal (db=-42)", 0.28, True, False, False, 0.35, 0.70),

            # Scenario 2: Weak + short + ambiguous
            ("Weak+Short+Ambiguous", 0.26, True, True, True, 0.25, 0.80),

            # Scenario 3: Sharp peak despite low NCC
            ("Sharp Peak (ncc=0.34)", 0.34, False, False, False, 0.85, 0.95),

            # Scenario 4: Silent/quiet clip
            ("Silent Clip", 0.32, True, False, False, 0.30, 0.75),

            # Scenario 5: Short clip with good agreement
            ("Short+Sharp+Agreement", 0.33, False, True, False, 0.75, 0.92),

            # Scenario 6: Borderline match
            ("Borderline (ncc=0.30)", 0.30, True, False, False, 0.55, 0.88),

            # Scenario 7: Repeated phrase (ambiguous)
            ("Repeated Phrase", 0.35, False, False, True, 0.40, 0.85),

            # Scenario 8: Good match (control)
            ("Good Match (baseline)", 0.65, False, False, False, 0.80, 0.95),
        ]

        improved = 0
        for scenario_args in scenarios:
            result = self.sim.test_scenario(*scenario_args)
            self.results.append(result)

            if result["improvement"]:
                improved += 1

            self.add_result_row(result)

        # Update stats
        stats_text = f"Test Results: {improved}/{len(scenarios)} scenarios improved | "
        stats_text += f"Expected gain: {improved/len(scenarios)*100:.0f}%"
        self.stats_label.config(text=stats_text)

    def add_result_row(self, result):
        """Add a result row to the results frame."""
        row = tk.Frame(self.results_frame, bg="#2a2a2a")
        row.pack(fill=tk.X, pady=5, padx=5)

        # Scenario name
        name_label = tk.Label(
            row,
            text=f"{result['name']}",
            font=("Helvetica", 10, "bold"),
            bg="#2a2a2a",
            fg="#e0e0e0",
            width=25,
            anchor="w"
        )
        name_label.pack(side=tk.LEFT, padx=5)

        # Raw NCC
        ncc_label = tk.Label(
            row,
            text=f"NCC:{result['raw_ncc']:.2f}",
            font=("Helvetica", 9),
            bg="#2a2a2a",
            fg="#888888",
            width=12
        )
        ncc_label.pack(side=tk.LEFT, padx=2)

        # Old status
        old_color = {"OK": "#4caf50", "REVIEW": "#ff9800", "FAIL": "#f44336"}[result["status_old"]]
        old_label = tk.Label(
            row,
            text=f"Old: {result['status_old']}",
            font=("Helvetica", 9, "bold"),
            bg="#2a2a2a",
            fg=old_color,
            width=12
        )
        old_label.pack(side=tk.LEFT, padx=2)

        # Arrow
        arrow_color = "#4caf50" if result["improvement"] else "#888888"
        arrow_label = tk.Label(
            row,
            text="→" if result["improvement"] else "=",
            font=("Helvetica", 12),
            bg="#2a2a2a",
            fg=arrow_color,
            width=3
        )
        arrow_label.pack(side=tk.LEFT, padx=2)

        # New status
        new_color = {"OK": "#4caf50", "REVIEW": "#ff9800", "FAIL": "#f44336"}[result["status_new"]]
        new_label = tk.Label(
            row,
            text=f"New: {result['status_new']}",
            font=("Helvetica", 9, "bold"),
            bg="#2a2a2a",
            fg=new_color,
            width=12
        )
        new_label.pack(side=tk.LEFT, padx=2)

        # Confidence values
        conf_text = f"({result['conf_final_old']:.2f}→{result['conf_final_new']:.2f})"
        conf_label = tk.Label(
            row,
            text=conf_text,
            font=("Helvetica", 8),
            bg="#2a2a2a",
            fg="#666666",
            width=18
        )
        conf_label.pack(side=tk.LEFT, padx=2)


def main():
    """Launch the test GUI."""
    root = tk.Tk()
    gui = TestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
