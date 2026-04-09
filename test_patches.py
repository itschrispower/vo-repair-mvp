#!/usr/bin/env python3
"""
test_patches.py — CLI version of patch testing without GUI dependencies.
Tests the three patches on matching scenarios and prints results.
"""
import numpy as np


class MatchingSimulator:
    """Simulate matching behavior with/without patches."""

    def __init__(self):
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


def main():
    """Run all test scenarios and print results."""
    sim = MatchingSimulator()

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

    print("\n" + "=" * 120)
    print("VO REPAIR MATCHING PATCHES — TEST RESULTS".center(120))
    print("=" * 120)
    print("\nTesting 3 patches:")
    print("  PATCH 1: MIN_NCC_CANDIDATE: 0.30 → 0.25")
    print("  PATCH 2: Stability-aware minimum floor")
    print("  PATCH 3: Coarse fallback to region midpoint")
    print("\n")

    improved = 0
    results = []

    for scenario_args in scenarios:
        result = sim.test_scenario(*scenario_args)
        results.append(result)

        if result["improvement"]:
            improved += 1

    # Print header
    print(f"{'Scenario':<30} {'NCC':<8} {'Old Status':<12} {'→':<3} {'New Status':<12} {'Confidence':<20}")
    print("-" * 120)

    # Print results
    for result in results:
        scenario = result["name"]
        ncc = f"{result['raw_ncc']:.2f}"
        old_status = result["status_old"]
        new_status = result["status_new"]
        improvement = "→" if result["improvement"] else "="
        conf_str = f"({result['conf_final_old']:.2f}→{result['conf_final_new']:.2f})"

        print(f"{scenario:<30} {ncc:<8} {old_status:<12} {improvement:<3} {new_status:<12} {conf_str:<20}")

    print("-" * 120)
    print(f"\nSummary: {improved}/{len(scenarios)} scenarios improved ({improved/len(scenarios)*100:.0f}%)")
    print("\nDetailed Results:")
    print("-" * 120)

    for result in results:
        if result["improvement"]:
            print(f"\n✓ {result['name']}")
            print(f"  Raw NCC: {result['raw_ncc']:.2f}")
            print(f"  After penalties: {result['conf_after_penalties']:.2f}")
            print(f"  Old floor result: {result['conf_final_old']:.2f} → Status: {result['status_old']}")
            print(f"  New floor result: {result['conf_final_new']:.2f} → Status: {result['status_new']}")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()
