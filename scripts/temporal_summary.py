"""Generate temporal summary plots for available training embeddings."""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

# Add repo root to path so `rol` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from surge.analysis import LakeAnalyzer

ROLE_LAKES = {
    "Source": ["Finger Lake","Long Lake","Spirit Lake","South Rolly Lake",
               "Tern Lake","Walby Lake","Wik Lake","Watson Lake"],
    "Recipient": ["CC Lake","Crystal Lake","Fred Lake","Hope Lake","Leisure Lake",
                  "Leisure Pond","Loon Lake","Ranchero Lake"],
}
LAKE_ROLES = {}
for role, lakes in ROLE_LAKES.items():
    for l in lakes:
        LAKE_ROLES[l] = role


class MockData:
    def get_lake_role(self, lake): return LAKE_ROLES.get(lake, "Other")
    def get_lake_ecotype(self, lake): return "Unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Temporal divergence summary plots")
    parser.add_argument('--output-dir', required=True,
                        help='Pipeline output directory')
    args = parser.parse_args()

    # Panels: (strat, suffix, label).  year_lake gets one lake-level panel;
    # sex/infection get one panel PER status, since the per-lake temporal
    # series is only well-defined within a single status (suffixed keys
    # 'Lake (2021)-f' / '-m' collide at (lake, year) if merged).
    panels = (
        [("year_lake", None, "year_lake")]
        + [(s, suf, f"{s} ({suf.lstrip('-')})")
           for s, sufs in (("sex_year_lake", ["-f", "-m"]),
                           ("infection_year_lake", ["-0", "-1"]))
           for suf in sufs]
    )
    colors = {"Source": "#2196F3", "Recipient": "#F44336"}
    jitter = 0.08
    rng = np.random.default_rng(42)

    # Auto-detect available embeddings (joint, sequential, or both)
    candidates = [
        ("joint", "embeddings_joint.pkl", "temporal_summary_joint.png"),
        ("sequential", "embeddings.pkl", "temporal_summary_sequential.png"),
    ]

    for paradigm, emb_file, out_name in candidates:
        emb_path = os.path.join(args.output_dir, emb_file)
        if not os.path.exists(emb_path):
            print(f"  Skipping {paradigm} — {emb_path} not found")
            continue

        emb = pickle.load(open(emb_path, "rb"))
        # Load the matching codebook (saved by the pipeline) so the
        # Wasserstein OT uses the codebook cosine ground metric.
        cb_path = os.path.join(args.output_dir,
                               emb_file.replace('embeddings', 'codebook'))
        codebook = (pickle.load(open(cb_path, "rb"))
                    if os.path.exists(cb_path) else None)
        analyzer = LakeAnalyzer(emb, data=MockData(), codebook=codebook)

        def _draw_panel(ax, strat, suffix, label):
            """Draw one Source/Recipient endpoint panel for a (strat, suffix)."""
            sub = analyzer.for_stratification(strat)
            # wasserstein_temporal raises for suffixed-only stratifications
            # without a suffix (sex/infection), whose per-lake temporal series
            # is undefined when 'Lake (2021)-f' and 'Lake (2021)-m' collide per
            # (lake, year).  With a suffix it builds the per-status series.
            try:
                wass = sub.wasserstein_temporal(base_year=2019, suffix=suffix)
            except ValueError as exc:
                print(f"  [temporal_summary] '{label}' skipped: {exc}")
                ax.set_title(f"{label}\n(skipped)", fontsize=8)
                return
            wass_scaled, p95 = LakeAnalyzer.scale_wasserstein_p95(wass)

            src_vals, rec_vals = [], []
            src_names, rec_names = [], []
            unclassified = []
            last_years = []
            for lake, dists in wass_scaled.items():
                role = LAKE_ROLES.get(lake)
                if role is None:
                    # Loudly report lakes missing from the hardcoded 16-lake
                    # map instead of silently dropping them from the plot.
                    unclassified.append(lake)
                    continue
                endpoint = dists[-1][1]
                last_years.append(dists[-1][0])
                if role == "Source":
                    src_vals.append(endpoint)
                    src_names.append(lake)
                elif role == "Recipient":
                    rec_vals.append(endpoint)
                    rec_names.append(lake)
            if unclassified:
                print(f"  [temporal_summary] {label}: {len(unclassified)} "
                      f"lake(s) not in the hardcoded role map and dropped: "
                      f"{', '.join(sorted(unclassified))}")
            if last_years:
                distinct_last = sorted(set(last_years))
                if len(distinct_last) > 1:
                    print(f"  [temporal_summary] {label}: WARNING — lakes were "
                          f"last sampled in DIFFERENT years "
                          f"({distinct_last}); endpoint divergence compares "
                          f"lakes at unequal time horizons.")

            # Collect all labelled points, then place with de-conflicted offsets
            labelled = []  # (x, val, label, side)
            for i, (val, name) in enumerate(zip(src_vals, src_names)):
                x = rng.uniform(-jitter, jitter)
                ax.scatter(x, val, color=colors["Source"], s=60, zorder=5,
                           edgecolors="white", linewidth=0.5)
                if val > 0.9 or val < 0.35:
                    labelled.append((x, val,
                                     name.replace(" Lake","").replace(" Pond",""),
                                     'right'))
            for i, (val, name) in enumerate(zip(rec_vals, rec_names)):
                x = 1 + rng.uniform(-jitter, jitter)
                ax.scatter(x, val, color=colors["Recipient"], s=60, zorder=5,
                           edgecolors="white", linewidth=0.5)
                if val > 0.9:
                    labelled.append((x, val,
                                     name.replace(" Lake","").replace(" Pond",""),
                                     'right'))

            # De-conflict labels: sort by y, stagger offsets for close neighbours
            labelled.sort(key=lambda t: t[1])
            offsets = [0, -6, 6, -10, 10, -14, 14]  # cycling stagger distances
            for i, (x, val, lab, side) in enumerate(labelled):
                # Check how many previous labels are close in y
                n_nearby = sum(1 for j in range(i)
                               if abs(labelled[j][1] - val) < 0.03)
                y_off = offsets[min(n_nearby, len(offsets) - 1)]
                ax.annotate(lab, (x, val), fontsize=6, alpha=0.8,
                            xytext=(15, y_off), textcoords="offset points",
                            va='center')

            for idx, vals in [(0, src_vals), (1, rec_vals)]:
                if vals:
                    mean = np.mean(vals)
                    ax.plot([idx - 0.2, idx + 0.2], [mean, mean],
                            color="black", linewidth=2.5, zorder=10)

            if len(src_vals) >= 2 and len(rec_vals) >= 2:
                t_s, t_p = ttest_ind(src_vals, rec_vals)
                mw_s, mw_p = mannwhitneyu(src_vals, rec_vals)
                pooled_sd = np.sqrt(
                    (np.std(src_vals, ddof=1)**2 +
                     np.std(rec_vals, ddof=1)**2) / 2)
                # Degenerate groups (all-equal values → pooled SD == 0) make
                # Cohen's d undefined; report NaN rather than 0/0 or inf.
                d = ((np.mean(src_vals) - np.mean(rec_vals)) / pooled_sd
                     if pooled_sd > 0 else float('nan'))
                ax.set_title(
                    f"{label}\nd = {d:.2f},  "
                    f"t-test p = {t_p:.3f},  MW p = {mw_p:.3f}", fontsize=8)
            else:
                ax.set_title(label, fontsize=9)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Source", "Recipient"])
            ax.set_ylabel("P95-scaled Wasserstein distance\nfrom baseline")
            ax.set_ylim(bottom=-0.05)
            ax.grid(axis="y", alpha=0.3)

        fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * 4.8, 5))
        for ax, (strat, suffix, label) in zip(axes, panels):
            _draw_panel(ax, strat, suffix, label)

        fig.suptitle(
            f"Temporal Divergence by Experimental Role — {paradigm} training",
            fontsize=13, fontweight="bold")
        fig.tight_layout()

        out_dir = os.path.join(args.output_dir, 'postprocess')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

        # Print stats
        for strat, suffix, label in panels:
            sub = analyzer.for_stratification(strat)
            wass = sub.wasserstein_temporal_or_skip(base_year=2019, suffix=suffix)
            wass_scaled, _ = LakeAnalyzer.scale_wasserstein_p95(wass)
            sv, rv = [], []
            for lake, dists in wass_scaled.items():
                role = LAKE_ROLES.get(lake, "Other")
                ep = dists[-1][1]
                if role == "Source": sv.append(ep)
                elif role == "Recipient": rv.append(ep)
            if len(sv) >= 2 and len(rv) >= 2:
                t_s, t_p = ttest_ind(sv, rv)
                mw_s, mw_p = mannwhitneyu(sv, rv)
                pooled_sd = np.sqrt(
                    (np.std(sv, ddof=1)**2 + np.std(rv, ddof=1)**2) / 2)
                d = ((np.mean(sv) - np.mean(rv)) / pooled_sd
                     if pooled_sd > 0 else float('nan'))
                print(f"    {label}: S={np.mean(sv):.3f} vs R={np.mean(rv):.3f}, "
                      f"d={d:.2f}, t-test p={t_p:.3f}, MW p={mw_p:.3f}")
            else:
                print(f"    {label}: insufficient data")


if __name__ == "__main__":
    main()
