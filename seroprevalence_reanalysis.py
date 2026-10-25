"""
Supplementary reanalysis of the antibody pattern counts

Source: Tables 3 and 4 of the BSc project "Seroprevalence of SARS-CoV-2 among
vaccinated people in Asubiaro General Hospital and Federal Medical Centre,
Owo" (Redeemer's University, 2024). Only the aggregate counts from those
tables are used. The participant-level optical density (OD) values behind the
project's figures are not included here.

Why this reanalysis exists
The project reports percentages out of all 68 participants (for example,
"females 42.65% vs males 22.06% positive for both antibodies"). Because 48 of
the 68 participants were female, those figures mostly reflect group size.
Here every percentage is calculated within its own group (for example, the
share of females with both antibodies out of all females), with 95% Wilson
confidence intervals and Fisher exact tests.

Antibody patterns
  SRBD only : anti-spike antibodies only, consistent with a vaccine response
  N only    : anti-nucleocapsid antibodies only
  Both      : anti-spike and anti-nucleocapsid antibodies. The N antigen is not
              in the vaccines used in Nigeria, so N antibodies point to past
              infection in addition to vaccination.

All tests are exploratory, uncorrected for multiple testing, and the groups
are small.

Run:  python seroprevalence_reanalysis.py
Needs: numpy, pandas, scipy, matplotlib
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import odds_ratio

OUT = Path(__file__).parent
df = pd.read_csv(OUT / "antibody_pattern_counts.csv")
results = []


def wilson(k, n):
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return ci.low * 100, ci.high * 100


def counts(table, grouping):
    """Participants by group and antibody pattern, pooled over location."""
    s = df[(df.source_table == table) & (df.grouping == grouping)]
    return s.pivot_table(index="group", columns="antibody_pattern", values="participants", aggfunc="sum").fillna(0).astype(int)


def two_group_test(name, label_a, a, label_b, b, outcome_name):
    """Fisher exact test comparing the share of an outcome in two groups. a and b are (successes, total)."""
    table = [[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]]
    _, p = stats.fisher_exact(table)
    ci = odds_ratio(table).confidence_interval(0.95)
    or_ = odds_ratio(table).statistic
    results.append({
        "outcome": outcome_name, "comparison": name,
        f"group_1": f"{label_a}: {a[0]}/{a[1]} ({a[0] / a[1] * 100:.1f}%)",
        f"group_2": f"{label_b}: {b[0]}/{b[1]} ({b[0] / b[1] * 100:.1f}%)",
        "odds_ratio": round(or_, 2), "or_95ci": f"{ci.low:.2f} to {ci.high:.2f}", "fisher_p": round(p, 4),
    })
    print(f"   {outcome_name}, {name}: {label_a} {a[0] / a[1] * 100:.1f}% vs {label_b} {b[0] / b[1] * 100:.1f}%, "
          f"OR {or_:.2f} (95% CI {ci.low:.2f} to {ci.high:.2f}), Fisher p = {p:.3f}")


def cochran_armitage(successes, totals, scores):
    """Test for a linear trend in proportions across ordered groups."""
    x, n = np.asarray(successes, float), np.asarray(totals, float)
    s = np.asarray(scores, float)
    p_bar = x.sum() / n.sum()
    t = np.sum(s * (x - n * p_bar))
    var = p_bar * (1 - p_bar) * (np.sum(n * s**2) - np.sum(n * s) ** 2 / n.sum())
    z = t / np.sqrt(var)
    return z, 2 * stats.norm.sf(abs(z))


# 1. Overall -------------------------------------------------------------
overall = df[df.source_table == "Table 3"].groupby("antibody_pattern").participants.sum()
N = int(overall.sum())
print(f"Participants: {N}")
print("\n1. OVERALL ANTIBODY PATTERNS")
for pat in ["SRBD only", "N only", "Both"]:
    lo, hi = wilson(overall[pat], N)
    print(f"   {pat}: {overall[pat]}/{N} = {overall[pat] / N * 100:.1f}% (95% CI {lo:.1f} to {hi:.1f})")
    results.append({"outcome": "Overall pattern", "comparison": pat, "group_1": f"{overall[pat]}/{N} ({overall[pat] / N * 100:.1f}%)",
                    "group_2": "", "odds_ratio": "", "or_95ci": f"95% CI {lo:.1f} to {hi:.1f}", "fisher_p": ""})
any_n = int(overall["N only"] + overall["Both"])
any_s = int(overall["SRBD only"] + overall["Both"])
for label, k in [("Any N antibodies (past infection signal)", any_n), ("Any SRBD antibodies", any_s)]:
    lo, hi = wilson(k, N)
    print(f"   {label}: {k}/{N} = {k / N * 100:.1f}% (95% CI {lo:.1f} to {hi:.1f})")
    results.append({"outcome": "Overall pattern", "comparison": label, "group_1": f"{k}/{N} ({k / N * 100:.1f}%)",
                    "group_2": "", "odds_ratio": "", "or_95ci": f"95% CI {lo:.1f} to {hi:.1f}", "fisher_p": ""})

# 2. By location ---------------------------------------------------------
print("\n2. BY LOCATION (Asubiaro vs Owo)")
loc = df[df.source_table == "Table 3"].pivot_table(index="location", columns="antibody_pattern", values="participants", aggfunc="sum")
loc["total"] = loc.sum(axis=1)
loc["Any N"] = loc["N only"] + loc["Both"]
for outcome, col in [("Both antibodies", "Both"), ("Any N antibodies", "Any N"), ("Anti-spike only", "SRBD only")]:
    two_group_test("Asubiaro vs Owo", "Asubiaro", (loc.loc["Asubiaro", col], loc.loc["Asubiaro", "total"]),
                   "Owo", (loc.loc["Owo", col], loc.loc["Owo", "total"]), outcome)

# 3. By sex --------------------------------------------------------------
print("\n3. BY SEX (males vs females)")
sex = counts("Table 3", "Sex")
sex["total"] = sex.sum(axis=1)
sex["Any N"] = sex["N only"] + sex["Both"]
for outcome, col in [("Both antibodies", "Both"), ("Any N antibodies", "Any N"), ("Anti-spike only", "SRBD only")]:
    two_group_test("Male vs Female", "Male", (sex.loc["Male", col], sex.loc["Male", "total"]),
                   "Female", (sex.loc["Female", col], sex.loc["Female", "total"]), outcome)

# 4. By age --------------------------------------------------------------
print("\n4. BY AGE GROUP (any N antibodies)")
age = counts("Table 4", "Age group").loc[["Young (18-29)", "Middle-aged (30-50)", "Older (over 50)"]]
age["total"] = age.sum(axis=1)
age["Any N"] = age["N only"] + age["Both"]
for g, r in age.iterrows():
    lo, hi = wilson(r["Any N"], r["total"])
    print(f"   {g}: {r['Any N']}/{r['total']} = {r['Any N'] / r['total'] * 100:.1f}% (95% CI {lo:.1f} to {hi:.1f})")
    results.append({"outcome": "Any N antibodies by age", "comparison": g, "group_1": f"{r['Any N']}/{r['total']} ({r['Any N'] / r['total'] * 100:.1f}%)",
                    "group_2": "", "odds_ratio": "", "or_95ci": f"95% CI {lo:.1f} to {hi:.1f}", "fisher_p": ""})
older, rest = age.loc["Older (over 50)"], age.drop("Older (over 50)").sum()
two_group_test("Older vs younger groups combined", "Older", (older["Any N"], older["total"]),
               "Younger groups", (rest["Any N"], rest["total"]), "Any N antibodies")
z, p = cochran_armitage(age["Any N"], age["total"], [0, 1, 2])
print(f"   Trend across age groups (Cochran-Armitage): z = {z:.2f}, p = {p:.3f}")
results.append({"outcome": "Any N antibodies by age", "comparison": "Linear trend across age groups", "group_1": f"z = {z:.2f}",
                "group_2": "", "odds_ratio": "", "or_95ci": "", "fisher_p": round(p, 4)})

pd.DataFrame(results).to_csv(OUT / "reanalysis_results.csv", index=False)


# Figure -----------------------------------------------------------------
def share_table(frame, order):
    t = frame.loc[order, ["SRBD only", "N only", "Both"]]
    return t.div(t.sum(axis=1), axis=0) * 100, t.sum(axis=1)


colors = {"SRBD only": "#0e7490", "N only": "#f2b544", "Both": "#c2410c"}
panels = [
    ("By location", share_table(loc, ["Asubiaro", "Owo"])),
    ("By sex", share_table(sex, ["Female", "Male"])),
    ("By age group", share_table(age, ["Young (18-29)", "Middle-aged (30-50)", "Older (over 50)"])),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2, 2, 3]})
for ax, (title, (shares, totals)) in zip(axes, panels):
    left = np.zeros(len(shares))
    for pat in ["SRBD only", "N only", "Both"]:
        ax.barh(range(len(shares)), shares[pat], left=left, color=colors[pat], label=pat if title == "By location" else None)
        for i, (v, l) in enumerate(zip(shares[pat], left)):
            if v >= 8:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=10)
        left += shares[pat].to_numpy()
    ax.set_yticks(range(len(shares)))
    ax.set_yticklabels([f"{g}\n(n = {int(totals[g])})" for g in shares.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent of the group")
    ax.set_title(title)
fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Antibody pattern within each group (percentages calculated within the group)", y=1.0)
fig.tight_layout(rect=(0, 0.05, 1, 0.98))
fig.savefig(OUT / "antibody_patterns.png", dpi=160, bbox_inches="tight")
print("\nSaved: reanalysis_results.csv, antibody_patterns.png")
