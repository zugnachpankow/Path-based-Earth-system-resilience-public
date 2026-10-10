"""32_tables_fig4.py — full results tables behind Fig. 4 (Extended Data / SI).

Table A (panel a): current-policies branch-off grid (start year × decarbonisation
  rate) -> path-based resilience [%] with 95% multi-level crossed-bootstrap CI (from
  09, as in Table 1). Where the bootstrap collapses at zero, the one-sided 97.5%
  Clopper-Pearson upper bound (n = configs) is shown instead, marked '*'.
  Plus marginal means ± SD, the start-year linear fit (pp/yr, pp per 5 yr, R²),
  and the decarb-rate linear fit reported ONLY as a diagnostic (the rate marginal
  is non-linear / saturating).

Table B (panels b, c): one row per target (+1, +10 pp) × conditioning (all 841
  configs; GFP 10/30/50/70/90th). Columns: GFP target value, window mean GFP,
  n configs, R [% below NDC baseline 2035, all emissions], CO2 cut in 2035
  [GtCO2/yr], NDC baseline CO2 in 2035 [GtCO2/yr], baseline resilience [%],
  confirmed gain [pp], 95% CI [pp].

Reads stored outputs only (no reruns). Light — runs on login.
Output -> output/tables/fig4_tableA_branchoffs.{csv,md}, fig4_tableB_targets.{csv,md}
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.stats.proportion import proportion_confint

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from calculator import compute_fragility_masks
from fair_config import FAIR_PARAMS
from figures_common import time_to_net_zero_yr

OUT_DIR = os.path.join("tables"); os.makedirs(OUT_DIR, exist_ok=True)

SUMMARY_CSV = ("output/resilience/resilience_conditions_time2100_temp1.5_rate04/"
               "summary/resilience_summary_time2100_temp1.5_rate04.csv")
CONF_1   = "output/calculator/confirmator/confirmed_results_gain0.01.csv"
CONF_10  = "output/calculator/confirmator/confirmed_results_gain0.1.csv"
CONF_GFP = "output/calculator/confirmator/confirmed_gfp_percentile.csv"
EMISSIONS_FILE = "data/processed/NGFS_historic_merged_extended_to_2110.csv"
PARAMS_FILE    = FAIR_PARAMS   # single calibration (v1.4.0) via src/fair_config
NDC_SCENARIO   = "REMIND-MAgPIE_3.3-4.8___Nationally_Determined_Contributions_NDCs"

N_CP = len(pd.read_csv(PARAMS_FILE, index_col=0))   # 841 configs = effective independent n


def cp_ci(p, n=N_CP):
    """Clopper-Pearson 95% CI [%] for resilience p (fraction). Returns (lo, hi, flag)
    where flag='*' marks a one-sided bound at a saturated cell (p~0 or p~1)."""
    k = int(round(p * n))
    lo, hi = proportion_confint(k, n, alpha=0.05, method="beta")
    lo = 0.0 if np.isnan(lo) else lo
    hi = 1.0 if np.isnan(hi) else hi
    flag = "*" if (k == 0 or k == n) else ""
    return lo * 100, hi * 100, flag


# ── Table A: branch-off grid ────────────────────────────────────────────────────
# one-sided 97.5% Clopper-Pearson bounds (k=0 / k=n); the "<…" / ">…" label threshold
BOUND_LO = 100.0 * (1.0 - 0.025 ** (1.0 / N_CP))   # ≈ 0.4377  -> "<0.44"
BOUND_HI = 100.0 * 0.025 ** (1.0 / N_CP)           # ≈ 99.5623 -> ">99.56"


def _cell_label(v):
    """Figure-consistent cell string from the UNROUNDED value v [%]."""
    if v > BOUND_HI:
        return f">{BOUND_HI:.2f}"
    if v < BOUND_LO:
        return f"<{BOUND_LO:.2f}"
    return f"{v:.1f}"


df = pd.read_csv(SUMMARY_CSV)
recs = []
for _, row in df.iterrows():
    m = re.search(r"start(\d+\.?\d*)_r(\d+\.?\d*)", row["scenario"])
    if not m:
        continue
    sy = int(float(m.group(1))); rt = int(round(float(m.group(2)) * 100))
    p = float(row["resilience"])
    res = p * 100.0                                   # UNROUNDED resilience [%]
    # Table-1 convention: crossed-bootstrap CI (from 09); fall back to the one-sided
    # 97.5% Clopper-Pearson upper bound only where the bootstrap collapses at zero.
    boot_lo = float(row["resilience_ci_low"]) * 100
    boot_hi = float(row["resilience_ci_high"]) * 100
    if boot_lo == 0.0 and boot_hi == 0.0:
        cp_lo, cp_hi, _ = cp_ci(p)
        lo, hi, method = cp_lo, cp_hi, "CP"
    else:
        lo, hi, method = boot_lo, boot_hi, "bootstrap"
    recs.append({"start_year": sy, "rate_pct_per_yr": rt, "resilience": res,
                 "ci_low": lo, "ci_high": hi, "ci_method": method})
A = pd.DataFrame(recs).sort_values(["start_year", "rate_pct_per_yr"]).reset_index(drop=True)

# CSV: single-round to one decimal at output (CI upper kept at 2 dp so the CP 0.44 shows)
A_csv = pd.DataFrame({
    "start_year": A["start_year"], "rate_pct_per_yr": A["rate_pct_per_yr"],
    "resilience_pct": A["resilience"].round(1),
    "cell_label": [_cell_label(v) for v in A["resilience"]],
    "ci_low_pct": A["ci_low"].round(2), "ci_high_pct": A["ci_high"].round(2),
    "ci_method": A["ci_method"],
})
A_csv.to_csv(os.path.join(OUT_DIR, "fig4_tableA_branchoffs.csv"), index=False)

# marginals + fits — all from UNROUNDED values, rounded once at output
pivot = A.pivot(index="start_year", columns="rate_pct_per_yr", values="resilience").sort_index()
RATES = sorted(A["rate_pct_per_yr"].unique())
START_YEARS = sorted(A["start_year"].unique())
row_mean = pivot.mean(axis=1); row_sd = pivot.std(axis=1, ddof=1)      # per start year, over rates
col_mean = pivot.mean(axis=0); col_sd = pivot.std(axis=0, ddof=1)      # per rate, over start years
s = linregress(START_YEARS, row_mean.values)
r = linregress(RATES, col_mean.values)
marg = pd.concat([
    pd.DataFrame({"marginal": "start_year", "value": START_YEARS,
                  "mean": row_mean.round(1).values, "std": row_sd.round(1).values}),
    pd.DataFrame({"marginal": "decarb_rate", "value": RATES,
                  "mean": col_mean.round(1).values, "std": col_sd.round(1).values}),
], ignore_index=True)
marg.to_csv(os.path.join(OUT_DIR, "fig4_tableA_marginals.csv"), index=False)
fits = pd.DataFrame([
    {"fit": "resilience vs start-year", "slope": round(s.slope, 4), "unit": "pp/yr",
     "per_5yr_earlier_pp": round(-s.slope * 5, 2), "r2": round(s.rvalue**2, 4), "note": "primary"},
    {"fit": "resilience vs decarb-rate", "slope": round(r.slope, 4), "unit": "pp per %/yr",
     "per_5yr_earlier_pp": np.nan, "r2": round(r.rvalue**2, 4),
     "note": "DIAGNOSTIC ONLY - marginal is non-linear/saturating"},
])
fits.to_csv(os.path.join(OUT_DIR, "fig4_tableA_fits.csv"), index=False)

# ── LaTeX (booktabs grid) ────────────────────────────────────────────────────────
TIME_TO_ZERO = [time_to_net_zero_yr(rt) for rt in RATES]   # generated once (shared with 17)


def _tex_cell(v):
    lab = _cell_label(v)
    return "$<$" + lab[1:] if lab.startswith("<") else ("$>$" + lab[1:] if lab.startswith(">") else lab)


L = []
L.append(r"\footnotesize")
L.append(r"\setlength{\tabcolsep}{3.5pt}")
L.append(r"\begin{tabular}{@{}lrrrrrrrrrr|rr@{}}")
L.append(r"& \multicolumn{10}{c|}{Decarbonisation rate [\%\,yr$^{-1}$]} & & \\")
L.append("Start year & " + " & ".join(str(rt) for rt in RATES) + r" & Mean & SD \\")
L.append(r"\cmidrule(r){2-11}")
L.append(r"\textit{Time to zero [yr]} & " +
         " & ".join(f"\\textit{{{t}}}" for t in TIME_TO_ZERO) + r" & & \\")
L.append(r"\midrule")
for sy in START_YEARS:
    cells = " & ".join(_tex_cell(pivot.loc[sy, rt]) for rt in RATES)
    L.append(f"{sy} & {cells} & {row_mean[sy]:.1f} & {row_sd[sy]:.1f} " + r"\\")
L.append(r"\midrule")
L.append("Mean & " + " & ".join(f"{col_mean[rt]:.1f}" for rt in RATES) + r" & & \\")
L.append("SD & " + " & ".join(f"{col_sd[rt]:.1f}" for rt in RATES) + r" & & \\")
L.append(r"\botrule")
L.append(r"\end{tabular}")
with open(os.path.join(OUT_DIR, "fig4_tableA_branchoffs.tex"), "w") as fh:
    fh.write("\n".join(L) + "\n")

# ── Markdown (long, with CIs) ────────────────────────────────────────────────────
def _ci_str(lo, hi, mth):
    return (f"[{lo:.1f}, {hi:.2f}]" if mth == "CP" else f"[{lo:.1f}, {hi:.1f}]")


A_md = pd.DataFrame({
    "Start year": A["start_year"],
    "Rate [%/yr]": A["rate_pct_per_yr"],
    "Resilience [%]": [_cell_label(v) for v in A["resilience"]],
    "95% CI [%]": [_ci_str(lo, hi, mth) for lo, hi, mth in zip(A["ci_low"], A["ci_high"], A["ci_method"])],
})
with open(os.path.join(OUT_DIR, "fig4_tableA_branchoffs.md"), "w") as fh:
    fh.write("## Table A — path-based resilience over the current-policies branch-off grid\n\n")
    fh.write("Start year × decarbonisation rate. 95% CI from the multi-level crossed bootstrap "
             f"(from 09); where the bootstrap collapses at zero, the one-sided 97.5% Clopper–Pearson "
             f"upper bound (n = {N_CP} configs) is shown instead. `<0.44` = unrounded estimate below "
             "that bound.\n\n")
    fh.write(A_md.to_markdown(index=False) + "\n\n")
    fh.write("### Marginal means ± SD (from unrounded values)\n\n")
    fh.write(marg.to_markdown(index=False) + "\n\n")
    fh.write("### Linear fits\n\n")
    fh.write(f"- resilience vs **start-year**: {s.slope:.3f} pp/yr "
             f"({-s.slope*5:.2f} pp per 5 yr earlier), R² = {s.rvalue**2:.4f}\n")
    fh.write(f"- resilience vs **decarb-rate**: {r.slope:.3f} pp per %/yr, R² = {r.rvalue**2:.4f} "
             "(diagnostic only — the rate marginal is non-linear/saturating)\n")

print(f"[Table A] {len(A)} branch-offs | start-year {s.slope:.3f} pp/yr "
      f"({-s.slope*5:.2f} pp/5yr, R2={s.rvalue**2:.4f}) | "
      f"rate {r.slope:.3f} pp per %/yr R2={r.rvalue**2:.4f} (diagnostic)", flush=True)


# ── Table B: targets × conditioning ─────────────────────────────────────────────
def _ndc_co2_2035():
    em = pd.read_csv(EMISSIONS_FILE)
    m = ((em["scenario"].str.replace("(", "", regex=False).str.replace(")", "", regex=False) == NDC_SCENARIO)
         & (em["region"] == "World") & (em["variable"].isin(["CO2 FFI", "CO2 AFOLU"])))
    return float(em.loc[m, "2035.5"].sum()) / 1000.0


ndc_co2_2035 = _ndc_co2_2035()

# window mean GFP per percentile (PCA; deterministic)
cfg = pd.read_csv(PARAMS_FILE, index_col=0)
_, gf = compute_fragility_masks(cfg.index.tolist(), PARAMS_FILE)
HALFWIDTH = 0.10
win_mean = {}
for pct in [10, 30, 50, 70, 90]:
    t = np.percentile(gf.values, pct)
    mask = (gf >= t - HALFWIDTH) & (gf <= t + HALFWIDTH)
    win_mean[pct] = float(gf[mask].mean())

rows = []
for gain, conf_csv in [(0.01, CONF_1), (0.10, CONF_10)]:
    cr = pd.read_csv(conf_csv)
    rr = cr[cr["target_year"].round(1) == 2035.5].iloc[0]
    base = float(rr["baseline_resilience"])
    rows.append({
        "target_pp": int(gain * 100), "conditioning": "all configs",
        "gfp_target": np.nan, "window_mean_gfp": np.nan, "n_configs": N_CP,
        "R_pct": round(float(rr["reduction_frac"]) * 100, 1),
        "co2_cut_2035_GtCO2_yr": round(float(rr["co2_GtCO2"]) / 1000.0, 2),
        "ndc_co2_2035_GtCO2_yr": round(ndc_co2_2035, 2),
        "baseline_resilience_pct": round(base * 100, 3),
        "gain_pp": round(float(rr["achieved_gain"]) * 100, 2),
        "resilience_after_pct": round(float(rr["achieved_resilience"]) * 100, 2),
        "gain_ci_low_pct": round(float(rr["ci_low"]) * 100, 2),
        "gain_ci_high_pct": round(float(rr["ci_high"]) * 100, 2),
    })

gfp = pd.read_csv(CONF_GFP)
for gain in [0.01, 0.10]:
    sub = gfp[gfp["gain_target"] == gain].sort_values("percentile")
    for _, rr in sub.iterrows():
        base = float(rr["baseline_resilience"]); pct = int(rr["percentile"])
        rows.append({
            "target_pp": int(gain * 100), "conditioning": f"GFP {pct}th",
            "gfp_target": round(float(rr["gf_value"]), 4),
            "window_mean_gfp": round(win_mean[pct], 4),
            "n_configs": int(rr["n_configs"]),
            "R_pct": round(float(rr["reduction_frac"]) * 100, 1),
            "co2_cut_2035_GtCO2_yr": round(float(rr["co2_GtCO2"]) / 1000.0, 2),
            "ndc_co2_2035_GtCO2_yr": round(ndc_co2_2035, 2),
            "baseline_resilience_pct": round(base * 100, 3),
            "gain_pp": round(float(rr["achieved_gain"]) * 100, 2),
            "resilience_after_pct": round(float(rr["achieved_resilience"]) * 100, 2),
            "gain_ci_low_pct": round(float(rr["ci_low"]) * 100, 2),
            "gain_ci_high_pct": round(float(rr["ci_high"]) * 100, 2),
        })

B = pd.DataFrame(rows)
B.to_csv(os.path.join(OUT_DIR, "fig4_tableB_targets.csv"), index=False)

_gain_ci = [f"{g:.2f} [{lo:.2f}, {hi:.2f}]" for g, lo, hi in
            zip(B["gain_pp"], B["gain_ci_low_pct"], B["gain_ci_high_pct"])]
B_md = pd.DataFrame({
    "Target [pp]": B["target_pp"],
    "Conditioning": B["conditioning"],
    "GFP": [("—" if np.isnan(v) else f"{v:.3f}") for v in B["gfp_target"]],
    "mean GFP": [("—" if np.isnan(v) else f"{v:.3f}") for v in B["window_mean_gfp"]],
    "n": B["n_configs"],
    "R [%]": [f"{v:.1f}" for v in B["R_pct"]],
    "CO2 cut 2035 [GtCO2/yr]": [f"{v:.2f}" for v in B["co2_cut_2035_GtCO2_yr"]],
    "NDC CO2 2035 [GtCO2/yr]": [f"{v:.2f}" for v in B["ndc_co2_2035_GtCO2_yr"]],
    "Baseline res. [%]": [f"{v:.3f}" for v in B["baseline_resilience_pct"]],
    "Gain [pp] (95 % CI)": _gain_ci,
    "Resilience after cut [%]": [f"{v:.2f}" for v in B["resilience_after_pct"]],
})
with open(os.path.join(OUT_DIR, "fig4_tableB_targets.md"), "w") as fh:
    fh.write("## Table B — required reduction and confirmed resilience gain (Fig. 4b,c)\n\n")
    fh.write("R = additional reduction below the NDC baseline in 2035 (all emissions, same fraction "
             "per species). Gain = confirmed path-based resilience gain (point estimate); its 95 % CI "
             "is the crossed-bootstrap CI of the PAIRED candidate-minus-NDC difference (ci_low/ci_high, "
             "from the same run). Resilience after cut = baseline + gain (point estimate).\n\n")
    fh.write(B_md.to_markdown(index=False) + "\n")

# LaTeX (booktabs) — Table B layout; CI column = PAIRED candidate-minus-NDC gain CI.
# Header order/alignment must match B_md.columns above.
_hdr = ["Target [pp]", "Conditioning", "GFP", "mean GFP", "$n$", "$R$ [\\%]",
        "CO$_2$ cut 2035 [GtCO$_2$\\,yr$^{-1}$]", "NDC CO$_2$ 2035 [GtCO$_2$\\,yr$^{-1}$]",
        "Baseline res.\\ [\\%]", "Gain [pp] (95\\,\\% CI)", "Resilience after cut [\\%]"]
_Ltx = [r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}llrrrrrrrlr@{}}", r"\toprule",
        " & ".join(_hdr) + r" \\", r"\midrule"]
for _, rr in B_md.iterrows():
    vals = [str(rr[c]).replace("—", "--") for c in B_md.columns]
    _Ltx.append(" & ".join(vals) + r" \\")
_Ltx += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(OUT_DIR, "fig4_tableB_targets.tex"), "w") as fh:
    fh.write("\n".join(_Ltx) + "\n")

print(f"[Table B] {len(B)} rows | NDC CO2 2035 = {ndc_co2_2035:.2f} GtCO2/yr", flush=True)
print("\nTable B:\n", B.to_string(index=False), flush=True)
