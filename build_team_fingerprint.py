import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
INPUT = DATA_DIR / "team_matches_clean.csv"

OUT_FINGERPRINT = DATA_DIR / "wc26_final_team_fingerprint_handoff.csv"
OUT_DEFINITIONS = DATA_DIR / "wc26_final_facet_definitions.csv"
OUT_STABILITY = DATA_DIR / "wc26_final_stability_summary.csv"
OUT_LOO = DATA_DIR / "wc26_final_leave_one_match_out.csv"

FACETS = [
    "Control",
    "Access",
    "Line-breaking",
    "Defensive approach",
    "Attacking approach",
    "Chance threat",
]


def zscore(series):
    sd = series.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean(skipna=True)) / sd


def build_match_level_facets(df):
    opp_passes = df[["match_id", "team", "passes"]].rename(
        columns={"team": "opponent", "passes": "opp_passes"}
    )
    m = df.merge(opp_passes, on=["match_id", "opponent"], how="left")

    m["z_possession"] = zscore(m["possession_pct"])
    m["z_pass_completion"] = zscore(m["pass_completion_pct"])
    m["Control"] = m[["z_possession", "z_pass_completion"]].mean(axis=1)

    access_cols = [
        "receptions_final_third",
        "offers_made_final_third",
        "offers_made_inside_shape",
    ]
    for col in access_cols:
        m[f"z_{col}"] = zscore(m[col])
    m["Access"] = m[[f"z_{c}" for c in access_cols]].mean(axis=1)

    m["linebreaking_raw"] = (
        100 * m["defensive_line_breaks"] / m["passes"].replace(0, np.nan)
    )
    m["Line-breaking"] = zscore(m["linebreaking_raw"])

    m["pressures_per100_opp_passes"] = (
        100 * m["def_pressures"] / m["opp_passes"].replace(0, np.nan)
    )

    front_foot = [
        "phase_high_press_pct",
        "phase_high_block_pct",
        "phase_counter_press_pct",
        "pressures_per100_opp_passes",
    ]
    deep_block = [
        "phase_low_press_pct",
        "phase_low_block_pct",
    ]

    for col in front_foot + deep_block:
        m[f"z_{col}"] = zscore(m[col])

    m["inverse_low_block_height"] = -m["oop_low_block_line_height_m"]
    m["z_inverse_low_block_height"] = zscore(m["inverse_low_block_height"])

    front_score = m[[f"z_{c}" for c in front_foot]].mean(axis=1)
    deep_score = m[
        [f"z_{c}" for c in deep_block] + ["z_inverse_low_block_height"]
    ].mean(axis=1)
    m["Defensive approach"] = front_score - deep_score

    transition_cols = [
        "phase_attacking_transition_pct",
        "phase_counter_attack_pct",
    ]
    for col in transition_cols:
        m[f"z_{col}"] = zscore(m[col])

    transition_score = m[[f"z_{c}" for c in transition_cols]].mean(axis=1)
    m["Attacking approach"] = m["Control"] - transition_score

    m["Chance threat"] = zscore(m["xg"])

    return m


def build_team_scores(match_facets, raw):
    team_scores = match_facets.groupby("team")[FACETS].mean()

    opp = raw[["match_id", "team", "xg", "goals"]].rename(
        columns={
            "team": "opponent",
            "xg": "opp_xg",
            "goals": "opp_goals",
        }
    )
    r = raw.merge(opp, on=["match_id", "opponent"], how="left")

    outcomes = r.groupby("team").agg(
        matches=("match_id", "nunique"),
        xg_for_per_match=("xg", "mean"),
        xg_against_per_match=("opp_xg", "mean"),
        goals_per_match=("goals", "mean"),
    )

    outcomes["xg_diff_per_match"] = (
        outcomes["xg_for_per_match"] - outcomes["xg_against_per_match"]
    )
    outcomes["goals_minus_xg_per_match"] = (
        outcomes["goals_per_match"] - outcomes["xg_for_per_match"]
    )

    final = team_scores.join(outcomes)

    for facet in FACETS:
        final[f"{facet}_pctile"] = 100 * final[facet].rank(pct=True)

    return final


def leave_one_match_out(match_facets, final):
    rows = []

    for team, g in match_facets.groupby("team"):
        match_ids = g["match_id"].dropna().unique()
        if len(match_ids) < 2:
            continue

        full_scores = g[FACETS].mean()

        for left_out in match_ids:
            h = g[g["match_id"] != left_out]
            row = {"team": team, "left_out_match": left_out}

            for facet in FACETS:
                score = h[facet].mean()
                row[facet] = score
                row[f"{facet}_delta"] = score - full_scores[facet]

            rows.append(row)

    loo = pd.DataFrame(rows)

    distributions = {
        facet: final[facet].dropna().values
        for facet in FACETS
    }

    stability_rows = []

    for team, g in loo.groupby("team"):
        row = {"team": team}
        widths = []
        max_shifts = []

        for facet in FACETS:
            dist = distributions[facet]
            pcts = [
                100 * np.mean(dist <= value)
                for value in g[facet]
            ]

            low = float(np.min(pcts))
            high = float(np.max(pcts))
            width = high - low

            row[f"{facet}_pctile_low"] = low
            row[f"{facet}_pctile_high"] = high
            row[f"{facet}_pctile_width"] = width
            row[f"{facet}_loo_sd"] = float(g[facet].std(ddof=0))
            row[f"{facet}_max_abs_delta"] = float(
                g[f"{facet}_delta"].abs().max()
            )

            widths.append(width)
            max_shifts.append(row[f"{facet}_max_abs_delta"])

        row["mean_pctile_uncertainty_width"] = float(np.mean(widths))
        row["max_pctile_uncertainty_width"] = float(np.max(widths))
        row["max_score_shift_leave_one_out"] = float(np.max(max_shifts))

        stability_rows.append(row)

    stability = pd.DataFrame(stability_rows).set_index("team")
    stability["stability_flag"] = np.select(
        [
            stability["max_pctile_uncertainty_width"] <= 15,
            stability["max_pctile_uncertainty_width"] <= 30,
        ],
        ["Stable", "Moderate"],
        default="Sample-sensitive",
    )

    return loo, stability


def write_definitions():
    definitions = pd.DataFrame(
        [
            [
                "Control",
                "Magnitude",
                "Possession %; Pass completion %",
                "Higher = more ball dominance and circulation security",
            ],
            [
                "Access",
                "Magnitude",
                "Final-third receptions; Final-third offers; Offers inside opposition shape",
                "Higher = more advanced availability and reception",
            ],
            [
                "Line-breaking",
                "Magnitude",
                "Defensive-line breaks per 100 passes",
                "Higher = more deep penetration through the defensive line",
            ],
            [
                "Chance threat",
                "Magnitude",
                "xG per match",
                "Higher = greater underlying chance threat",
            ],
            [
                "Attacking approach",
                "Bipolar style axis",
                "Control composite minus transition-attack composite",
                "Higher = more control-oriented; lower = more transition-oriented",
            ],
            [
                "Defensive approach",
                "Bipolar style axis",
                "Front-foot pressing/high-block composite minus deep-block composite",
                "Higher = more front-foot; lower = more deep-block oriented",
            ],
        ],
        columns=["facet", "type", "headline_inputs", "direction"],
    )
    definitions.to_csv(OUT_DEFINITIONS, index=False)


def main():
    raw = pd.read_csv(INPUT)

    match_facets = build_match_level_facets(raw)
    final = build_team_scores(match_facets, raw)
    loo, stability = leave_one_match_out(match_facets, final)

    final = final.join(stability, how="left").reset_index()

    final.to_csv(OUT_FINGERPRINT, index=False)
    stability.reset_index().to_csv(OUT_STABILITY, index=False)
    loo.to_csv(OUT_LOO, index=False)
    write_definitions()

    print(f"Wrote {OUT_FINGERPRINT}")
    print(f"Wrote {OUT_DEFINITIONS}")
    print(f"Wrote {OUT_STABILITY}")
    print(f"Wrote {OUT_LOO}")


if __name__ == "__main__":
    main()
