from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUT_DIR = Path("analysis/output")

PLAYERS = DATA_DIR / "wc26_player_atlas_120min_FINAL_BACKEND_GEMINI_AUDITED_606.csv"
NEIGHBORS = DATA_DIR / "wc26_player_similarity_120min_606_neighbors.csv"

PROFILE_VARS = [
    "distribution_pct",
    "line_breaking_pct",
    "carrying_pct",
    "movement_activity_pct",
    "physical_intensity_pct",
    "front_foot_defending_pct",
    "pressing_activity_pct",
    "deep_defending_pct",
    "aerial_presence_pct",
    "passer_carrier_axis_pct",
    "space_movement_axis_pct",
    "advanced_role_axis_pct",
    "wide_role_axis_pct",
]

OUTCOME_VARS = [
    "attacking_threat_pct",
    "defending_pct",
    "shots_p90_position_pct",
    "goals_p90_position_pct",
]


def role_counts(players):
    return (
        players.groupby(["traditional_home", "function_archetype"], as_index=False)
        .size()
        .rename(columns={"size": "players"})
        .sort_values(["traditional_home", "players"], ascending=[True, False])
    )


def role_profiles(players):
    rows = []

    for role, group in players.groupby("function_archetype"):
        row = {
            "function_archetype": role,
            "traditional_home": group["traditional_home"].iloc[0],
            "players": len(group),
        }

        for var in PROFILE_VARS:
            row[f"{var}_q25"] = group[var].quantile(0.25)
            row[f"{var}_median"] = group[var].median()
            row[f"{var}_q75"] = group[var].quantile(0.75)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["traditional_home", "function_archetype"])


def position_mix(players):
    mix = (
        players.groupby(["function_archetype", "position_canonical"])
        .size()
        .rename("players")
        .reset_index()
    )

    totals = mix.groupby("function_archetype")["players"].transform("sum")
    mix["share"] = mix["players"] / totals

    return mix.sort_values(
        ["function_archetype", "players"],
        ascending=[True, False],
    )


def role_outcomes(players):
    return (
        players.groupby("function_archetype")[OUTCOME_VARS]
        .median()
        .reset_index()
        .sort_values("function_archetype")
    )


def validate(players, neighbors):
    assert len(players) == 606
    assert players["function_archetype"].notna().all()
    assert players["function_archetype"].nunique() == 17
    assert players[["team", "player_number"]].drop_duplicates().shape[0] == 606
    assert len(neighbors) == 606 * 15
    assert neighbors.groupby(["team", "player_name"]).size().eq(15).all()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    players = pd.read_csv(PLAYERS)
    neighbors = pd.read_csv(NEIGHBORS)

    validate(players, neighbors)

    role_counts(players).to_csv(OUT_DIR / "role_counts.csv", index=False)
    role_profiles(players).to_csv(OUT_DIR / "role_profiles.csv", index=False)
    position_mix(players).to_csv(OUT_DIR / "role_position_mix.csv", index=False)
    role_outcomes(players).to_csv(OUT_DIR / "role_outcomes.csv", index=False)


if __name__ == "__main__":
    main()
