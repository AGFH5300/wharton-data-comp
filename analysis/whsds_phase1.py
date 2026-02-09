#!/usr/bin/env python3
"""WHL Phase 1 data science pipeline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42


@dataclass
class StrengthResult:
    strengths: pd.Series
    model_name: str
    cv_log_loss: float
    cv_brier: float


def load_data(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "whl_2025.csv")
    return df


def build_game_level(df: pd.DataFrame) -> pd.DataFrame:
    rows_per_game = df["game_id"].value_counts()
    print("Rows per game_id summary:")
    print(rows_per_game.describe())

    agg_dict = {
        "home_xg": "sum",
        "away_xg": "sum",
        "home_goals": "sum",
        "away_goals": "sum",
        "home_shots": "sum",
        "away_shots": "sum",
    }

    game_df = (
        df.groupby("game_id", as_index=False)
        .agg({
            **agg_dict,
            "home_team": "first",
            "away_team": "first",
        })
    )

    inconsistent_home = (
        df.groupby("game_id")["home_team"].nunique().loc[lambda s: s > 1]
    )
    inconsistent_away = (
        df.groupby("game_id")["away_team"].nunique().loc[lambda s: s > 1]
    )
    if not inconsistent_home.empty or not inconsistent_away.empty:
        raise ValueError("Inconsistent home/away teams within game_id.")

    game_df["home_win"] = (game_df["home_goals"] > game_df["away_goals"]).astype(int)

    print(f"Game-level rows: {len(game_df)}")
    return game_df


def compute_team_xg_strength(game_df: pd.DataFrame) -> pd.Series:
    home_stats = game_df[["home_team", "home_xg", "away_xg"]].rename(
        columns={"home_team": "team", "home_xg": "xg_for", "away_xg": "xg_against"}
    )
    away_stats = game_df[["away_team", "away_xg", "home_xg"]].rename(
        columns={"away_team": "team", "away_xg": "xg_for", "home_xg": "xg_against"}
    )
    combined = pd.concat([home_stats, away_stats], ignore_index=True)
    team_totals = combined.groupby("team", as_index=True).sum()
    games_played = combined.groupby("team").size()
    strength = (team_totals["xg_for"] - team_totals["xg_against"]) / games_played
    strength.name = "strength_score"
    return strength


def compute_bradley_terry_strength(game_df: pd.DataFrame) -> pd.Series:
    teams = pd.Index(sorted(pd.unique(game_df[["home_team", "away_team"]].values.ravel())))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}

    n_games = len(game_df)
    n_teams = len(teams)
    x = np.zeros((n_games, n_teams))
    for i, (home, away) in enumerate(zip(game_df["home_team"], game_df["away_team"])):
        x[i, team_to_idx[home]] = 1
        x[i, team_to_idx[away]] = -1

    y = game_df["home_win"].values

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    model.fit(x, y)
    strengths = pd.Series(model.coef_[0], index=teams, name="strength_score")
    return strengths


def evaluate_strength_method(
    game_df: pd.DataFrame, method: str
) -> Tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    log_losses = []
    briers = []

    for train_idx, test_idx in skf.split(game_df, game_df["home_win"]):
        train_df = game_df.iloc[train_idx]
        test_df = game_df.iloc[test_idx]

        if method == "xg_diff":
            strengths = compute_team_xg_strength(train_df)
        elif method == "bradley_terry":
            strengths = compute_bradley_terry_strength(train_df)
        else:
            raise ValueError(f"Unknown method {method}")

        strength_diff_train = (
            train_df["home_team"].map(strengths) - train_df["away_team"].map(strengths)
        ).values.reshape(-1, 1)
        strength_diff_test = (
            test_df["home_team"].map(strengths) - test_df["away_team"].map(strengths)
        ).values.reshape(-1, 1)

        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        model.fit(strength_diff_train, train_df["home_win"].values)
        preds = model.predict_proba(strength_diff_test)[:, 1]
        log_losses.append(log_loss(test_df["home_win"].values, preds))
        briers.append(brier_score_loss(test_df["home_win"].values, preds))

    return float(np.mean(log_losses)), float(np.mean(briers))


def select_strength_method(game_df: pd.DataFrame) -> StrengthResult:
    xg_log_loss, xg_brier = evaluate_strength_method(game_df, "xg_diff")
    bt_log_loss, bt_brier = evaluate_strength_method(game_df, "bradley_terry")

    print(f"CV log loss (xG diff): {xg_log_loss:.4f}, Brier: {xg_brier:.4f}")
    print(f"CV log loss (Bradley-Terry): {bt_log_loss:.4f}, Brier: {bt_brier:.4f}")

    if bt_log_loss < xg_log_loss:
        strengths = compute_bradley_terry_strength(game_df)
        return StrengthResult(
            strengths=strengths,
            model_name="bradley_terry",
            cv_log_loss=bt_log_loss,
            cv_brier=bt_brier,
        )
    strengths = compute_team_xg_strength(game_df)
    return StrengthResult(
        strengths=strengths,
        model_name="xg_diff",
        cv_log_loss=xg_log_loss,
        cv_brier=xg_brier,
    )


def build_team_rankings(game_df: pd.DataFrame, strengths: pd.Series) -> pd.DataFrame:
    home_stats = game_df[["home_team", "home_xg", "away_xg", "home_goals", "away_goals"]].rename(
        columns={
            "home_team": "team",
            "home_xg": "xg_for",
            "away_xg": "xg_against",
            "home_goals": "goals_for",
            "away_goals": "goals_against",
        }
    )
    away_stats = game_df[["away_team", "away_xg", "home_xg", "away_goals", "home_goals"]].rename(
        columns={
            "away_team": "team",
            "away_xg": "xg_for",
            "home_xg": "xg_against",
            "away_goals": "goals_for",
            "home_goals": "goals_against",
        }
    )
    combined = pd.concat([home_stats, away_stats], ignore_index=True)
    team_totals = combined.groupby("team", as_index=True).sum()
    games_played = combined.groupby("team").size()
    wins = (combined["goals_for"] > combined["goals_against"]).groupby(combined["team"]).sum()
    win_pct = wins / games_played

    rankings = pd.DataFrame({
        "team": team_totals.index,
        "strength_score": strengths.reindex(team_totals.index).values,
        "games_played": games_played.values,
        "xg_for": team_totals["xg_for"].values,
        "xg_against": team_totals["xg_against"].values,
        "xg_diff_per_game": (team_totals["xg_for"] - team_totals["xg_against"]) / games_played.values,
        "win_pct": win_pct.values,
    })

    rankings = rankings.sort_values("strength_score", ascending=False).reset_index(drop=True)
    rankings.insert(0, "rank", np.arange(1, len(rankings) + 1))
    return rankings


def train_strength_logit(game_df: pd.DataFrame, strengths: pd.Series) -> Tuple[LogisticRegression, float, float]:
    strength_diff = (
        game_df["home_team"].map(strengths) - game_df["away_team"].map(strengths)
    ).values.reshape(-1, 1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    log_losses = []
    briers = []

    for train_idx, test_idx in skf.split(strength_diff, game_df["home_win"]):
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        model.fit(strength_diff[train_idx], game_df["home_win"].values[train_idx])
        preds = model.predict_proba(strength_diff[test_idx])[:, 1]
        log_losses.append(log_loss(game_df["home_win"].values[test_idx], preds))
        briers.append(brier_score_loss(game_df["home_win"].values[test_idx], preds))

    print(f"Final model CV log loss: {np.mean(log_losses):.4f}")
    print(f"Final model CV Brier score: {np.mean(briers):.4f}")

    final_model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    final_model.fit(strength_diff, game_df["home_win"].values)
    return final_model, float(np.mean(log_losses)), float(np.mean(briers))


def predict_round1(
    matchups_path: Path,
    strengths: pd.Series,
    model: LogisticRegression,
    model_version: str,
) -> pd.DataFrame:
    matchups = pd.read_excel(matchups_path)
    strength_diff = (
        matchups["home_team"].map(strengths) - matchups["away_team"].map(strengths)
    ).values.reshape(-1, 1)
    p_home = model.predict_proba(strength_diff)[:, 1]
    preds = pd.DataFrame({
        "home_team": matchups["home_team"],
        "away_team": matchups["away_team"],
        "p_home_win": p_home,
        "implied_favorite": np.where(p_home >= 0.5, "home", "away"),
        "model_version": model_version,
    })
    return preds


def build_offense_rows(df: pd.DataFrame) -> pd.DataFrame:
    home_rows = df[[
        "home_team",
        "home_off_line",
        "home_xg",
        "toi",
        "away_def_pairing",
    ]].rename(
        columns={
            "home_team": "team",
            "home_off_line": "off_line",
            "home_xg": "xg",
            "away_def_pairing": "opp_def",
        }
    )
    away_rows = df[[
        "away_team",
        "away_off_line",
        "away_xg",
        "toi",
        "home_def_pairing",
    ]].rename(
        columns={
            "away_team": "team",
            "away_off_line": "off_line",
            "away_xg": "xg",
            "home_def_pairing": "opp_def",
        }
    )
    return pd.concat([home_rows, away_rows], ignore_index=True)


def compute_line_disparity(df: pd.DataFrame) -> pd.DataFrame:
    offense_rows = build_offense_rows(df)
    offense_rows = offense_rows[offense_rows["off_line"].isin(["first_off", "second_off"])].copy()

    offense_rows["toi"] = offense_rows["toi"].fillna(0)
    offense_rows["xg"] = offense_rows["xg"].fillna(0)

    league_def_weights = offense_rows.groupby("opp_def")["toi"].sum()

    grouped = offense_rows.groupby(["team", "off_line", "opp_def"], as_index=False).agg(
        xg_sum=("xg", "sum"),
        toi_sum=("toi", "sum"),
    )
    grouped["xg60"] = np.where(
        grouped["toi_sum"] > 0,
        grouped["xg_sum"] / grouped["toi_sum"] * 3600,
        np.nan,
    )

    records = []
    for (team, off_line), subset in grouped.groupby(["team", "off_line"]):
        weights = league_def_weights.reindex(subset["opp_def"]).values
        xg60_values = subset["xg60"].values
        valid = ~np.isnan(xg60_values) & (weights > 0)
        if valid.any():
            adj_xg60 = np.average(xg60_values[valid], weights=weights[valid])
        else:
            adj_xg60 = np.nan
        total_toi = subset["toi_sum"].sum()
        records.append({
            "team": team,
            "off_line": off_line,
            "adj_xg60": adj_xg60,
            "toi_sum": total_toi,
        })

    adj_df = pd.DataFrame(records)
    pivot = adj_df.pivot(index="team", columns="off_line", values="adj_xg60")
    pivot_toi = adj_df.pivot(index="team", columns="off_line", values="toi_sum")

    result = pd.DataFrame({
        "team": pivot.index,
        "first_off_xg60": pivot.get("first_off"),
        "second_off_xg60": pivot.get("second_off"),
        "toi_first_off": pivot_toi.get("first_off"),
        "toi_second_off": pivot_toi.get("second_off"),
    })

    result["disparity_ratio"] = result["first_off_xg60"] / result["second_off_xg60"]

    def note_row(row: pd.Series) -> str:
        notes = []
        if row["toi_first_off"] == 0 or row["toi_second_off"] == 0:
            notes.append("low_toi")
        notes.append("adj_by_league_def_mix")
        return ";".join(notes)

    result["notes"] = result.apply(note_row, axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=["disparity_ratio"]).sort_values(
        "disparity_ratio", ascending=False
    )
    result = result.reset_index(drop=True).head(10)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main(data_dir: Path, output_dir: Path) -> None:
    df = load_data(data_dir)
    game_df = build_game_level(df)

    strength_result = select_strength_method(game_df)
    print(f"Selected strength method: {strength_result.model_name}")

    rankings = build_team_rankings(game_df, strength_result.strengths)

    model, model_log_loss, model_brier = train_strength_logit(game_df, strength_result.strengths)
    _ = (model_log_loss, model_brier)
    model_version = f"v1_{strength_result.model_name}_logit"

    predictions = predict_round1(
        data_dir / "WHSDSC_Rnd1_matchups.xlsx",
        strength_result.strengths,
        model,
        model_version,
    )

    disparity = compute_line_disparity(df)

    ensure_output_dir(output_dir)
    rankings.to_csv(output_dir / "team_rankings.csv", index=False)
    predictions.to_csv(output_dir / "rnd1_predictions.csv", index=False)
    disparity.to_csv(output_dir / "line_disparity_top10.csv", index=False)

    print("Outputs written:")
    print(output_dir / "team_rankings.csv")
    print(output_dir / "rnd1_predictions.csv")
    print(output_dir / "line_disparity_top10.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
