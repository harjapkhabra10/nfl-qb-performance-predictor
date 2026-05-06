import pandas as pd
import numpy as np
import nflreadpy as nfl

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

pd.set_option("display.max_columns", None)


# 1. LOAD RAW NFL DATA
def load_raw_data():
    """
    Exact replica of the notebook loading logic.
    """
    # player stats (game-level)
    player = nfl.load_player_stats(seasons=range(2018, 2026)).to_pandas()
    # regular season only
    player = player[player["season_type"] == "REG"]

    # team game stats (off + def)
    team_stats = nfl.load_team_stats(seasons=range(2018, 2026)).to_pandas()

    # schedules
    schedule_df = nfl.load_schedules(seasons=range(2018, 2026)).to_pandas()

    return player, team_stats, schedule_df


# 2. BUILD FULL QB DATASET (qbs_merged)
def build_qb_dataset(player, team_stats, schedule_df):
    """
    This is a line-for-line functional port of notebook sections 2–8
    that produce qbs_merged.
    """

    # base QB filter
    qbs = player[player["position"] == "QB"].copy()
    qbs = qbs[qbs["season_type"] == "REG"]
    qbs = qbs[qbs["attempts"].fillna(0) >= 10]
    qbs = qbs[qbs["season"] >= 2018]

    qb_cols = [
        "player_id",
        "player_name",
        "season",
        "week",
        "team",
        "opponent_team",

        # passing
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "passing_air_yards",
        "passing_yards_after_catch",
        "passing_first_downs",
        "passing_epa",
        "passing_cpoe",

        # pressure
        "sacks_suffered",
        "sack_yards_lost",

        # rushing
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_epa",
    ]
    qbs = qbs[qb_cols]
    qbs = qbs.sort_values(["player_id", "season", "week"])


    # MERGE SINGLE-GAME OPPONENT DEFENSIVE STATS
    defense_cols = [
        "season", "week", "team",
        "def_sacks",
        "def_qb_hits",
        "def_interceptions",
        "def_pass_defended",
        "def_tackles_for_loss",
        "def_safeties",
        "def_fumbles",
        "penalties",
        "penalty_yards"
    ]

    defense = team_stats[defense_cols].copy()
    defense = defense.rename(
        columns=lambda c: f"opp_{c}" if c not in ["season", "week", "team"] else c
    )

    qbs_merged = qbs.merge(
        defense,
        left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "team"],
        how="left"
    )

    qbs_merged = qbs_merged.drop(columns=["team_y"])
    qbs_merged = qbs_merged.rename(columns={"team_x": "team"})

    # ADD OPPONENT DEFENSIVE ROLLING FEATURES
    defense_raw = team_stats[defense_cols].copy()
    defense_raw = defense_raw.sort_values(["team", "season", "week"])

    def_stats = [
        "def_sacks",
        "def_qb_hits",
        "def_interceptions",
        "def_pass_defended",
        "def_tackles_for_loss",
        "def_fumbles",
        "penalties",
        "penalty_yards"
    ]

    for stat in def_stats:
        # 3-game
        defense_raw[f"{stat}_3"] = (
            defense_raw.groupby("team")[stat]
            .rolling(window=3, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        # 5-game
        defense_raw[f"{stat}_5"] = (
            defense_raw.groupby("team")[stat]
            .rolling(window=5, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        # full season
        defense_raw[f"{stat}_season"] = (
            defense_raw.groupby(["team", "season"])[stat]
            .expanding(min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=[0, 1], drop=True)
        )

    qbs_merged = qbs_merged.merge(
        defense_raw,
        left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "team"],
        how="left",
        suffixes=("", "_defraw")
    )

    qbs_merged = qbs_merged.drop(columns=["team_defraw"])

    # drop duplicate defense rolling columns
    dup_cols = [c for c in qbs_merged.columns if c.endswith("_defraw")]
    qbs_merged = qbs_merged.drop(columns=dup_cols)

    raw_def_base = [
        "def_sacks",
        "def_qb_hits",
        "def_interceptions",
        "def_pass_defended",
        "def_tackles_for_loss",
        "def_safeties",
        "def_fumbles",
        "penalties",
        "penalty_yards"
    ]
    raw_def_base = [c for c in raw_def_base if c in qbs_merged.columns]
    qbs_merged = qbs_merged.drop(columns=raw_def_base)

    # some renaming
    rename_map = {}
    for c in qbs_merged.columns:
        for stat in def_stats:
            if c.startswith(stat + "_"):
                rename_map[c] = "opp_" + c
                break

    qbs_merged = qbs_merged.rename(columns=rename_map)

    # ADD TEAM OFFENSIVE ROLLING FEATURES
    team_off_stats = [
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "passing_epa",
        "passing_first_downs",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_epa"
    ]

    team_stats_off = team_stats[["season", "week", "team"] + team_off_stats].copy()
    team_stats_off = team_stats_off.sort_values(["team", "season", "week"])

    for stat in team_off_stats:
        team_stats_off[f"{stat}_3"] = (
            team_stats_off.groupby("team")[stat]
            .rolling(window=3, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        team_stats_off[f"{stat}_5"] = (
            team_stats_off.groupby("team")[stat]
            .rolling(window=5, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        team_stats_off[f"{stat}_season"] = (
            team_stats_off.groupby(["team", "season"])[stat]
            .expanding(min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=[0, 1], drop=True)
        )

    qbs_merged = qbs_merged.merge(
        team_stats_off,
        on=["season", "week", "team"],
        how="left"
    )

    rename_map = {
        # QB-level
        "attempts_x": "qb_attempts",
        "completions_x": "qb_completions",
        "passing_yards_x": "qb_passing_yards",
        "passing_tds_x": "qb_passing_tds",
        "passing_interceptions_x": "qb_passing_int",
        "passing_epa_x": "qb_passing_epa",
        "passing_first_downs_x": "qb_passing_first_downs",
        "carries_x": "qb_carries",
        "rushing_yards_x": "qb_rushing_yards",
        "rushing_tds_x": "qb_rushing_tds",
        "rushing_epa_x": "qb_rushing_epa",

        # Team-level
        "attempts_y": "team_attempts",
        "completions_y": "team_completions",
        "passing_yards_y": "team_passing_yards",
        "passing_tds_y": "team_passing_tds",
        "passing_interceptions_y": "team_passing_int",
        "passing_epa_y": "team_passing_epa",
        "passing_first_downs_y": "team_passing_first_downs",
        "carries_y": "team_carries",
        "rushing_yards_y": "team_rushing_yards",
        "rushing_tds_y": "team_rushing_tds",
        "rushing_epa_y": "team_rushing_epa",
    }

    qbs_merged = qbs_merged.rename(columns=rename_map)

    # ADD is_home FLAG
    schedule = schedule_df[["season", "week", "home_team", "away_team"]].drop_duplicates()

    schedule_home = schedule.copy()
    schedule_home["is_home"] = 1
    home = schedule_home[["season", "week", "home_team", "is_home"]].rename(
        columns={"home_team": "team"}
    )

    schedule_away = schedule.copy()
    schedule_away["is_home"] = 0
    away = schedule_away[["season", "week", "away_team", "is_home"]].rename(
        columns={"away_team": "team"}
    )

    is_home_df = pd.concat([home, away], ignore_index=True)

    qbs_merged = qbs_merged.merge(
        is_home_df,
        on=["season", "week", "team"],
        how="left"
    )

    qbs_merged = qbs_merged.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # ADD QB ROLLING FEATURES
    qbs_merged = qbs_merged.sort_values(["player_id", "season", "week"])

    qb_roll_stats = {
        "qb_attempts": "qb_attempts",
        "qb_completions": "qb_completions",
        "qb_passing_yards": "qb_passing_yards",
        "qb_passing_tds": "qb_passing_tds",
        "qb_passing_int": "qb_passing_int",
        "qb_passing_epa": "qb_passing_epa",
        "sacks_suffered": "sacks_suffered",
        "qb_rushing_yards": "qb_rushing_yards",
        "qb_rushing_tds": "qb_rushing_tds",
        "qb_rushing_epa": "qb_rushing_epa",
        "passing_cpoe": "passing_cpoe"
    }

    for col in qb_roll_stats.values():
        # 3-game
        qbs_merged[f"{col}_3"] = (
            qbs_merged.groupby("player_id")[col]
            .rolling(3, min_periods=1).mean()
            .shift(1).reset_index(level=0, drop=True)
        )

        # 5-game
        qbs_merged[f"{col}_5"] = (
            qbs_merged.groupby("player_id")[col]
            .rolling(5, min_periods=1).mean()
            .shift(1).reset_index(level=0, drop=True)
        )

        # full season
        qbs_merged[f"{col}_season"] = (
            qbs_merged.groupby(["player_id", "season"])[col]
            .expanding(min_periods=1).mean()
            .shift(1).reset_index(level=[0, 1], drop=True)
        )

    # CLIP OUTLIERS (EPA & CPOE)
    qbs_merged["qb_passing_epa"] = qbs_merged["qb_passing_epa"].clip(-10, 10)
    qbs_merged["qb_rushing_epa"] = qbs_merged["qb_rushing_epa"].clip(-10, 10)
    qbs_merged["passing_cpoe"]   = qbs_merged["passing_cpoe"].clip(-10, 10)

    for col in ["qb_passing_epa", "qb_rushing_epa", "passing_cpoe"]:
        qbs_merged[f"{col}_3"]      = qbs_merged[f"{col}_3"].clip(-10, 10)
        qbs_merged[f"{col}_5"]      = qbs_merged[f"{col}_5"].clip(-10, 10)
        qbs_merged[f"{col}_season"] = qbs_merged[f"{col}_season"].clip(-10, 10)

    return qbs_merged


# 3. BUILD FEATURE MATRIX
def build_feature_matrix(qbs_merged):
    y = qbs_merged["qb_passing_yards"]

    drop_cols = [
        "player_id",
        "player_name",
        "qb_passing_yards",
        "season",
        "week",
        "opponent_team",
        "team"
    ]

    X = qbs_merged.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=["float64", "int64"])

    # drop all-nan & constant columns
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique() > 1]

    return X, y, list(X.columns)


# 4. BUILD qb1_2025 FROM DEPTH CHARTS
def build_qb1_2025():
    depth = nfl.load_depth_charts(seasons=2025).to_pandas()
    depth_qb = depth[depth["pos_abb"] == "QB"]
    depth_qb1 = depth_qb[depth_qb["pos_rank"] == 1]

    depth_qb1_latest = (
        depth_qb1.sort_values(["team", "dt"], ascending=[True, False])
        .groupby("team")
        .first()
        .reset_index()
    )

    qb1_2025 = (
        depth_qb1_latest[["team", "player_name", "gsis_id"]]
        .rename(columns={"gsis_id": "player_id"})
        .sort_values("team")
        .reset_index(drop=True)
    )

    return qb1_2025


# 5. FUTURE QB ROWS (2025 WEEKS 13–18)
def build_future_qb_rows(qb1_2025, schedule_df):
    sched_2025 = schedule_df[schedule_df["season"] == 2025].copy()

    future_sched = sched_2025[
        (sched_2025["game_type"] == "REG") &
        (sched_2025["week"] >= 13) &
        (sched_2025["week"] <= 18)
    ][["week", "home_team", "away_team"]].reset_index(drop=True)

    # merge starting QBs
    future_sched = future_sched.merge(
        qb1_2025.rename(columns={"team": "home_team", "player_name": "home_qb"}),
        on="home_team", how="left"
    )

    future_sched = future_sched.merge(
        qb1_2025.rename(columns={"team": "away_team", "player_name": "away_qb"}),
        on="away_team", how="left"
    )

    home_rows = pd.DataFrame({
        "season":        2025,
        "week":          future_sched["week"],
        "team":          future_sched["home_team"],
        "opponent_team": future_sched["away_team"],
        "player_name":   future_sched["home_qb"],
        "player_id":     future_sched["player_id_x"],
        "is_home":       1
    })

    away_rows = pd.DataFrame({
        "season":        2025,
        "week":          future_sched["week"],
        "team":          future_sched["away_team"],
        "opponent_team": future_sched["home_team"],
        "player_name":   future_sched["away_qb"],
        "player_id":     future_sched["player_id_y"],
        "is_home":       0
    })

    future_qb_rows = pd.concat([home_rows, away_rows]).reset_index(drop=True)
    future_qb_rows = future_qb_rows.sort_values(
        ["week", "team", "is_home"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    return future_qb_rows


# 6. BUILD SNAPSHOTS AND FUTURE FEATURE MATRIX
def build_snapshots_and_future_matrix(qbs_merged, X, future_qb_rows):
    feature_cols = list(X.columns)

    opp_cols = [c for c in feature_cols if c.startswith("opp_")]
    base_off_cols = [c for c in feature_cols if c not in opp_cols + ["is_home"]]

    PAST_CUTOFF = (qbs_merged["season"] < 2025) | (
        (qbs_merged["season"] == 2025) &
        (qbs_merged["week"] <= 12)
    )

    historical = qbs_merged[PAST_CUTOFF].copy()

    # defensive snapshot
    defense_snapshot = (
        historical
        .sort_values(["opponent_team", "season", "week"])
        .groupby("opponent_team")
        .tail(1)[["opponent_team"] + opp_cols]
        .rename(columns={"opponent_team": "def_team"})
        .reset_index(drop=True)
    )

    # latest QB stats
    latest_qb_stats = (
        historical
        .sort_values(["player_id", "season", "week"])
        .groupby("player_id")
        .tail(1)[["player_id"] + base_off_cols]
    )

    is_home_col = future_qb_rows["is_home"].astype(float)
    future_feature_rows = future_qb_rows.drop(columns=["is_home"]).copy()

    future_feature_rows = future_feature_rows.merge(
        latest_qb_stats,
        on="player_id",
        how="left"
    )

    future_feature_rows = future_feature_rows.merge(
        defense_snapshot,
        left_on="opponent_team",
        right_on="def_team",
        how="left"
    )

    future_feature_rows = future_feature_rows.drop(columns=["def_team"])
    future_feature_rows["is_home"] = is_home_col

    X_future_final = future_feature_rows[feature_cols].copy()

    return X_future_final, future_feature_rows


# 7. TRAIN FINAL XGBOOST MODELS
def train_final_models(X, qbs_merged):
    completed_mask = (qbs_merged["season"] < 2025) | (
        (qbs_merged["season"] == 2025) &
        (qbs_merged["week"] <= 12)
    )

    X_train_final = X.loc[completed_mask]

    y_yards_all       = qbs_merged["qb_passing_yards"]
    y_attempts_all    = qbs_merged["qb_attempts"]
    y_completions_all = qbs_merged["qb_completions"]

    y_train_yards_final       = y_yards_all[completed_mask]
    y_train_attempts_final    = y_attempts_all[completed_mask]
    y_train_completions_final = y_completions_all[completed_mask]

    xgb_params = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    xgb_final_yards = XGBRegressor(**xgb_params)
    xgb_final_yards.fit(X_train_final, y_train_yards_final)

    xgb_final_attempts = XGBRegressor(**xgb_params)
    xgb_final_attempts.fit(X_train_final, y_train_attempts_final)

    xgb_final_completions = XGBRegressor(**xgb_params)
    xgb_final_completions.fit(X_train_final, y_train_completions_final)

    return xgb_final_yards, xgb_final_attempts, xgb_final_completions


# 8. BUILD FUTURE RESULTS (WEEKS 13–18)
def build_future_results(future_qb_rows, X_future_final,
                         xgb_final_yards, xgb_final_attempts, xgb_final_completions):
    future_yards       = xgb_final_yards.predict(X_future_final)
    future_attempts    = xgb_final_attempts.predict(X_future_final)
    future_completions = xgb_final_completions.predict(X_future_final)

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_attempts   = np.where(future_attempts <= 0, np.nan, future_attempts)
        future_comp_pct = (future_completions / safe_attempts) * 100.0
        future_ypa      = future_yards / safe_attempts

    future_results = pd.DataFrame({
        "season":        future_qb_rows["season"],
        "week":          future_qb_rows["week"],
        "team":          future_qb_rows["team"],
        "opponent_team": future_qb_rows["opponent_team"],
        "player_name":   future_qb_rows["player_name"],
        "player_id":     future_qb_rows["player_id"],
        "is_home":       future_qb_rows["is_home"].astype(int),

        "predicted_attempts":      np.round(future_attempts, 1),
        "predicted_completions":   np.round(future_completions, 1),
        "predicted_passing_yards": np.round(future_yards, 1),
        "predicted_comp_pct":      np.round(future_comp_pct, 1),
        "predicted_ypa":           np.round(future_ypa, 2),
    })

    future_results = future_results.sort_values(["week", "team"]).reset_index(drop=True)
    return future_results


# 9. PIPELINE FOR STREAMLIT 
def run_full_pipeline():
    """
    This is the function Streamlit calls.
    It wires together all necessary functions
    """
    player, team_stats, schedule_df = load_raw_data()

    qbs_merged = build_qb_dataset(player, team_stats, schedule_df)
    X, _, _ = build_feature_matrix(qbs_merged)

    qb1_2025 = build_qb1_2025()
    future_qb_rows = build_future_qb_rows(qb1_2025, schedule_df)

    X_future_final, _ = build_snapshots_and_future_matrix(qbs_merged, X, future_qb_rows)

    xgb_final_yards, xgb_final_attempts, xgb_final_completions = train_final_models(X, qbs_merged)

    future_results = build_future_results(
        future_qb_rows,
        X_future_final,
        xgb_final_yards,
        xgb_final_attempts,
        xgb_final_completions
    )

    return future_results
