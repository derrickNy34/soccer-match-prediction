"""
02_extract_and_preprocess.py
============================
Extract data from the European Soccer Database, engineer features,
and produce a clean train/test split ready for modeling.

Outputs:
  - data/train.csv
  - data/test.csv
  - data/feature_summary.csv

Usage: python 02_extract_and_preprocess.py
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings("ignore")

DB_PATH = "database.sqlite"
OUTPUT_DIR = "data"
RANDOM_STATE = 42


# =============================================================================
# STEP 1: EXTRACT RAW DATA FROM SQLITE
# =============================================================================

def extract_matches(conn):
    """Pull match data with team names and league info."""
    query = """
    SELECT
        m.id as match_id,
        m.country_id,
        l.name as league_name,
        m.season,
        m.stage,
        m.date,
        m.match_api_id,
        m.home_team_api_id,
        m.away_team_api_id,
        ht.team_long_name as home_team,
        at.team_long_name as away_team,
        m.home_team_goal,
        m.away_team_goal,
        -- Bet365 odds (most consistently available)
        m.B365H, m.B365D, m.B365A,
        -- Bet Win odds (backup)
        m.BWH, m.BWD, m.BWA,
        -- Home player IDs (for aggregating FIFA ratings)
        m.home_player_1, m.home_player_2, m.home_player_3,
        m.home_player_4, m.home_player_5, m.home_player_6,
        m.home_player_7, m.home_player_8, m.home_player_9,
        m.home_player_10, m.home_player_11,
        -- Away player IDs
        m.away_player_1, m.away_player_2, m.away_player_3,
        m.away_player_4, m.away_player_5, m.away_player_6,
        m.away_player_7, m.away_player_8, m.away_player_9,
        m.away_player_10, m.away_player_11
    FROM Match m
    JOIN League l ON m.league_id = l.id
    JOIN Team ht ON m.home_team_api_id = ht.team_api_id
    JOIN Team at ON m.away_team_api_id = at.team_api_id
    WHERE m.home_team_goal IS NOT NULL
      AND m.away_team_goal IS NOT NULL
    ORDER BY m.date
    """
    df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Extracted {len(df):,} matches with valid scores")
    return df


def extract_player_attributes(conn):
    """Pull player FIFA attributes — use the most recent rating before each date."""
    query = """
    SELECT
        player_api_id,
        date,
        overall_rating,
        attacking_work_rate,
        defensive_work_rate
    FROM Player_Attributes
    WHERE overall_rating IS NOT NULL
    ORDER BY player_api_id, date
    """
    df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Extracted {len(df):,} player attribute records")
    return df


def extract_team_attributes(conn):
    """Pull team tactical attributes."""
    query = """
    SELECT
        team_api_id,
        date,
        buildUpPlaySpeed,
        buildUpPlayPassing,
        chanceCreationShooting,
        chanceCreationPassing,
        chanceCreationCrossing,
        defencePressure,
        defenceAggression,
        defenceTeamWidth
    FROM Team_Attributes
    WHERE buildUpPlaySpeed IS NOT NULL
    ORDER BY team_api_id, date
    """
    df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Extracted {len(df):,} team attribute records")
    return df


# =============================================================================
# STEP 2: CREATE THE TARGET VARIABLE
# =============================================================================

def create_target(df):
    """Three-class target: Home Win (H), Draw (D), Away Win (A)."""
    conditions = [
        df["home_team_goal"] > df["away_team_goal"],
        df["home_team_goal"] == df["away_team_goal"],
        df["home_team_goal"] < df["away_team_goal"],
    ]
    labels = ["H", "D", "A"]
    df["result"] = np.select(conditions, labels, default="D")
    print(f"  Target distribution:")
    for label in labels:
        n = (df["result"] == label).sum()
        print(f"    {label}: {n:,} ({n/len(df)*100:.1f}%)")
    return df


# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================

def add_home_advantage(df):
    """Binary flag — always 1 since home team is always listed as home."""
    df["home_advantage"] = 1
    return df


def add_recent_form(df, n_matches=5):
    """
    Rolling form: points earned in last N matches for each team.
    3 pts for win, 1 for draw, 0 for loss.
    Calculated separately for home and away teams.
    """
    print(f"  Computing last-{n_matches} form (this takes a moment)...")

    # Build a long-form results table: one row per team per match
    home = df[["date", "home_team_api_id", "away_team_api_id",
               "home_team_goal", "away_team_goal"]].copy()
    home.columns = ["date", "team", "opponent", "goals_for", "goals_against"]

    away = df[["date", "away_team_api_id", "home_team_api_id",
               "away_team_goal", "home_team_goal"]].copy()
    away.columns = ["date", "team", "opponent", "goals_for", "goals_against"]

    all_results = pd.concat([home, away]).sort_values("date").reset_index(drop=True)

    # Points per match
    all_results["points"] = np.where(
        all_results["goals_for"] > all_results["goals_against"], 3,
        np.where(all_results["goals_for"] == all_results["goals_against"], 1, 0)
    )

    # Goal difference per match
    all_results["gd"] = all_results["goals_for"] - all_results["goals_against"]

    # Rolling stats per team
    form_stats = {}
    for team_id, group in all_results.groupby("team"):
        group = group.sort_values("date")
        # shift(1) so we don't include the current match
        group["form_points"] = group["points"].shift(1).rolling(
            n_matches, min_periods=1
        ).sum()
        group["form_gd"] = group["gd"].shift(1).rolling(
            n_matches, min_periods=1
        ).mean()
        group["form_wins"] = (group["points"].shift(1) == 3).rolling(
            n_matches, min_periods=1
        ).sum()
        for _, row in group.iterrows():
            form_stats[(team_id, row["date"])] = {
                "form_points": row["form_points"],
                "form_gd": row["form_gd"],
                "form_wins": row["form_wins"],
            }

    # Map back to original dataframe
    df["home_form_points"] = df.apply(
        lambda r: form_stats.get(
            (r["home_team_api_id"], r["date"]), {}
        ).get("form_points", np.nan), axis=1
    )
    df["away_form_points"] = df.apply(
        lambda r: form_stats.get(
            (r["away_team_api_id"], r["date"]), {}
        ).get("form_points", np.nan), axis=1
    )
    df["home_form_gd"] = df.apply(
        lambda r: form_stats.get(
            (r["home_team_api_id"], r["date"]), {}
        ).get("form_gd", np.nan), axis=1
    )
    df["away_form_gd"] = df.apply(
        lambda r: form_stats.get(
            (r["away_team_api_id"], r["date"]), {}
        ).get("form_gd", np.nan), axis=1
    )
    df["form_diff"] = df["home_form_points"] - df["away_form_points"]

    filled = df["home_form_points"].notna().sum()
    print(f"    Form computed for {filled:,}/{len(df):,} matches")
    return df


def add_head_to_head(df, n_matches=5):
    """
    Historical head-to-head record between the two teams.
    Looks at last N encounters regardless of home/away.
    Returns home team's H2H win rate.
    """
    print(f"  Computing head-to-head records (last {n_matches} meetings)...")

    h2h_cache = {}
    df = df.sort_values("date").reset_index(drop=True)

    h2h_win_rates = []
    for _, row in df.iterrows():
        t1, t2 = row["home_team_api_id"], row["away_team_api_id"]
        key = (min(t1, t2), max(t1, t2))

        if key not in h2h_cache:
            h2h_cache[key] = []

        # Calculate from previous encounters
        past = h2h_cache[key][-n_matches:]  # last N meetings
        if len(past) > 0:
            # Count how many times team t1 (home) won
            wins = sum(1 for r in past if r["winner"] == t1)
            h2h_win_rates.append(wins / len(past))
        else:
            h2h_win_rates.append(np.nan)

        # Record this match result for future lookups
        if row["home_team_goal"] > row["away_team_goal"]:
            winner = t1
        elif row["home_team_goal"] < row["away_team_goal"]:
            winner = t2
        else:
            winner = None
        h2h_cache[key].append({"date": row["date"], "winner": winner})

    df["h2h_home_win_rate"] = h2h_win_rates
    filled = df["h2h_home_win_rate"].notna().sum()
    print(f"    H2H computed for {filled:,}/{len(df):,} matches")
    return df


def add_betting_features(df):
    """
    Convert betting odds to implied probabilities.
    Uses Bet365 as primary, BetWin as fallback.
    """
    print("  Processing betting odds...")

    # Use B365 odds, fill gaps with BW
    df["odds_H"] = df["B365H"].fillna(df["BWH"])
    df["odds_D"] = df["B365D"].fillna(df["BWD"])
    df["odds_A"] = df["B365A"].fillna(df["BWA"])

    # Convert to implied probability (remove overround by normalizing)
    raw_H = 1.0 / df["odds_H"]
    raw_D = 1.0 / df["odds_D"]
    raw_A = 1.0 / df["odds_A"]
    total = raw_H + raw_D + raw_A

    df["prob_H"] = raw_H / total
    df["prob_D"] = raw_D / total
    df["prob_A"] = raw_A / total

    # Market favorite indicator
    odds_subset = df[["prob_H", "prob_D", "prob_A"]]
    df["market_favorite"] = odds_subset.apply(
        lambda r: r.idxmax() if r.notna().any() else np.nan, axis=1
    )
    df["market_favorite"] = df["market_favorite"].map({
        "prob_H": "H", "prob_D": "D", "prob_A": "A"
    })

    available = df["odds_H"].notna().sum()
    print(f"    Betting odds available for {available:,}/{len(df):,} matches")
    return df


def add_team_ratings(df, player_attrs):
    """
    Aggregate player FIFA overall_rating per team per match.
    Uses the most recent rating available before the match date.
    """
    print("  Aggregating team player ratings (this takes a while)...")

    # Sort player attributes by date for efficient lookup
    player_attrs = player_attrs.sort_values("date")

    # Build a lookup: for each player, get latest rating before a given date
    player_latest = {}
    for _, row in player_attrs.iterrows():
        pid = row["player_api_id"]
        if pid not in player_latest:
            player_latest[pid] = []
        player_latest[pid].append((row["date"], row["overall_rating"]))

    def get_rating(player_id, match_date):
        """Get the most recent rating for a player before the match date."""
        if player_id is None or np.isnan(player_id):
            return np.nan
        pid = int(player_id)
        if pid not in player_latest:
            return np.nan
        ratings = player_latest[pid]
        # Find latest rating before match_date
        valid = [r for d, r in ratings if d <= match_date]
        return valid[-1] if valid else (ratings[0][1] if ratings else np.nan)

    # Calculate average team rating for home and away
    home_cols = [f"home_player_{i}" for i in range(1, 12)]
    away_cols = [f"away_player_{i}" for i in range(1, 12)]

    home_ratings = []
    away_ratings = []

    for idx, row in df.iterrows():
        match_date = row["date"]

        # Home team average
        h_ratings = [get_rating(row[c], match_date) for c in home_cols]
        h_valid = [r for r in h_ratings if not np.isnan(r)]
        home_ratings.append(np.mean(h_valid) if h_valid else np.nan)

        # Away team average
        a_ratings = [get_rating(row[c], match_date) for c in away_cols]
        a_valid = [r for r in a_ratings if not np.isnan(r)]
        away_ratings.append(np.mean(a_valid) if a_valid else np.nan)

        if idx % 5000 == 0 and idx > 0:
            print(f"    ...processed {idx:,} matches")

    df["home_avg_rating"] = home_ratings
    df["away_avg_rating"] = away_ratings
    df["rating_diff"] = df["home_avg_rating"] - df["away_avg_rating"]

    filled = df["home_avg_rating"].notna().sum()
    print(f"    Ratings computed for {filled:,}/{len(df):,} matches")
    return df


def add_team_tactical_attrs(df, team_attrs):
    """
    Add team tactical attributes (build-up speed, defence pressure, etc.)
    Uses most recent team attributes before the match date.
    """
    print("  Adding team tactical attributes...")

    team_attrs = team_attrs.sort_values("date")
    attr_cols = [c for c in team_attrs.columns if c not in
                 ["team_api_id", "date"]]

    # Build lookup per team
    team_latest = {}
    for _, row in team_attrs.iterrows():
        tid = row["team_api_id"]
        if tid not in team_latest:
            team_latest[tid] = []
        team_latest[tid].append((row["date"], {c: row[c] for c in attr_cols}))

    def get_team_attrs(team_id, match_date):
        if team_id not in team_latest:
            return {c: np.nan for c in attr_cols}
        entries = team_latest[team_id]
        valid = [attrs for d, attrs in entries if d <= match_date]
        return valid[-1] if valid else (entries[0][1] if entries else
                                        {c: np.nan for c in attr_cols})

    for col in attr_cols:
        df[f"home_{col}"] = df.apply(
            lambda r: get_team_attrs(r["home_team_api_id"], r["date"]).get(col, np.nan),
            axis=1
        )
        df[f"away_{col}"] = df.apply(
            lambda r: get_team_attrs(r["away_team_api_id"], r["date"]).get(col, np.nan),
            axis=1
        )

    print(f"    Added {len(attr_cols)} tactical attributes per team")
    return df


# =============================================================================
# STEP 4: CLEAN UP AND FINALIZE
# =============================================================================

def select_features(df):
    """Select the final feature set and drop non-generalizable columns."""

    # Columns to drop (IDs, raw data, leakage risks)
    drop_cols = [
        "match_id", "country_id", "match_api_id",
        "home_team_api_id", "away_team_api_id",
        "home_team", "away_team",  # text names — not features
        "home_team_goal", "away_team_goal",  # leakage
        "date",  # used for engineering, not for modeling
        "B365H", "B365D", "B365A", "BWH", "BWD", "BWA",  # raw odds replaced by probs
        "odds_H", "odds_D", "odds_A",  # intermediate
        "market_favorite",  # categorical — encoded below
    ]
    # Drop player ID columns
    drop_cols += [f"home_player_{i}" for i in range(1, 12)]
    drop_cols += [f"away_player_{i}" for i in range(1, 12)]

    # Only drop columns that actually exist
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Encode categoricals
    if "league_name" in df.columns:
        df = pd.get_dummies(df, columns=["league_name"], prefix="league", dtype=int)
    if "season" in df.columns:
        df = pd.get_dummies(df, columns=["season"], prefix="season", dtype=int)
    if "market_favorite" in df.columns:
        df = pd.get_dummies(df, columns=["market_favorite"], prefix="mkt_fav", dtype=int)

    df = df.drop(columns=drop_cols, errors="ignore")

    print(f"  Final feature count: {len(df.columns) - 1}")  # minus target
    return df


def handle_missing(df):
    """Handle missing values — report then fill or drop."""
    total = len(df)
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) > 0:
        print(f"  Missing values before imputation:")
        for col, n in missing.head(10).items():
            print(f"    {col:<35} {n:>6,} ({n/total*100:.1f}%)")

    # Strategy: drop rows where target is missing, fill features with median
    df = df.dropna(subset=["result"])

    feature_cols = [c for c in df.columns if c != "result"]
    for col in feature_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    remaining = df.isnull().sum().sum()
    print(f"  Missing values after imputation: {remaining}")
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    print("=" * 60)
    print("STEP 1: EXTRACTING DATA FROM SQLITE")
    print("=" * 60)
    df = extract_matches(conn)
    player_attrs = extract_player_attributes(conn)
    team_attrs = extract_team_attributes(conn)

    print("\n" + "=" * 60)
    print("STEP 2: CREATING TARGET VARIABLE")
    print("=" * 60)
    df = create_target(df)

    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 60)
    df = add_home_advantage(df)
    df = add_recent_form(df, n_matches=5)
    df = add_head_to_head(df, n_matches=5)
    df = add_betting_features(df)
    df = add_team_ratings(df, player_attrs)
    df = add_team_tactical_attrs(df, team_attrs)

    print("\n" + "=" * 60)
    print("STEP 4: FEATURE SELECTION AND CLEANUP")
    print("=" * 60)
    df = select_features(df)
    df = handle_missing(df)

    print("\n" + "=" * 60)
    print("STEP 5: TRAIN/TEST SPLIT")
    print("=" * 60)
    X = df.drop(columns=["result"])
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Recombine for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"  Train set: {len(train_df):,} rows")
    print(f"  Test set:  {len(test_df):,} rows")
    print(f"  Class balance (train):")
    for label in ["H", "D", "A"]:
        n = (y_train == label).sum()
        print(f"    {label}: {n:,} ({n/len(y_train)*100:.1f}%)")

    # Feature summary
    summary = pd.DataFrame({
        "feature": X.columns,
        "dtype": X.dtypes.values,
        "missing_pct": (X.isnull().sum() / len(X) * 100).values,
        "mean": X.mean(numeric_only=True).reindex(X.columns).values,
        "std": X.std(numeric_only=True).reindex(X.columns).values,
    })
    summary.to_csv(os.path.join(OUTPUT_DIR, "feature_summary.csv"), index=False)

    conn.close()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Output files in '{OUTPUT_DIR}/':")
    print(f"    train.csv           — training data")
    print(f"    test.csv            — test data")
    print(f"    feature_summary.csv — feature statistics")
    print(f"\n  Next step: python 03_train_models.py")


if __name__ == "__main__":
    main()
