"""
01_explore_database.py
=====================
Explore the European Soccer Database SQLite file.
Run this first to understand what you're working with.

Usage: python 01_explore_database.py
"""

import sqlite3
import pandas as pd

DB_PATH = "database.sqlite"


def main():
    conn = sqlite3.connect(DB_PATH)

    # ---- 1. List all tables ----
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
    )
    print("=" * 60)
    print("TABLES IN DATABASE")
    print("=" * 60)
    for t in tables["name"]:
        count = pd.read_sql(f"SELECT COUNT(*) as n FROM [{t}]", conn)["n"][0]
        print(f"  {t:<25} {count:>8,} rows")

    # ---- 2. Column info for each table ----
    print("\n" + "=" * 60)
    print("COLUMN DETAILS PER TABLE")
    print("=" * 60)
    for t in tables["name"]:
        cols = pd.read_sql(f"PRAGMA table_info([{t}])", conn)
        print(f"\n--- {t} ({len(cols)} columns) ---")
        for _, row in cols.iterrows():
            print(f"  {row['name']:<35} {row['type']}")

    # ---- 3. Preview Match table (the main one) ----
    print("\n" + "=" * 60)
    print("MATCH TABLE PREVIEW (first 3 rows)")
    print("=" * 60)
    matches = pd.read_sql("SELECT * FROM Match LIMIT 3", conn)
    for col in matches.columns:
        print(f"  {col:<35} {str(matches[col].iloc[0])[:60]}")

    # ---- 4. Check data quality ----
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECK — MATCH TABLE")
    print("=" * 60)
    match_full = pd.read_sql("SELECT * FROM Match", conn)
    total = len(match_full)
    print(f"  Total matches: {total:,}")
    print(f"\n  Missing values (top columns):")
    missing = match_full.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(20)
    for col, n in missing.items():
        pct = n / total * 100
        print(f"    {col:<35} {n:>6,} ({pct:.1f}%)")

    # ---- 5. Target variable distribution ----
    print("\n" + "=" * 60)
    print("TARGET VARIABLE DISTRIBUTION")
    print("=" * 60)
    results = pd.read_sql("""
        SELECT
            CASE
                WHEN home_team_goal > away_team_goal THEN 'Home Win'
                WHEN home_team_goal = away_team_goal THEN 'Draw'
                ELSE 'Away Win'
            END as result,
            COUNT(*) as count
        FROM Match
        WHERE home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL
        GROUP BY result
        ORDER BY count DESC
    """, conn)
    for _, row in results.iterrows():
        print(f"  {row['result']:<15} {row['count']:>6,}")

    # ---- 6. Leagues and date range ----
    print("\n" + "=" * 60)
    print("LEAGUES AND SEASONS")
    print("=" * 60)
    leagues = pd.read_sql("""
        SELECT l.name as league, COUNT(*) as matches,
               MIN(m.date) as first_match, MAX(m.date) as last_match
        FROM Match m
        JOIN League l ON m.league_id = l.id
        GROUP BY l.name
        ORDER BY matches DESC
    """, conn)
    for _, row in leagues.iterrows():
        print(f"  {row['league']:<35} {row['matches']:>5} matches  "
              f"({row['first_match'][:4]}–{row['last_match'][:4]})")

    # ---- 7. Betting odds availability ----
    print("\n" + "=" * 60)
    print("BETTING ODDS AVAILABILITY")
    print("=" * 60)
    odds_cols = [c for c in match_full.columns if any(
        c.startswith(p) for p in ["B365", "BW", "IW", "LB", "WH", "VC"]
    )]
    print(f"  Betting columns found: {len(odds_cols)}")
    for col in odds_cols[:12]:
        avail = match_full[col].notna().sum()
        pct = avail / total * 100
        print(f"    {col:<20} {avail:>6,} available ({pct:.1f}%)")

    # ---- 8. Player attributes sample ----
    print("\n" + "=" * 60)
    print("PLAYER ATTRIBUTES SAMPLE")
    print("=" * 60)
    pa = pd.read_sql("SELECT * FROM Player_Attributes LIMIT 3", conn)
    key_attrs = ["overall_rating", "attacking_work_rate", "defensive_work_rate",
                 "crossing", "finishing", "short_passing", "volleys",
                 "aggression", "interceptions", "positioning", "vision"]
    for attr in key_attrs:
        if attr in pa.columns:
            print(f"  {attr:<30} {pa[attr].iloc[0]}")

    conn.close()
    print("\n✓ Exploration complete. Run 02_extract_and_preprocess.py next.")


if __name__ == "__main__":
    main()
