"""
Loads and unifies polling-average data (three different raw formats) and
actual election results into a single clean schema:

    polls_df:   cycle, state, party, candidate, date, pct_estimate
    results_df: cycle, state, party, candidate, pct_actual, winner
"""

import pandas as pd

RAW = "data/raw"

# States/districts to drop from state-level modeling (national polls, and
# congressional-district-level splits we're not modeling separately)
DROP_GEOS = {"National", "ME-1", "ME-2", "NE-1", "NE-2", "NE-3"}

# Manual party maps for files that don't carry a party column
PARTY_2016 = {
    "Donald Trump": "REP",
    "Hillary Rodham Clinton": "DEM",
}
PARTY_2020 = {
    "Donald Trump": "REP",
    "Joseph R. Biden Jr.": "DEM",
}


def load_polls_2016() -> pd.DataFrame:
    df = pd.read_csv(f"{RAW}/pres_pollaverages_1968-2016.csv", low_memory=False)
    df = df[df["cycle"] == 2016]
    df = df[df["candidate_name"].isin(PARTY_2016)]
    df = df[~df["state"].isin(DROP_GEOS)]
    df["party"] = df["candidate_name"].map(PARTY_2016)
    df["date"] = pd.to_datetime(df["modeldate"], errors="coerce")
    out = df.rename(columns={"candidate_name": "candidate"})
    return out[["cycle", "state", "party", "candidate", "date", "pct_estimate"]]


def load_polls_2020() -> pd.DataFrame:
    df = pd.read_csv(f"{RAW}/presidential_general_averages_2024.csv")
    df = df[df["cycle"] == 2020]
    df = df[df["candidate"].isin(PARTY_2020)]
    df = df[~df["state"].isin(DROP_GEOS)]
    df["party"] = df["candidate"].map(PARTY_2020)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # The 2020 rows in this file predate the 'pct_estimate' column being
    # populated -- 'pct_trend_adjusted' is the equivalent averaged
    # estimate for that era, so we use it instead.
    df["pct_estimate"] = df["pct_trend_adjusted"]
    return df[["cycle", "state", "party", "candidate", "date", "pct_estimate"]]

def load_polls_2024() -> pd.DataFrame:
    df = pd.read_csv(f"{RAW}/presidential_general_averages_2024.csv")
    df = df[df["cycle"] == 2024]
    df = df[df["candidate"].isin(["Trump", "Harris"])]  # two-party race
    df = df[~df["state"].isin(DROP_GEOS)]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[["cycle", "state", "party", "candidate", "date", "pct_estimate"]]

def load_all_polls() -> pd.DataFrame:
    polls = pd.concat(
        [load_polls_2016(), load_polls_2020(), load_polls_2024()],
        ignore_index=True,
    )
    polls = polls.dropna(subset=["pct_estimate", "date"])
    return polls


def closing_polls(polls_df: pd.DataFrame) -> pd.DataFrame:
    """Last available poll-average reading per cycle/state/party.

    For 2016 and 2020 this is effectively election eve. For 2024 the
    public data only goes through Sept 12, 2024 (538's live site with the
    full cycle was taken offline before this project was built) -- so the
    2024 'closing' average here is really an as-of-Sept-12 snapshot, not
    a true election-eve number. Documented as a known limitation.
    """
    idx = polls_df.groupby(["cycle", "state", "party"])["date"].idxmax()
    return polls_df.loc[idx].reset_index(drop=True)


def load_results() -> pd.DataFrame:
    df = pd.read_csv(f"{RAW}/election_results_presidential.csv", low_memory=False)
    df = df[(df["stage"] == "general") & (df["cycle"].isin([2016, 2020, 2024]))]
    df = df[df["state"].notna() & (df["state"] != "")]
    df = df[df["state"] != "Puerto Rico"]
    df = df[~df["state"].str.contains("CD-", na=False)]  # drop ME/NE district splits
    df = df[df["ballot_party"].isin(["DEM", "REP"])]
    df = df.drop(columns=["party"]).rename(columns={"ballot_party": "party"})

    # 2024 has multiple DEM rows historically (Biden withdrew) -- keep the
    # candidate with the higher vote total per state/party as the nominee
    # actually on the ballot.
    df = df.sort_values("votes", ascending=False)
    df = df.drop_duplicates(subset=["cycle", "state", "party"], keep="first")

    df = df.rename(columns={"candidate_name": "candidate", "percent": "pct_actual"})
    return df[["cycle", "state", "party", "candidate", "pct_actual", "winner"]]

if __name__ == "__main__":
    polls = load_all_polls()
    closing = closing_polls(polls)
    results = load_results()
    print("Polls rows:", len(polls))
    print("Closing polls rows:", len(closing))
    print(closing.groupby("cycle")["state"].nunique())
    print("Results rows:", len(results))
    print(results.groupby("cycle")["state"].nunique())