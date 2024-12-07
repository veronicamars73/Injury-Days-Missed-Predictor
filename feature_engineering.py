import pandas as pd

# Read Cleaned DFs
matches_df = pd.read_csv("assets/cleaned_matches.csv")
injuries_df = pd.read_csv("assets/cleaned_injuries.csv")
players_df = pd.read_csv("assets/cleaned_players.csv")

""" Getting Cumulative Minutes Played Before Injury """

# Separating injuries and matches by season
injuries_df['From Date'] = pd.to_datetime(injuries_df['From Date'], errors='coerce')
injuries_df["Season"] = injuries_df["From Date"].dt.year
matches_df['Date'] = pd.to_datetime(matches_df['Date'], errors='coerce')
matches_df["Season"] = matches_df["Date"].dt.year

# Merge injury and match data
injury_matches = injuries_df.merge(matches_df, on=["Player Id", "Season"])

# Filter matches that occurred before each injury
injury_matches = injury_matches[injury_matches["Date"] < injury_matches["From Date"]]

# Calculate cumulative minutes played and number of matches
injury_workload = (
    injury_matches.groupby(["Player Id", "Injury Type", "From Date"])
    .agg({"Minutes Played": "sum", "Date": "count"})
    .reset_index()
    .rename(columns={
        "Minutes Played": "Cumulative Minutes Played",
        "Date": "Matches Played Before Injury"
    })
)

# Merge workload data back to the injury dataset
injuries_df = injuries_df.merge(injury_workload, on=["Player Id", "Injury Type", "From Date"], how="left")

""" Getting Injury History Metrics """

# Count previous injuries and total days missed
injury_history = (
    injuries_df[injuries_df["From Date"] < injuries_df["From Date"].max()]
    .groupby("Player Id")
    .agg({"Days Missed": "sum", "Injury Type": "count"})
    .reset_index()
    .rename(columns={"Days Missed": "Total Days Missed", "Injury Type": "Previous Injuries"})
)

# Merge injury history back into the main dataset
injuries_df = injuries_df.merge(injury_history, on="Player Id", how="left")
injuries_df[["Total Days Missed", "Previous Injuries"]] = injuries_df[["Total Days Missed", "Previous Injuries"]].fillna(0)


""" Build Modeling Dataset """

# Merge all data
final_df = injuries_df.merge(players_df, on="Player Id").merge(matches_df, on="Player Id", how="left")

# Select relevant columns
final_df = final_df[
    [
        "Player Name", "Age", "Position", "Injury Type", "From Date", 
        "Cumulative Minutes Played", "Matches Played Before Injury",
        "Total Days Missed", "Previous Injuries", "Days Missed"
    ]
]

# Drop Duplicates and Reset Index
final_df = final_df.drop_duplicates()
final_df = final_df.reset_index(drop=True)

# Save final dataset
final_df.to_csv("assets/final_injury_dataset.csv", index=False)
