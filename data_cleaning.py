import pandas as pd

# Load the datasets
players_df = pd.read_csv("assets/brasileirao_players.csv")
matches_df = pd.read_csv("assets/player_match_data.csv")
injuries_df = pd.read_csv("assets/player_injury_data.csv")

# Normalize date formats
matches_df["Date"] = pd.to_datetime(matches_df["Date"], format="%d/%m/%y")
injuries_df["From Date"] = pd.to_datetime(injuries_df["From Date"], format="%d/%m/%Y")
injuries_df["Until Date"] = pd.to_datetime(injuries_df["Until Date"], format="%d/%m/%Y")

# Handle missing Until Date
injuries_df["Until Date"] = injuries_df["Until Date"].fillna(injuries_df["From Date"])

# Convert Minutes Played to integers
matches_df["Minutes Played"] = matches_df["Minutes Played"].str.replace("'", "").astype(int)

# Add a home/away indicator
matches_df["Home Game"] = matches_df["Home-Game"].map({1: "Home", 0: "Away"})

# Save cleaned data
players_df.to_csv("assets/cleaned_players.csv", index=False)
matches_df.to_csv("assets/cleaned_matches.csv", index=False)
injuries_df.to_csv("assets/cleaned_injuries.csv", index=False)