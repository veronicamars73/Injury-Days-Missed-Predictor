from selenium import webdriver
import pandas as pd
from scrape_injury_data import scrape_injury_data

# Configuring the webdriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Remova esta linha para ver a execução em tempo real
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=options)

# Load player DataFrame
players_df = pd.read_csv("assets/brasileirao_players.csv")

# Collect match data for all players
all_injury_data = []
for index, row in players_df.iterrows():
    print(f"Scraping match data for: {row['Player Name']}")
    injuries = scrape_injury_data(row["URL"], row["Player Name"],driver)
    all_injury_data.extend(injuries)

# Save data to CSV
injury_df = pd.DataFrame(all_injury_data)

injury_df.to_csv("assets/player_injury_data.csv", index=False)

print("Data scraping completed!")