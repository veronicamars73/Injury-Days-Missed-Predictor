from selenium import webdriver
import pandas as pd
from scrape_match_list import scrape_match_data

# Configuring the webdriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Remova esta linha para ver a execução em tempo real
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=options)


# Load player DataFrame
players_df = pd.read_csv("assets/brasileirao_players.csv")

# Collect match data for all players
all_match_data = []
for index, row in players_df.iterrows():
    print(f"Scraping match data for: {row['Name']}")
    matches = scrape_match_data(row["URL"], row["Name"],driver)
    all_match_data.extend(matches)
    
# Save data to CSV
match_data_df = pd.DataFrame(all_match_data)
match_data_df.to_csv("assets/player_match_data.csv", index=False)

# Close the Selenium driver
driver.quit()

print("Match data scraping completed!")

print(match_data_df.head(10))