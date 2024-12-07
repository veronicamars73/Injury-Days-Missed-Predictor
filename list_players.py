from selenium import webdriver
import pandas as pd
from scrape_list_of_players import scrape_players

# Define URLs for each position
urls = {
    "Goalkeepers": "https://www.transfermarkt.com.br/campeonato-brasileiro-serie-a/marktwerte/wettbewerb/BRA1/ajax/yw1/pos/Torwart/detailpos//altersklasse/alle/land_id/0/plus//galerie/0/page/",
    "Defenders": "https://www.transfermarkt.com.br/campeonato-brasileiro-serie-a/marktwerte/wettbewerb/BRA1/ajax/yw1/pos/Abwehr/detailpos//altersklasse/alle/land_id/0/plus//galerie/0/page/",
    "Midfielders": "https://www.transfermarkt.com.br/campeonato-brasileiro-serie-a/marktwerte/wettbewerb/BRA1/ajax/yw1/pos/Mittelfeld/detailpos//altersklasse/alle/land_id/0/plus//galerie/0/page/",
    "Forwards": "https://www.transfermarkt.com.br/campeonato-brasileiro-serie-a/marktwerte/wettbewerb/BRA1/ajax/yw1/pos/Sturm/detailpos//altersklasse/alle/land_id/0/plus//galerie/0/page/",
}

# Configuring the webdriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Remova esta linha para ver a execução em tempo real
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=options)


all_players = []
for position, url in urls.items():
    print(f"Scraping players for position: {position}")
    position_players = scrape_players(url,driver)
    for player in position_players:
        player["Position"] = position  # Add position info
    all_players.extend(position_players)
    

df = pd.DataFrame(all_players)
df.to_csv("assets/brasileirao_players.csv", index=False)

# Close the driver
driver.quit()

print(df.head(10))