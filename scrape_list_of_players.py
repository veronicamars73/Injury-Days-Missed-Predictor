from bs4 import BeautifulSoup
import time


# Scrape Players Function
def scrape_players(url,driver):

    players = []

    count = 1
    while count<4:
        #print(url+str(count))
        driver.get(url+str(count))
        time.sleep(3)  # Wait for page to load
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract player rows
        table = soup.find("table", {"class": "items"})
        if not table:
            break  # If no table found, exit

        rows = table.find_all("tr", {"class": ["odd", "even"]})
        for row in rows:
            try:
                name_cell = row.find("table", {"class": "inline-table"}).find("a")
                #print(name_cell)
                player_name = name_cell.text.strip()
                player_url = name_cell['href']
                market_value = row.find("td",{"class": "rechts hauptlink"}).text.strip()
                age = row.find_all("td",{"class": "zentriert"})[2].text.strip()
                players.append({"Name": player_name, "Player Id":player_url.split('/')[-1], "URL": f"https://www.transfermarkt.com.br{player_url}", "Market Value": market_value, "Age": age})
            except Exception as e:
                print(f"Error parsing row: {e}")
        count +=1

    return players