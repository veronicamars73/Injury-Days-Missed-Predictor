from bs4 import BeautifulSoup
import time

def scrape_injury_data(player_url, player_name, driver):
    injury_data = []
    #url = f"{player_url}/verletzungen/spieler/{player_url.split('/')[-1]}/plus/1"
    url = f"{player_url.split('/')[0]}//{player_url.split('/')[2]}/{player_url.split('/')[3]}/verletzungen/spieler/{player_url.split('/')[-1]}/plus/1"
    print(url)
    driver.get(url)
    time.sleep(2)  # Wait for page to load

    # Parse the page with Beautiful Soup
    soup = BeautifulSoup(driver.page_source, "html.parser")
    # Find the injury table
    table = soup.find("table", {"class": "items"})
    if table:
        rows = table.find("tbody").find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells)>2:
                data = {
                    "Player Name": player_name,
                    "Player Id": player_url.split('/')[-1],
                    "Injury Type": cells[1].text.strip(),
                    "From Date": cells[2].text.strip(),
                    "Until Date": cells[3].text.strip(),
                    "Days Missed": cells[4].text.replace('dias','').strip(),
                }
                injury_data.append(data)
    return injury_data