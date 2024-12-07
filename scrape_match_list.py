from bs4 import BeautifulSoup
import time


# Scrape matches by player Function
def scrape_match_data(player_url, player_name, driver):
    match_data = []
    for season in ["2023", "2022", "2021"]:
        url = f"{player_url.split('/')[0]}//{player_url.split('/')[2]}/{player_url.split('/')[3]}/leistungsdaten/spieler/{player_url.split('/')[-1]}/plus/0?saison={season}"
        print(url)
        driver.get(url)
        time.sleep(2)  # Wait for page to load

        # Parse the page with Beautiful Soup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table_divs = soup.find_all("div", {"class": "responsive-table"})[1:]
        print(len(table_divs))
        for div in table_divs:
          table = div.find("table").find('tbody')

          if table:
              rows = table.find_all("tr")[1:]
              for row in rows:
                  #print(row)
                    if row.get('class')==[]:
                      cells = row.find_all("td")
                      if len(cells)>2:
                          data = {
                              "Player Name": player_name,
                              "Player Id": player_url.split('/')[-1],
                              "Season": season,
                              "Date": cells[1].text.strip(),
                              "Home-Game": 1 if cells[2].text.strip() == "C" else 0,
                              "Minutes Played": cells[-1].text.strip(),
                              }
                          match_data.append(data)
    return match_data