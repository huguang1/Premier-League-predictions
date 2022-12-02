import requests
#
# url = standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
# data = requests.get(url)
#
# # print(data.text)
from bs4 import BeautifulSoup
# soup = BeautifulSoup(data.text)
# standings_table = soup.select('table.stats_table')[0]
#
# # print(standings_table)
#
# links = standings_table.find_all('a')
# print(links[:5])
#
# links = [l.get("href") for l in links]
# links[:5]
# links = [l for l in links if '/squads/' in l]
# links[:5]
# team_urls = [f"https://fbref.com{l}" for l in links]
# team_urls[0]
# data = requests.get(team_urls[0])
import pandas as pd
# matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
#
# soup = BeautifulSoup(data.text)
# links = soup.find_all('a')
# links = [l.get("href") for l in links]
# links = [l for l in links if l and 'all_comps/shooting/' in l]
#
# data = requests.get(f"https://fbref.com{links[0]}")
# shooting = pd.read_html(data.text, match="Shooting")[0]
# shooting
# shooting.columns = shooting.columns.droplevel()
# team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
# team_data
years = list(range(2023, 2021, -1))
all_matches = []
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

import time

for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select('table.stats_table')[0]
    time.sleep(10)

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        time.sleep(5)
        soup = BeautifulSoup(data.text)
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        time.sleep(10)
        try:
            shooting = pd.read_html(data.text, match="Shooting")[0]
            shooting.columns = shooting.columns.droplevel()

            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        team_data = team_data[team_data["Comp"] == "Premier League"]

        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(10)

print(all_matches)
len(all_matches)
match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]
print(match_df)
match_df.to_csv("Premier_league_19_20.csv")







