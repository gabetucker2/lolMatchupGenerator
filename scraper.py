import requests
import pandas as pd

from parsel import Selector

from functions import normalize_champ_name

PATCH = "" # blank will use current patch, otherwise try something like "15.22" to use a prior patch
CHAMPIONS_URL_TEMPLATE = "https://op.gg/lol/champions?position={role}&patch={patch}"
MATCHUPS_URL_TEMPLATE = "https://op.gg/lol/champions/{champion}/counters/{role}?patch={patch}"


def get_champions_for_role(role):
    """Scrape the list of top champions for a given role from OP.GG."""
    url = CHAMPIONS_URL_TEMPLATE.format(role=role, patch=PATCH)
    html = requests.get(url).text
    sel = Selector(text=html)
    champions = sel.css("table tbody tr td:nth-child(2) ::text").getall()
    champions.sort()
    return champions


def create_empty_df(champions):
    """Create an empty square dataframe with champions on both rows and columns."""
    empty_df = pd.DataFrame(index=champions, columns=champions)
    empty_df.index.name = "VS (below)"
    return empty_df


def mark_diagonal(win_rate_df):
    """Mark the diagonal of the dataframe with '/' since a champion cannot play against themselves."""
    for champion in win_rate_df.index:
        win_rate_df.at[champion, champion] = '/'
    return win_rate_df


def get_url_champion_name(champion_name):
    """OP.GG has some weird conventions for how certain champion name are put into URLs."""
    norm_champ_name = normalize_champ_name(champion_name)
    norm_champ_name = norm_champ_name.replace(" ", "")
    norm_champ_name = "monkeyking" if norm_champ_name == "wukong" else norm_champ_name
    norm_champ_name = "renata" if norm_champ_name == "renataglasc" else norm_champ_name
    norm_champ_name = "nunu" if norm_champ_name == "nunuwillump" else norm_champ_name
    return norm_champ_name


def get_matchup_win_rates_bulk(win_rate_df, role):
    """We visit each champion's matchup page once and extract all the available matchup win rates from there.
    Some matchups may be missing if OP.GG does not have data for them because they are low frequency.
    We use the winrates we can get to fill in the values in our dataframe."""
    for base_champion in win_rate_df.columns:
        url = MATCHUPS_URL_TEMPLATE.format(champion=get_url_champion_name(base_champion), role=role, patch=PATCH)
        html = requests.get(url).text
        sel = Selector(text=html)
        matchup_champions = sel.css("aside ul li div:nth-child(2) ::text").getall()
        matchup_winrates = sel.css("aside ul li div:nth-child(3) ::text").getall()
        matchup_winrates = [winrate for winrate in matchup_winrates if winrate != '%']
        for champion, winrate in zip(matchup_champions, matchup_winrates):
            if champion in win_rate_df.index:
                win_rate_df.at[champion, base_champion] = winrate
        
    return win_rate_df


def export_tsv(win_rate_df, role):
    filename = f"{role}.tsv"
    win_rate_df.to_csv(f'inputs/{filename}', sep="\t")


def scrape_single_role(role):
    champions = get_champions_for_role(role)
    df = create_empty_df(champions)
    df = mark_diagonal(df)
    df = get_matchup_win_rates_bulk(df, role)
    export_tsv(df, role)
    print(f"Scraped and saved data for role: {role}")


def scrape_all_roles():
    roles = ["top", "jungle", "mid", "adc", "support"]
    for role in roles:
        scrape_single_role(role)

if __name__ == "__main__":
    scrape_all_roles()
