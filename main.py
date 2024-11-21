import requests
import pandas as pd
import random

# Data Dragon Base URL for champion data
DATA_DRAGON_URL = "https://ddragon.leagueoflegends.com/cdn/13.20.1/data/en_US/champion.json"

# Champion list (static for consistency in matchup table rows)
champions = [
    "Aatrox", "Akali", "Akshan", "Ambessa", "Aurora", "Camille", "Cassiopeia",
    "Cho'Gath", "Darius", "Dr. Mundo", "Fiora", "Gangplank",
    "Garen", "Gnar", "Gragas", "Gwen", "Heimerdinger",
    "Illaoi", "Irelia", "Jax", "Jayce", "K'Sante", "Kayle",
    "Kennen", "Kled", "Lillia", "Malphite", "Maokai",
    "Mordekaiser", "Nasus", "Nocturne", "Olaf", "Ornn",
    "Pantheon", "Poppy", "Quinn", "Renekton", "Riven",
    "Ryze", "Sett", "Shen", "Singed", "Sion", "Smolder", "Swain",
    "Sylas", "Tahm Kench", "Teemo", "Trundle", "Tryndamere",
    "Udyr", "Urgot", "Varus", "Vayne", "Vladimir", "Volibear",
    "Warwick", "Wukong", "Yasuo", "Yone", "Yorick", "Zac"
]

# Specify the "as X" champions to include in the output
selected_champions = ["Jax", "Ornn", "Gwen", "Darius", "Singed"]

def get_champion_ids():
    """Fetch champion IDs from Data Dragon."""
    print(f"Requesting champion data from Data Dragon: {DATA_DRAGON_URL}")
    response = requests.get(DATA_DRAGON_URL)
    if response.status_code != 200:
        print(f"Failed to fetch champion data: HTTP {response.status_code}")
        print(f"Response content: {response.text[:500]}")
        return {}

    try:
        data = response.json()
        print("Successfully fetched champion data.")
        return {champ['id']: int(champ['key']) for champ in data['data'].values()}
    except KeyError as e:
        print(f"Error parsing champion ID data: {e}")
        print(f"Response content: {response.text[:500]}")
        return {}

def generate_matchup_win_rates():
    """Simulate win rates and sample sizes."""
    matchup_data = {}
    for champion in champions:
        matchup_data[champion] = {}
        for opponent in champions:
            if champion == opponent:
                matchup_data[champion][opponent] = "/"  # Same champion matchups are not relevant
            else:
                sample_size = random.randint(50, 500)  # Random sample size for demonstration
                win_rate = random.uniform(40, 60)  # Simulated win rate between 40% and 60%
                matchup_data[champion][opponent] = f"{win_rate:.1f}% ({sample_size})"
    return matchup_data

def main():
    print("Fetching champion IDs...")
    champion_ids = get_champion_ids()
    if not champion_ids:
        print("Failed to retrieve champion IDs.")
        return

    print("Generating matchup win rates...")
    matchup_data = generate_matchup_win_rates()

    # Format data for the DataFrame
    data = {"VS": champions}
    for champion in selected_champions:
        print(f"Processing data for {champion}...")
        if champion not in matchup_data:
            print(f"No data available for {champion}.")
            data[f"as {champion}"] = ["/" for _ in champions]
            continue

        # Map matchup data for each opponent
        row = [matchup_data[champion].get(opponent, "/") for opponent in champions]
        data[f"as {champion}"] = row

    # Create a DataFrame and filter columns to include only selected champions
    selected_columns = ["VS"] + [f"as {champion}" for champion in selected_champions]
    df = pd.DataFrame(data)[selected_columns]

    # Save the DataFrame to a CSV file
    file_path = "league_matchup_win_rates.csv"
    df.to_csv(file_path, index=False)
    print(f"Matchup win rates saved to {file_path}")

if __name__ == "__main__":
    main()
