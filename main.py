import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Load configuration from the JSON file
config_file_path = "config.json"
with open(config_file_path, "r") as config_file:
    config = json.load(config_file)

# Extract values from the configuration
file_path = config["input_file"]
num_champions_per_pool = config["num_champions_per_pool"]
top_pools_to_display = config["top_pools_to_display"]
heatmap_settings = config["heatmap_settings"]
exclude_champions = config.get("exclude_champions", [])
current_champion_pool = config.get("current_champion_pool", [])
only_include_champions = config.get("only_include_champions", [])
visualize_general_stats = config.get("visualize_general_stats", False)
visualize_pool_wr_stats = config.get("visualize_pool_wr_stats", False)
visualize_pool_generator_stats = config.get("visualize_pool_generator_stats", False)
do_exclude_champions = config.get("do_exclude_champions", False)
do_only_include_champions = config.get("do_only_include_champions", False)
pool_appearance_limit = config["pool_appearance_limit"]

# Load the data from the TSV file specified in the config
updated_data = pd.read_csv(os.path.join(os.getcwd(), file_path), sep='\t', index_col=0)

# Replace non-numeric values ("/") with NaN
updated_data.replace("/", pd.NA, inplace=True)

# Convert data to numeric
updated_data = updated_data.apply(pd.to_numeric)

# Filtering logic based on configuration flags
filtered_columns = updated_data.columns.tolist()

# Implement column-based filtering logic
if do_only_include_champions and only_include_champions:
    # Keep only valid champions in the dataset
    filtered_columns = [champ for champ in only_include_champions if champ in updated_data.columns]
elif do_exclude_champions and exclude_champions:
    # Exclude specified champions if they exist
    filtered_columns = [champ for champ in updated_data.columns if champ not in exclude_champions]

# Check if filtered columns exist
if len(filtered_columns) == 0:
    print("No valid columns available after filtering. Please check your configuration.")
    sys.exit()

# Compute the correlation matrix using the full dataset
full_correlation_matrix = updated_data.corr()

if visualize_general_stats:

    # Calculate the standard deviation of winrates for each champion
    champ_std_devs = updated_data.std(axis=0)

    # Sort champions by standard deviation in descending order
    sorted_champ_std_devs = champ_std_devs.sort_values(ascending=False)

    # Display the champions sorted by standard deviation
    print("Champs sorted by average winrate standard deviation:")
    for i, (champ, std_dev) in enumerate(sorted_champ_std_devs.items(), start=1):
        print(f"{i}. {champ}: {std_dev}")

    print("---------------------------------------------------------------")

    # Calculate the mean of winrate correlations for each champion
    champ_means = full_correlation_matrix.mean(axis=0)

    # Sort champions by mean correlation in descending order
    sorted_champ_means = champ_means.sort_values(ascending=False)

    # Display the champions sorted by mean correlation
    print("Champs sorted by average correlation with others:")
    for i, (champ, mean_corr) in enumerate(sorted_champ_means.items(), start=1):
        print(f"{i}. {champ}: {mean_corr}")

    print("---------------------------------------------------------------")

if visualize_pool_wr_stats:

    # Create a list of champions NOT in current_champion_pool
    remaining_champions = [champ for champ in updated_data.columns if champ not in current_champion_pool]

    # Calculate the initial average maximum winrate of the current champ pool
    if current_champion_pool:
        current_pool_data = updated_data[current_champion_pool]
        current_pool_max_values = current_pool_data.max(axis=1)
        initial_avg_max_winrate = current_pool_max_values.mean()
        print(f"Initial average maximum winrate of the current champ pool: {initial_avg_max_winrate}")
    else:
        print("Current champ pool is empty. Cannot calculate initial average maximum winrate.")

    print("---------------------------------------------------------------")

    # Dictionary to store the average maximum winrate for each remaining champion
    champion_avg_max_winrates = {}

    # Iterate through each champion in the remaining list
    for champ in remaining_champions:
        # Create a temporary pool including the current champion pool plus the iterated champion
        temp_pool = current_champion_pool + [champ]

        # Extract relevant data from the updated_data DataFrame for the temporary pool
        temp_data = updated_data[temp_pool]

        # Calculate the maximum value in each row (max winrate of the pool plus the iterated champion)
        max_values_per_row = temp_data.max(axis=1)

        # Calculate the average of these maximum values
        avg_max_winrate = max_values_per_row.mean()

        # Assign the average value to the iterated champion
        champion_avg_max_winrates[champ] = avg_max_winrate

    # Sort the champions by their average maximum winrate in descending order
    sorted_avg_max_winrates = sorted(champion_avg_max_winrates.items(), key=lambda x: x[1], reverse=True)

    # Print the champions sorted by their average maximum winrate
    print("Champs sorted by average maximum winrate when added to the current pool:")
    for i, (champ, avg_winrate) in enumerate(sorted_avg_max_winrates, start=1):
        print(f"{i}. {champ}: {avg_winrate}")

    print("---------------------------------------------------------------")

    # Dictionary to store the best pool winrate without each champion from the current pool
    champion_removed_winrates = {}

    # Iterate through each champion in the current pool
    for champ in current_champion_pool:
        # Create a temporary pool excluding the iterated champion
        temp_pool = [c for c in current_champion_pool if c != champ]

        # Extract relevant data from the updated_data DataFrame for the temporary pool
        temp_data = updated_data[temp_pool]

        # Calculate the maximum value in each row (max winrate of the remaining pool)
        max_values_per_row = temp_data.max(axis=1)

        # Calculate the average of these maximum values
        avg_max_winrate = max_values_per_row.mean()

        # Assign the average value to the iterated champion as the winrate without them
        champion_removed_winrates[champ] = avg_max_winrate

    # Sort the champions by their winrate without them in descending order
    sorted_removed_winrates = sorted(champion_removed_winrates.items(), key=lambda x: x[1], reverse=True)

    # Print the champions sorted by their winrate without them
    print("Champs sorted by best pool winrate when removed from the current pool:")
    for i, (champ, winrate) in enumerate(sorted_removed_winrates, start=1):
        print(f"{i}. {champ}: {winrate}")

    print("---------------------------------------------------------------")

# Extract the filtered champions for final output
final_correlation_matrix = full_correlation_matrix.loc[filtered_columns, filtered_columns]

if visualize_pool_generator_stats:

    # Calculate the total correlation coefficient for each champion
    total_correlation = final_correlation_matrix.sum()

    # Generate all combinations of champions based on the config
    champion_combinations = combinations(final_correlation_matrix.columns, num_champions_per_pool)

    # Ensure `pool_appearance_dict` is initialized properly
    pool_appearance_dict = {champ: 0 for champ in final_correlation_matrix.columns}

    # Calculate the combined correlation for each combination
    combination_scores = {}
    for combo in champion_combinations:
        combo_score = sum(total_correlation[list(combo)])
        combination_scores[combo] = combo_score

    # Sort combinations by their total correlation score (lowest to highest)
    sorted_combinations = sorted(combination_scores.items(), key=lambda x: x[1])

    # Implement champion appearance restriction
    valid_combinations = []

    for combo, score in sorted_combinations:
        # Check if all champions in the combo meet the restriction based on `pool_appearance_limit`
        if all(pool_appearance_dict.get(champ, 0) < pool_appearance_limit for champ in combo):
            # Temporarily update counts to reserve these champions
            valid = True
            for champ in combo:
                if pool_appearance_dict[champ] + 1 > pool_appearance_limit:
                    valid = False
                    break

            # If valid, add the combination and update the dictionary
            if valid:
                for champ in combo:
                    pool_appearance_dict[champ] += 1
                valid_combinations.append((combo, score))
            
            # Stop if enough valid combinations have been found
            if len(valid_combinations) >= top_pools_to_display:
                break

    # Display the best pools with restricted appearances
    print(f"Top {top_pools_to_display} champ pools with the lowest total correlation coefficients with {pool_appearance_limit} max appearances of the same champ between champ pools:")
    for i, (combo, score) in enumerate(valid_combinations, 1):
        print(f"Pool {i}: {combo} with total correlation score: {score}")

    print("---------------------------------------------------------------")

if heatmap_settings["visualize"]:

    # Generate heatmap if data is valid
    if not final_correlation_matrix.empty and not final_correlation_matrix.isna().all().all():
        plt.figure(figsize=tuple(heatmap_settings["figsize"]))
        sns.heatmap(
            final_correlation_matrix,
            annot=heatmap_settings["annot"],
            cmap=heatmap_settings["cmap"],
            linewidths=heatmap_settings["linewidths"],
            fmt=heatmap_settings["fmt"],
            annot_kws={"size": heatmap_settings["textsize"]}
        )
        plt.title(heatmap_settings["title"])
        plt.show()
    else:
        print("Correlation matrix is empty or contains only NaN values. Heatmap cannot be generated.")

# Wait for user input before closing
input("Press Enter to exit...")
