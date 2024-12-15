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
only_include_champions = config.get("only_include_champions", [])
do_exclude_champions = config.get("do_exclude_champions", False)
do_only_include_champions = config.get("do_only_include_champions", False)

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
else:
    # Compute the correlation matrix using the full dataset
    full_correlation_matrix = updated_data.corr()

    # Extract the filtered champions for final output
    final_correlation_matrix = full_correlation_matrix.loc[filtered_columns, filtered_columns]

    # Calculate the total correlation coefficient for each champion
    total_correlation = final_correlation_matrix.sum()

    # Generate all combinations of champions based on the config
    champion_combinations = combinations(final_correlation_matrix.columns, num_champions_per_pool)

    # Calculate the combined correlation for each combination
    combination_scores = {}
    for combo in champion_combinations:
        combo_score = sum(total_correlation[list(combo)])
        combination_scores[combo] = combo_score

    # Sort combinations by their total correlation score (lowest to highest)
    sorted_combinations = sorted(combination_scores.items(), key=lambda x: x[1])

    # Extract the top pools as specified in the config
    best_pools = sorted_combinations[:top_pools_to_display]

    # Display the best pools
    print(f"Top {top_pools_to_display} Champion Pools with the Lowest Total Correlation Coefficients:")
    for i, (combo, score) in enumerate(best_pools, 1):
        print(f"Pool {i}: {combo} with total correlation score: {score}")

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
