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
adjust_winrate_baseline = config["adjust_winrate_baseline"]
num_champions_per_pool = config["num_champions_per_pool"]
top_pools_to_display = config["top_pools_to_display"]
heatmap_settings = config["heatmap_settings"]
exclude_columns = config.get("exclude_columns", [])  # New variable for columns to exclude

# Load the data from the TSV file specified in the config
updated_data = pd.read_csv(os.path.join(os.getcwd(), file_path), sep='\t', index_col=0)

# Replace non-numeric values ("/") with NaN
updated_data.replace("/", pd.NA, inplace=True)

# Convert data to numeric
updated_data = updated_data.apply(pd.to_numeric)

# Drop columns specified in the exclude_columns variable
if exclude_columns:
    updated_data.drop(columns=exclude_columns, inplace=True, errors='ignore')

# Create a copy of the dataset for adjustments
adjusted_data = updated_data.copy()

# Iterate over each column to adjust win rates
for col in adjusted_data.columns:
    # Calculate the column average
    col_average = adjusted_data[col].mean()
    
    # Calculate the adjustment value
    adjustment = adjust_winrate_baseline - col_average
    
    # Apply the adjustment to all elements in the column
    adjusted_data[col] += adjustment

# Calculate the correlation matrix for the adjusted data
adjusted_correlation_matrix = adjusted_data.corr()

# Calculate the total correlation coefficient for each champion
total_correlation = adjusted_correlation_matrix.sum()

# Find the best champion pools
# Generate all combinations of champions based on the config
champion_combinations = combinations(adjusted_correlation_matrix.columns, num_champions_per_pool)

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

if heatmap_settings["visualize"]:

    # Create a heatmap of the adjusted correlation matrix
    plt.figure(figsize=tuple(heatmap_settings["figsize"]))
    sns.heatmap(
        adjusted_correlation_matrix,
        annot=heatmap_settings["annot"],
        cmap=heatmap_settings["cmap"],
        linewidths=heatmap_settings["linewidths"],
        fmt=heatmap_settings["fmt"],
        annot_kws={"size": 6}
    )
    plt.title(heatmap_settings["title"])
    plt.show()
