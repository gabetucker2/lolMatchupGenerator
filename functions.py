import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import re
import numpy as np

def print_break():
    """Print a standardized separator for console output."""
    print("---------------------------------------------------------------")

# Configuration Management
def load_config(config_file_path):
    """Load configuration from the specified JSON file."""
    with open(config_file_path, "r") as config_file:
        return json.load(config_file)

def save_config(config, config_file_path):
    """Save configuration back to the specified JSON file."""
    with open(config_file_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

# Data Loading and Normalization
def load_data(file_path):
    """Load data from a TSV file and handle missing or invalid entries."""
    data = pd.read_csv(os.path.join(os.getcwd(), file_path), sep='\t', index_col=0)
    data.replace("/", pd.NA, inplace=True)
    return data.apply(pd.to_numeric)

def normalize_champ_name(name):
    """Normalize champion names for consistent matching."""
    import unicodedata
    # Strip accents, remove special characters, and lowercase
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    name = re.sub(r"[^\w\s]", "", name)  # Remove punctuation
    name = re.sub(r"\s+", " ", name)  # Normalize spaces
    return name.strip().lower()

def create_name_mapping(columns):
    """Create a mapping from normalized names to original column names."""
    mapping = {normalize_champ_name(col): col for col in columns}
    if len(mapping) != len(columns):
        print("Warning: Some champion names may have duplicate normalized forms!")
    return mapping

# Data Analysis
def calculate_champ_std_devs(data):
    """Calculate the standard deviation of winrates for each champion."""
    return data.std(axis=0).sort_values(ascending=False)

def calculate_champ_mean_corrs(data):
    """Calculate the mean correlation of each champion with others."""
    return data.corr().mean(axis=0).sort_values(ascending=False)

def normalize_data(data):
    """Normalize win rates to have a mean of 50 and preserve relative variability."""
    normalized_data = data.copy()
    col_means = data.mean()

    normalized_data = (data - col_means) + 50

    return normalized_data

def visualize_std_devs(data):
    """Display champions sorted by the standard deviation of their winrates."""
    champ_std_devs = calculate_champ_std_devs(data)
    print("Champs sorted by average winrate standard deviation:")
    for i, (champ, std_dev) in enumerate(champ_std_devs.items(), start=1):
        print(f"{i}. {champ.title()}: {std_dev:.2f}")

def visualize_mean_corrs(data):
    """Display champions sorted by their mean correlation with others."""
    champ_means = calculate_champ_mean_corrs(data)
    print("Champs sorted by average correlation with others in terms of which champs they counter:")
    for i, (champ, mean_corr) in enumerate(champ_means.items(), start=1):
        print(f"{i}. {champ.title()}: {mean_corr:.2f}")

# Pool Analysis
def calculate_pool_wr_stats(data, current_pool):
    """Calculate the maximum average winrate for the current champion pool."""
    if current_pool:
        pool_data = data[current_pool]
        return pool_data.max(axis=1).mean()
    return None

def calculate_best_champ_additions(data, current_pool):
    """Identify champions that would most improve the pool's max winrate."""
    name_mapping = create_name_mapping(data.columns)
    current_pool = [name_mapping.get(normalize_champ_name(champ), None) for champ in current_pool]
    current_pool = [champ for champ in current_pool if champ is not None]

    remaining_champs = [champ for champ in data.columns if champ not in current_pool]
    if not remaining_champs:
        return []

    champ_addition_impact = {}
    for champ in remaining_champs:
        temp_pool = current_pool + [champ]
        temp_data = data[temp_pool]
        champ_addition_impact[champ] = temp_data.max(axis=1).mean()

    return sorted(champ_addition_impact.items(), key=lambda x: x[1], reverse=True)

def calculate_worst_champ_removals(data, current_pool):
    """Identify champions whose removal would least impact the pool's max winrate."""
    champ_removal_impact = {}

    for champ in current_pool:
        temp_pool = [c for c in current_pool if c != champ]
        if not temp_pool:
            continue
        temp_data = data[temp_pool]
        champ_removal_impact[champ] = temp_data.max(axis=1).mean()

    return sorted(champ_removal_impact.items(), key=lambda x: x[1])

def validate_input_champs(data, input_champs, name_mapping):
    """Validate and normalize input champions."""
    normalized_input = [normalize_champ_name(champ) for champ in input_champs]
    valid_champs = [name_mapping.get(champ) for champ in normalized_input if champ in name_mapping]
    invalid_champs = [champ for champ, mapped in zip(input_champs, valid_champs) if mapped is None]
    return valid_champs, invalid_champs

def analyze_champ_picks(data, current_pool, input_champs):
    """Analyze matchups for the provided input champions against the current pool."""
    
    # Normalize the dataset
    normalized_data = normalize_data(data)

    # Ensure input champions exist in the dataset
    valid_input_champs = [champ for champ in input_champs if champ in data.index]

    # Warn if no valid input champions are found
    if not valid_input_champs:
        raise ValueError("None of the provided input champions exist in the dataset.")

    # Results storage
    results_raw = {}
    results_normalized = {}

    # Collect win rates
    for pool_champ in current_pool:
        winrates_raw = [
            data.loc[input_champ, pool_champ]
            for input_champ in valid_input_champs
            if input_champ in data.index and pool_champ in data.columns
        ]
        winrates_normalized = [
            normalized_data.loc[input_champ, pool_champ]
            for input_champ in valid_input_champs
            if input_champ in normalized_data.index and pool_champ in normalized_data.columns
        ]
        results_raw[pool_champ] = winrates_raw
        results_normalized[pool_champ] = winrates_normalized

    # Calculate averages for normalized and non-normalized
    averages_raw = {
        champ: sum(winrates) / len(winrates) if winrates else 0
        for champ, winrates in results_raw.items()
    }
    averages_normalized = {
        champ: sum(winrates) / len(winrates) if winrates else 0
        for champ, winrates in results_normalized.items()
    }

    # Determine best champs
    best_raw = max(averages_raw, key=averages_raw.get, default="/")
    best_normalized = max(averages_normalized, key=averages_normalized.get, default="/")

    # Build the output table
    table = "+" + "-" * 160 + "+\n"
    table += "| {:<15} |".format("Champion")
    for champ in current_pool:
        table += " {:<15} |".format(champ)
    table += " {:<20} | {:<20} |\n".format("Best Raw Champ", "Best Norm Champ")
    table += "+" + "-" * 160 + "+\n"

    for input_champ in valid_input_champs:
        row = "| {:<15} |".format(input_champ.title())
        for champ in current_pool:
            if input_champ in data.index and champ in data.columns:
                row += " {:<15} |".format(f"{data.loc[input_champ, champ]:.2f}->{normalized_data.loc[input_champ, champ]:.2f}")
            else:
                row += " {:<15} |".format("/")
        best_non_norm = max(current_pool, key=lambda x: data.loc[input_champ, x] if input_champ in data.index and x in data.columns else 0, default='/')
        best_norm = max(current_pool, key=lambda x: normalized_data.loc[input_champ, x] if input_champ in normalized_data.index and x in normalized_data.columns else 0, default='/')
        row += " {:<20} | {:<20} |\n".format(best_non_norm, best_norm)
        table += row

    # Add minimum non-norm row
    minRow = "| {:<15} |".format("Raw Min")
    for champ in current_pool:
        min_val = data[champ].min()
        minRow += " {:<15.2f} |".format(min_val)
    minRow += " {:<20} | {:<20} |\n".format(best_raw, "/")
    table += minRow

    # Add minimum norm row
    minRow = "| {:<15} |".format("Norm Min")
    for champ in current_pool:
        min_val = normalized_data[champ].min()
        minRow += " {:<15.2f} |".format(min_val)
    minRow += " {:<20} | {:<20} |\n".format("/", best_normalized)
    table += minRow

    table += "+" + "-" * 160 + "+\n"

    table = table.replace('nan->nan', '/       ')

    return table

# Visualization
def generate_heatmap(correlation_matrix, settings):
    """Generate a heatmap from the correlation matrix using provided settings."""
    figsize = settings.get("figsize", (10, 8))
    annot = settings.get("annot", True)
    cmap = settings.get("cmap", "coolwarm")
    linewidths = settings.get("linewidths", 0.5)
    fmt = settings.get("fmt", ".2f")
    title = settings.get("title", "Heatmap")
    textsize = settings.get("textsize", 10)

    correlation_matrix.index = [col.title() for col in correlation_matrix.index]
    correlation_matrix.columns = [col.title() for col in correlation_matrix.columns]

    if not correlation_matrix.empty and not correlation_matrix.isna().all().all():
        plt.figure(figsize=figsize)
        sns.heatmap(
            correlation_matrix,
            annot=annot,
            cmap=cmap,
            linewidths=linewidths,
            fmt=fmt,
            annot_kws={"size": textsize}
        )
        plt.title(title)
        plt.show()
    else:
        print("Correlation matrix is empty or contains only NaN values. Heatmap cannot be generated.")

# Champ selection
def select_champs(data, current_pool, prompt="Choose champion selection option"):
    """Prompt user to select champions to include or exclude for analysis."""
    print(prompt)
    print("1. All champions")
    print("2. Current pool")
    print("3. Only include specific champions")
    print("4. Exclude specific champions")

    option = input("Enter your choice (1-4): ").strip()

    if option == "1":
        return data
    elif option == "2":
        return data[current_pool]
    elif option == "3":
        champs_to_include = input("Enter champions to include, separated by commas: ").strip().split(",")
        champs_to_include = [normalize_champ_name(champ) for champ in champs_to_include]
        included_champs = [col for col in data.columns if normalize_champ_name(col) in champs_to_include]
        return data[included_champs]
    elif option == "4":
        champs_to_exclude = input("Enter champions to exclude, separated by commas: ").strip().split(",")
        champs_to_exclude = [normalize_champ_name(champ) for champ in champs_to_exclude]
        remaining_champs = [col for col in data.columns if normalize_champ_name(col) not in champs_to_exclude]
        return data[remaining_champs]
    else:
        print("Invalid choice. Defaulting to all champions.")
        return data

def generate_pool_stats(correlation_matrix, num_champs, top_pools, max_global_appearances):
    """Generate pools of champions with the lowest average correlation."""

    # Extract champion names from the correlation matrix
    champ_names = correlation_matrix.columns.tolist()

    # Generate all combinations of `num_champs`
    all_combinations = list(combinations(champ_names, num_champs))

    # Calculate the average correlation for each combination
    pool_scores = []
    for combo in all_combinations:
        # Extract the sub-matrix for the current combination
        sub_matrix = correlation_matrix.loc[combo, combo]

        # Calculate the average correlation (excluding self-correlations)
        avg_correlation = np.mean(sub_matrix.values[np.triu_indices(len(combo), k=1)])
        pool_scores.append((combo, avg_correlation))

    # Sort pools by lowest average correlation
    pool_scores = sorted(pool_scores, key=lambda x: x[1])

    # Enforce the global max appearances constraint
    champ_global_count = {champ: 0 for champ in champ_names}
    valid_pools = []

    for pool, score in pool_scores:
        if all(champ_global_count[champ] < max_global_appearances for champ in pool):
            valid_pools.append((pool, score))
            for champ in pool:
                champ_global_count[champ] += 1
        if len(valid_pools) >= top_pools:
            break

    # Clean up the output to make it user-friendly
    clean_pools = [
        (", ".join(pool), round(score, 4)) for pool, score in valid_pools
    ]

    return clean_pools
