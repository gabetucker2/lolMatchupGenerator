# Setup
import os
from typing import Dict, List, Optional

import functions


config_file_path = "config.json"
config = functions.load_config(config_file_path)

LANE_FILE_MAP: Dict[str, str] = {
    "top": "matchups_top_expansive.tsv",
    "jungle": "matchups_jungle_expansive.tsv",
    "mid": "matchups_mid_expansive.tsv",
    "adc": "matchups_adc_expansive.tsv",
    "support": "matchups_support_expansive.tsv",
}

LANE_ALIASES: Dict[str, str] = {
    "top": "top",
    "jungle": "jungle",
    "jg": "jungle",
    "mid": "mid",
    "middle": "mid",
    "adc": "adc",
    "bot": "adc",
    "bottom": "adc",
    "support": "support",
    "sup": "support",
}

LANE_LABELS: Dict[str, str] = {
    "top": "top",
    "jungle": "jungle/jg",
    "mid": "mid",
    "adc": "adc/bot",
    "support": "support/sup",
}


file_path = ""
current_lane = "top"
data = None
matchup_games_data = None
normalized_data = None
name_mapping: Dict[str, str] = {}


def normalize_lane_input(user_input: str) -> Optional[str]:
    token = user_input.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    return LANE_ALIASES.get(token)


def ensure_startup_defaults() -> None:
    lane_mode = config.get("lane_mode")
    if lane_mode is None:
        raise ValueError("config.json is missing required field `lane_mode`.")

    normalized_lane = normalize_lane_input(str(lane_mode))
    if normalized_lane is None:
        raise ValueError(
            "config.json has invalid `lane_mode`. "
            "Use one of: top, jungle/jg, mid, adc/bot, support/sup."
        )

    expected_file = LANE_FILE_MAP[normalized_lane]
    configured_file = config.get("input_file")
    if configured_file != expected_file:
        raise ValueError(
            "config.json has mismatched `input_file` for lane_mode "
            f"`{normalized_lane}`. Expected `{expected_file}`, got `{configured_file}`."
        )

    if not isinstance(config.get("current_champion_pool"), list):
        raise ValueError("config.json field `current_champion_pool` must be a list.")


def remap_current_pool_to_dataset(show_warning: bool = True) -> bool:
    raw_pool = config["current_champion_pool"]
    normalized_pool = [functions.normalize_champ_name(champ) for champ in raw_pool]
    mapped_pool: List[str] = []
    unmapped: List[str] = []

    for raw_name, normalized_name in zip(raw_pool, normalized_pool):
        mapped_name = name_mapping.get(normalized_name)
        if mapped_name is None:
            unmapped.append(raw_name)
            continue
        if mapped_name not in mapped_pool:
            mapped_pool.append(mapped_name)

    if show_warning and unmapped:
        print(
            "Warning: The following champs in your pool do not exist in this lane's dataset: "
            f"{unmapped}"
        )

    changed = raw_pool != mapped_pool
    config["current_champion_pool"] = mapped_pool
    return changed


def load_lane_data(lane_choice: str, persist: bool = False) -> bool:
    global current_lane
    global file_path
    global data
    global matchup_games_data
    global normalized_data
    global name_mapping

    lane = normalize_lane_input(lane_choice)
    if lane is None:
        raise ValueError("Invalid lane. Use top, jungle/jg, mid, adc/bot, or support/sup.")

    target_file = LANE_FILE_MAP[lane]
    target_path = os.path.join(os.getcwd(), target_file)
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Lane dataset not found: {target_file}. "
            "Run webscrape.py to generate this lane first."
        )

    loaded_data = functions.load_data(target_file)

    loaded_games = functions.load_matchup_games_data(target_file, loaded_data.index, loaded_data.columns)
    loaded_normalized_data = functions.normalize_data(loaded_data.copy())
    loaded_name_mapping = functions.create_name_mapping(loaded_data.columns.tolist())

    current_lane = lane
    file_path = target_file
    data = loaded_data
    matchup_games_data = loaded_games
    normalized_data = loaded_normalized_data
    name_mapping = loaded_name_mapping

    config["lane_mode"] = lane
    config["input_file"] = target_file
    pool_changed = remap_current_pool_to_dataset(show_warning=True)

    if persist or pool_changed:
        functions.save_config(config, config_file_path)

    return True


def map_input_to_index(user_input, options):
    """Convert a numeric string or a string to an index in the list of options."""
    normalized_input = user_input.strip().lower().replace("_", "-").replace(" ", "-")
    normalized_options = [
        option.strip().lower().replace("_", "-").replace(" ", "-")
        for option in options
    ]

    try:
        index = int(normalized_input) - 1
        if 0 <= index < len(options):
            return index
    except ValueError:
        pass

    if normalized_input in normalized_options:
        return normalized_options.index(normalized_input)

    return -1


def map_input_to_string(user_input, options):
    """Convert a numeric string or a string to a string from options, using an index."""
    index = map_input_to_index(user_input, options)
    if index != -1:
        return options[index]
    return None


ensure_startup_defaults()

try:
    load_lane_data(config["lane_mode"], persist=False)
except Exception as error:
    raise SystemExit(f"Unable to load the configured lane dataset: {error}") from error


# Command dictionary for dynamic help
commands = {
    "generate-pools": "Generate new champ pools",
    "analyze-champs": "Analyze champions with different metrics",
    "analyze-pool": "Analyze your current pool's performance",
    "champ-pick": "Analyze matchups for specific champions",
    "best-bans": "Find the best bans based on lowest winrates",
    "heatmap": "Display a correlation heatmap",
    "champ-pool": "Manage champ pool (view/add/remove/list all champs)",
    "settings": "Edit settings such as heatmap options, name, or lane mode",
    "help": "Show this help menu",
    "exit": "Exit the program",
}


def show_help():
    """Display available commands."""
    print(f"Active lane mode: {current_lane}")
    print(f"Input file: {file_path}")
    print("Available commands:")
    for i, (cmd, desc) in enumerate(commands.items(), 1):
        print(f"{i}. {cmd} - {desc}")


def analyze_champs():
    """Subcommands for analyzing champion statistics with filtering."""
    try:
        selected_data = functions.select_champs(
            data,
            config["current_champion_pool"],
            "Choose champion selection option for analyze-champs",
        )
    except ValueError as error:
        print(error)
        return
    selected_champs = [champ for champ in selected_data.columns if champ in selected_data.index]
    if not selected_champs:
        print("No overlapping champion labels were found after filtering.")
        return

    filtered_data = selected_data.reindex(index=selected_champs, columns=selected_champs)

    filtered_games_data = None
    if matchup_games_data is not None:
        filtered_games_data = matchup_games_data.reindex(index=filtered_data.index, columns=filtered_data.columns)
    error_baseline = filtered_data.attrs.get(functions.CHAMPION_BASELINE_ATTR)
    if error_baseline is None:
        error_baseline = functions.load_source_page_winrates(file_path, filtered_data.columns)
    if error_baseline is None:
        raise ValueError(
            "Champion baseline win rates are unavailable for analyze-champs. "
            "Expected baseline attrs or raw bulk matchup data with source champion win rates."
        )

    metric_subcommands = {
        "rmse": {
            "visualizer": functions.visualize_rmse,
            "requires_games": False,
            "description": "RMSE_i = sqrt(mean_j((wr_ij - b_i)^2)) over valid matchups.",
        },
        "rmedse": {
            "visualizer": functions.visualize_rmedse,
            "requires_games": False,
            "description": "RMEDSE_i = sqrt(median_j((wr_ij - b_i)^2)) over valid matchups.",
        },
        "mae": {
            "visualizer": functions.visualize_mae,
            "requires_games": False,
            "description": "MAE_i = mean_j(|wr_ij - b_i|) over valid matchups.",
        },
        "medae": {
            "visualizer": functions.visualize_medae,
            "requires_games": False,
            "description": "MedAE_i = median_j(|wr_ij - b_i|) over valid matchups.",
        },
        "rmse_games_weighed": {
            "visualizer": functions.visualize_rmse_games_weighed,
            "requires_games": True,
            "description": "Games-weighted RMSE using matchup-likelihood proxy weights w_ij.",
        },
        "rmedse_games_weighed": {
            "visualizer": functions.visualize_rmedse_games_weighed,
            "requires_games": True,
            "description": "Games-weighted RMEDSE using weighted median of squared errors.",
        },
        "mae_games_weighed": {
            "visualizer": functions.visualize_mae_games_weighed,
            "requires_games": True,
            "description": "WMAE_i = sum_j(w_ij*|wr_ij-b_i|)/sum_j(w_ij).",
        },
        "medae_games_weighed": {
            "visualizer": functions.visualize_medae_games_weighed,
            "requires_games": True,
            "description": "Games-weighted MedAE using weighted median of absolute errors.",
        },
        "rmse_games_weighed_equalized": {
            "visualizer": functions.visualize_rmse_games_weighed_equalized,
            "requires_games": True,
            "description": "Games-weighted RMSE with row-wise equalization w_hat_ij = w_ij/mean_i(w_ij).",
        },
        "rmedse_games_weighed_equalized": {
            "visualizer": functions.visualize_rmedse_games_weighed_equalized,
            "requires_games": True,
            "description": "Games-weighted RMEDSE with row-wise equalization w_hat_ij = w_ij/mean_i(w_ij).",
        },
        "mae_games_weighed_equalized": {
            "visualizer": functions.visualize_mae_games_weighed_equalized,
            "requires_games": True,
            "description": "MAE_GAMES_EQUALIZED_i = sum_j(w_hat_ij*|wr_ij-b_i|)/sum_j(w_hat_ij), row-wise equalized.",
        },
        "medae_games_weighed_equalized": {
            "visualizer": functions.visualize_medae_games_weighed_equalized,
            "requires_games": True,
            "description": "Games-weighted MedAE with row-wise equalization w_hat_ij = w_ij/mean_i(w_ij).",
        },
    }

    subcommands = [
        "std-devs",
        "mean-corrs",
        *metric_subcommands.keys(),
        "help",
        "back",
    ]
    subcommand_descriptions = {
        "std-devs": "View champs sorted by standard deviation of winrates",
        "mean-corrs": "View champs sorted by mean correlation with others",
    }
    for metric_name, metric_info in metric_subcommands.items():
        subcommand_descriptions[metric_name] = metric_info["description"]

    def print_analyze_champs_subcommands():
        print("Analyze Champs Subcommands:")
        for i, cmd in enumerate(subcommands, 1):
            if cmd in subcommand_descriptions:
                print(f"{i}. {cmd} - {subcommand_descriptions[cmd]}")
            else:
                print(f"{i}. {cmd}")

    while True:
        print_analyze_champs_subcommands()

        subcommand = input("Enter a subcommand (type 'help' to see list of commands again): ").strip().lower()
        mapped_subcommand = map_input_to_string(subcommand, subcommands)

        if mapped_subcommand == "std-devs":
            functions.print_break()
            functions.visualize_std_devs(filtered_data)

        elif mapped_subcommand == "mean-corrs":
            functions.print_break()
            functions.visualize_mean_corrs(filtered_data)

        elif mapped_subcommand in metric_subcommands:
            functions.print_break()
            metric_info = metric_subcommands[mapped_subcommand]
            try:
                if metric_info["requires_games"]:
                    metric_info["visualizer"](filtered_data, filtered_games_data, baseline=error_baseline)
                else:
                    metric_info["visualizer"](filtered_data, baseline=error_baseline)
            except ValueError as error:
                print(error)

        elif mapped_subcommand == "help":
            print_analyze_champs_subcommands()

        elif mapped_subcommand == "back":
            break
        else:
            print("Invalid subcommand. Use the listed options, or the corresponding numbers.")


def analyze_pool():
    """Subcommands for analyzing the current champ pool."""
    subcommands = [
        "current-max",
        "current-max-norm",
        "best-add",
        "best-add-norm",
        "worst-remove",
        "worst-remove-norm",
        "help",
        "back",
    ]

    while True:
        print("---------------------------------------------------------------")
        print("Analyze Pool Subcommands:")
        for i, cmd in enumerate(subcommands, 1):
            if cmd not in {"help", "back"}:
                print(f"{i}. {cmd} - Show {cmd.replace('-', ' ').replace('norm', 'normalized')} for the pool")
            else:
                print(f"{i}. {cmd}")

        subcommand = input("Enter a subcommand (type 'help' to see list of commands again): ").strip().lower()
        mapped_subcommand = map_input_to_string(subcommand, subcommands)

        if mapped_subcommand == "current-max":
            pool_stat = functions.calculate_pool_wr_stats(data, config["current_champion_pool"])
            if pool_stat:
                print(f"Current maximum winrate average for the pool: {pool_stat:.2f}")
            else:
                print("Current pool is empty or no data available.")

        elif mapped_subcommand == "current-max-norm":
            pool_stat = functions.calculate_pool_wr_stats(normalized_data, config["current_champion_pool"])
            if pool_stat:
                print(f"Normalized maximum winrate average for the pool: {pool_stat:.2f}")
            else:
                print("Current pool is empty or no data available.")

        elif mapped_subcommand == "best-add":
            recommendations = functions.calculate_best_champ_additions_with_ci(
                data,
                config["current_champion_pool"],
                confidence_level=0.95,
            )
            if recommendations:
                print("Champs that would increase the pool's max winrate average (most to least):")
                formatted_rows = []
                for i, (champ, value, ci_low, ci_high, _) in enumerate(recommendations, 1):
                    if value == value and ci_low == ci_low and ci_high == ci_high:
                        left_text = f"{i}. {champ.title()}: {value:.2f}%"
                        ci_text = f"({ci_low:.2f}% - {ci_high:.2f}%)"
                    else:
                        left_text = f"{i}. {champ.title()}: /"
                        ci_text = "(/ - /)"
                    formatted_rows.append((left_text, ci_text))

                ci_column = max(len(left_text) for left_text, _ in formatted_rows) + 6
                for left_text, ci_text in formatted_rows:
                    print(f"{left_text.ljust(ci_column)}{ci_text}")
            else:
                print("No champs to recommend.")

        elif mapped_subcommand == "best-add-norm":
            recommendations = functions.calculate_best_champ_additions_with_ci(
                normalized_data,
                config["current_champion_pool"],
                confidence_level=0.95,
            )
            if recommendations:
                print("Champs that would increase the pool's normalized max winrate average (most to least):")
                formatted_rows = []
                for i, (champ, value, ci_low, ci_high, _) in enumerate(recommendations, 1):
                    if value == value and ci_low == ci_low and ci_high == ci_high:
                        left_text = f"{i}. {champ.title()}: {value:.2f}%"
                        ci_text = f"({ci_low:.2f}% - {ci_high:.2f}%)"
                    else:
                        left_text = f"{i}. {champ.title()}: /"
                        ci_text = "(/ - /)"
                    formatted_rows.append((left_text, ci_text))

                ci_column = max(len(left_text) for left_text, _ in formatted_rows) + 6
                for left_text, ci_text in formatted_rows:
                    print(f"{left_text.ljust(ci_column)}{ci_text}")
            else:
                print("No champs to recommend.")

        elif mapped_subcommand == "worst-remove":
            removals = functions.calculate_worst_champ_removals(data, config["current_champion_pool"])
            if removals:
                print("Champs that would decrease the pool's max winrate average (most to least):")
                for i, (champ, value) in enumerate(removals, 1):
                    print(f"{i}. {champ.title()}: {value:.2f}")
            else:
                print("No champs to remove.")

        elif mapped_subcommand == "worst-remove-norm":
            removals = functions.calculate_worst_champ_removals(normalized_data, config["current_champion_pool"])
            if removals:
                print("Champs that would decrease the pool's normalized max winrate average (most to least):")
                for i, (champ, value) in enumerate(removals, 1):
                    print(f"{i}. {champ.title()}: {value:.2f}")
            else:
                print("No champs to remove.")

        elif mapped_subcommand == "help":
            for i, cmd in enumerate(subcommands, 1):
                if cmd not in {"help", "back"}:
                    print(f"{i}. {cmd} - Show {cmd.replace('-', ' ').replace('norm', 'normalized')} for the pool")
                else:
                    print(f"{i}. {cmd}")

        elif mapped_subcommand == "back":
            break

        else:
            print("Invalid subcommand. Use the listed options, or the corresponding numbers.")


def champ_pick():
    """Analyze matchups for specific input champions against your current pool."""
    input_champs = input("Enter the champions to analyze (separated by commas): ").strip().split(",")
    input_champs = [functions.normalize_champ_name(champ) for champ in input_champs]

    mapped_input_champs = [name_mapping.get(champ, None) for champ in input_champs]
    mapped_input_champs = [champ for champ in mapped_input_champs if champ is not None]

    unmapped_input_champs = [
        champ for champ in input_champs
        if champ not in name_mapping or name_mapping.get(champ, None) is None
    ]
    if unmapped_input_champs:
        print(f"Warning: The following input champs do not exist in the dataset: {', '.join(unmapped_input_champs)}")

    if not mapped_input_champs:
        print("No valid input champions provided. Returning to the main menu.")
        return

    try:
        table = functions.analyze_champ_picks(data, config["current_champion_pool"], mapped_input_champs)
        print(table)
    except ValueError as error:
        print(f"Error during analysis: {error}")


def parse_pool_input(raw_names: str) -> tuple[List[str], List[str]]:
    entries = [name.strip() for name in raw_names.split(",") if name.strip()]
    mapped_entries: List[str] = []
    invalid_entries: List[str] = []

    for entry in entries:
        mapped = name_mapping.get(functions.normalize_champ_name(entry))
        if mapped is None:
            invalid_entries.append(entry)
        else:
            mapped_entries.append(mapped)

    return mapped_entries, invalid_entries


def champ_pool_menu():
    """Manage champion pool contents from inside the script."""
    subcommands = [
        "view-current",
        "view-all",
        "add",
        "remove",
        "help",
        "back",
    ]
    descriptions = {
        "view-current": "Show your current champ pool",
        "view-all": "Show all potential champs from active lane dataset",
        "add": "Add champs to your current pool",
        "remove": "Remove champs from your current pool",
    }

    while True:
        print("Champion Pool Options:")
        for i, cmd in enumerate(subcommands, 1):
            if cmd in descriptions:
                print(f"{i}. {cmd} - {descriptions[cmd]}")
            else:
                print(f"{i}. {cmd}")

        subcommand = input("Enter an option: ").strip().lower()
        mapped_subcommand = map_input_to_string(subcommand, subcommands)

        if mapped_subcommand == "view-current":
            if config["current_champion_pool"]:
                print(f"Current pool ({len(config['current_champion_pool'])} champs):")
                for i, champ in enumerate(config["current_champion_pool"], 1):
                    print(f"{i}. {champ}")
            else:
                print("Current pool is empty.")

        elif mapped_subcommand == "view-all":
            all_champs = list(data.columns)
            print(f"All potential champs in {current_lane} lane dataset ({len(all_champs)} total):")
            for i, champ in enumerate(all_champs, 1):
                print(f"{i}. {champ}")

        elif mapped_subcommand == "add":
            raw_input = input("Enter champ(s) to add (comma separated): ").strip()
            mapped_champs, invalid_champs = parse_pool_input(raw_input)
            added: List[str] = []
            already_in_pool: List[str] = []

            for champ in mapped_champs:
                if champ in config["current_champion_pool"]:
                    already_in_pool.append(champ)
                else:
                    config["current_champion_pool"].append(champ)
                    added.append(champ)

            if added:
                functions.save_config(config, config_file_path)
                print(f"Added to pool: {', '.join(added)}")
            if already_in_pool:
                print(f"Already in pool: {', '.join(already_in_pool)}")
            if invalid_champs:
                print(f"Not found in current lane dataset: {', '.join(invalid_champs)}")
            if not added and not already_in_pool and not invalid_champs:
                print("No champs were provided.")

        elif mapped_subcommand == "remove":
            if not config["current_champion_pool"]:
                print("Current pool is empty.")
                continue

            raw_input = input("Enter champ(s) to remove (comma separated): ").strip()
            mapped_champs, invalid_champs = parse_pool_input(raw_input)
            removed: List[str] = []
            not_in_pool: List[str] = []

            for champ in mapped_champs:
                if champ in config["current_champion_pool"]:
                    config["current_champion_pool"].remove(champ)
                    removed.append(champ)
                else:
                    not_in_pool.append(champ)

            if removed:
                functions.save_config(config, config_file_path)
                print(f"Removed from pool: {', '.join(removed)}")
            if not_in_pool:
                print(f"Not in current pool: {', '.join(not_in_pool)}")
            if invalid_champs:
                print(f"Not found in current lane dataset: {', '.join(invalid_champs)}")
            if not removed and not not_in_pool and not invalid_champs:
                print("No champs were provided.")

        elif mapped_subcommand == "help":
            for i, cmd in enumerate(subcommands, 1):
                if cmd in descriptions:
                    print(f"{i}. {cmd} - {descriptions[cmd]}")
                else:
                    print(f"{i}. {cmd}")

        elif mapped_subcommand == "back":
            break

        else:
            print("Invalid subcommand. Use the listed options, or the corresponding numbers.")


def settings():
    """Subcommands for editing settings."""
    subcommands = [
        "edit-heatmap",
        "edit-name",
        "edit-lane",
        "help",
        "back",
    ]

    descriptions = {
        "edit-heatmap": "Edit heatmap settings",
        "edit-name": "Edit your display name",
        "edit-lane": "Change lane mode (top/jungle/jg/mid/adc/bot/support/sup)",
    }

    while True:
        print("Settings Subcommands:")
        for i, cmd in enumerate(subcommands, 1):
            if cmd in descriptions:
                print(f"{i}. {cmd} - {descriptions[cmd]}")
            else:
                print(f"{i}. {cmd}")

        subcommand = input("Enter a subcommand (type 'help' to see list of commands again): ").strip().lower()
        mapped_subcommand = map_input_to_string(subcommand, subcommands)

        if mapped_subcommand == "edit-heatmap":
            functions.print_break()
            if hasattr(functions, "edit_heatmap_settings"):
                functions.edit_heatmap_settings(config, config_file_path)
            else:
                print("Heatmap settings editor is not available in functions.py.")

        elif mapped_subcommand == "edit-name":
            functions.print_break()
            new_name = input("Enter your new name: ").strip()
            config["your_name"] = new_name
            functions.save_config(config, config_file_path)
            print(f"Your name has been updated to {new_name}.")
            functions.print_break()

        elif mapped_subcommand == "edit-lane":
            functions.print_break()
            print("Lane options:")
            print(f"Current lane: {current_lane}")
            print(f"top, jungle/jg, mid, adc/bot, support/sup")
            lane_input = input("Enter the lane mode to use: ").strip()
            target_lane = normalize_lane_input(lane_input)

            if target_lane is None:
                print("Invalid lane. Use top, jungle/jg, mid, adc/bot, or support/sup.")
            elif target_lane == current_lane:
                print(f"Lane mode is already set to {current_lane}.")
            else:
                try:
                    load_lane_data(target_lane, persist=True)
                    print(
                        f"Lane mode changed to {target_lane}. "
                        f"Now exclusively using {LANE_FILE_MAP[target_lane]}."
                    )
                except Exception as error:
                    print(f"Lane mode was not changed: {error}")
            functions.print_break()

        elif mapped_subcommand == "help":
            for i, cmd in enumerate(subcommands, 1):
                if cmd in descriptions:
                    print(f"{i}. {cmd} - {descriptions[cmd]}")
                else:
                    print(f"{i}. {cmd}")

        elif mapped_subcommand == "back":
            break

        else:
            print("Invalid subcommand. Use the listed options, or the corresponding numbers.")


def best_bans():
    """Calculate the best bans based on user input."""
    print("Choose an option:")
    print("1. Best bans for a specific champion")
    print("2. Best bans for your current champion pool")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice not in {"1", "2"}:
        print("Invalid choice. Returning to main menu.")
        return

    try:
        n_bans = int(input("Enter the number of bans to calculate: ").strip())
    except ValueError:
        print("Invalid number. Returning to main menu.")
        return

    if choice == "1":
        champ = input("Enter the champion name: ").strip()
        champ = functions.normalize_champ_name(champ)
        champ = name_mapping.get(champ)

        if not champ:
            print(f"Champion {champ} not found in the dataset.")
            return

        try:
            bans = functions.calculate_best_bans(data, champ, n_bans)
            print("Best bans for", champ.title(), ":")
            for i, (ban, winrate) in enumerate(bans.items(), 1):
                print(f"{i}. {ban.title()} with winrate {winrate:.2f}%")
        except ValueError as error:
            print(error)

    elif choice == "2":
        try:
            bans = functions.calculate_best_bans(data, config["current_champion_pool"], n_bans)
            print("Best bans for your champion pool:")
            for i, (ban, winrate) in enumerate(bans.items(), 1):
                print(f"{i}. {ban.title()} with average winrate {winrate:.2f}%")
        except ValueError as error:
            print(error)


# Intro message
functions.print_break()
print(f"Greetings, {config['your_name']}!")
print(f"Active lane mode: {current_lane} ({LANE_LABELS[current_lane]})")
functions.print_break()
show_help()

# Command handling loop
while True:
    functions.print_break()
    command_options = list(commands.keys())
    command = input("Enter a command (type 'help' to see list of commands again): ").strip().lower()

    mapped_command = map_input_to_string(command, command_options)
    if mapped_command == "help":
        show_help()
    elif mapped_command == "exit":
        functions.print_break()
        print("Goodbye!")
        functions.print_break()
        break
    elif mapped_command is None:
        print("Invalid command. Type 'help' to see available commands.")
    else:
        if mapped_command == "analyze-champs":
            functions.print_break()
            try:
                analyze_champs()
            except ValueError as error:
                print(error)
        elif mapped_command == "analyze-pool":
            analyze_pool()
        elif mapped_command == "champ-pick":
            functions.print_break()
            champ_pick()
        elif mapped_command == "best-bans":
            functions.print_break()
            best_bans()
        elif mapped_command == "champ-pool":
            functions.print_break()
            champ_pool_menu()
        elif mapped_command == "generate-pools":
            functions.print_break()
            try:
                filtered_data = functions.select_champs(
                    data,
                    config["current_champion_pool"],
                    "Choose champion selection option for generate-pools",
                )
            except ValueError as error:
                print(error)
                continue
            num_champs = int(input("Enter the number of champs per pool: ").strip())
            top_pools = int(input("Enter the number of top pools to display: ").strip())
            max_global_appearances = int(
                input("Enter the max appearances of any champ across all pools: ").strip()
            )
            valid_combinations = functions.generate_pool_stats(
                filtered_data.corr(),
                num_champs,
                top_pools,
                max_global_appearances,
            )
            print("Best pools:")
            for i, (pool, score) in enumerate(valid_combinations, 1):
                print(f"Pool {i}: {pool} (Average Correlation: {score})")
        elif mapped_command == "heatmap":
            try:
                filtered_data = functions.select_champs(
                    data,
                    config["current_champion_pool"],
                    "Choose champion selection option for heatmap",
                )
            except ValueError as error:
                print(error)
                continue
            functions.generate_heatmap(filtered_data.corr(), config["heatmap_settings"])
        elif mapped_command == "settings":
            functions.print_break()
            settings()
