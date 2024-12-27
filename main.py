# Setup
import functions

# Load configuration
config_file_path = "config.json"
config = functions.load_config(config_file_path)

# Load data once
file_path = config["input_file"]
data = functions.load_data(file_path)

# Normalize data immediately after loading.
normalized_data = functions.normalize_data(data.copy())

# Create name mapping
name_mapping = functions.create_name_mapping(data.columns.tolist())

# Normalize the current champion pool
normalized_pool = [functions.normalize_champ_name(champ) for champ in config["current_champion_pool"]]
mapped_pool = [name_mapping.get(champ, None) for champ in normalized_pool]
mapped_pool = [champ for champ in mapped_pool if champ is not None]

# Warn about unmapped champs
unmapped_champs = [champ for champ in config["current_champion_pool"] if functions.normalize_champ_name(champ) not in name_mapping]
if unmapped_champs:
    print(f"Warning: The following champs in your pool do not exist in the dataset: {unmapped_champs}")

# Update the pool with normalized names
config["current_champion_pool"] = mapped_pool

# Intro message
functions.print_break()
print(f"Greetings, {config['your_name']}!")

# Command dictionary for dynamic help
commands = {
    "analyze-champs": "Analyze champions with different metrics",
    "analyze-pool": "Analyze your current pool's performance",
    "champ-pick": "Analyze matchups for specific champions",
    "generate-pools": "Generate new champ pools",
    "heatmap": "Display a correlation heatmap",
    "settings": "Edit settings such as heatmap options, name, file, or champ pool",
    "help": "Show this help menu",
    "exit": "Exit the program",
}

def map_input_to_index(user_input, options):
    """Convert a numeric string or a string to an index in the list of options."""
    try:
        index = int(user_input) - 1
        if 0 <= index < len(options):
            return index
    except ValueError:
        pass  # If not a number, it could be a string

    if user_input in options:
        return options.index(user_input)

    return -1

def map_input_to_string(user_input, options):
    """Convert a numeric string or a string to a string from options, using an index."""
    index = map_input_to_index(user_input, options)
    if index != -1:
        return options[index]
    else:
        return None

# Help menu
def show_help():
    """Display available commands."""
    print("Available commands:")
    for i, (cmd, desc) in enumerate(commands.items(), 1):
        print(f"{i}. {cmd} - {desc}")

def analyze_champs():
    """Subcommands for analyzing champion statistics with filtering."""
    filtered_data = functions.select_champs(data, config["current_champion_pool"], "Choose champion selection option for analyze-champs")

    subcommands = ["std-devs", "mean-corrs", "help", "back"]

    while True:
        print("Analyze Champs Subcommands:")
        for i, cmd in enumerate(subcommands, 1):
            if cmd != "help" and cmd != "back":
                print(f"{i}. {cmd} - View champs sorted by {cmd.replace('-', ' ').replace('std devs', 'standard deviation of winrates').replace('mean corrs', 'mean correlation with others')}")
            else:
                 print(f"{i}. {cmd}")

        subcommand = input("Enter a subcommand (type 'help' to see list of commands again): ").strip().lower()
        mapped_subcommand = map_input_to_string(subcommand, subcommands)

        if mapped_subcommand == "std-devs":
            functions.print_break()
            functions.visualize_std_devs(filtered_data)

        elif mapped_subcommand == "mean-corrs":
            functions.print_break()
            functions.visualize_mean_corrs(filtered_data)

        elif mapped_subcommand == "help":
             for i, cmd in enumerate(subcommands, 1):
                if cmd != "help" and cmd != "back":
                    print(f"{i}. {cmd} - View champs sorted by {cmd.replace('-', ' ').replace('std devs', 'standard deviation of winrates').replace('mean corrs', 'mean correlation with others')}")
                else:
                    print(f"{i}. {cmd}")

        elif mapped_subcommand == "back":
             break
        else:
            print(f"Invalid subcommand. Use the listed options, or the corresponding numbers.")

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
        # Display subcommand options once per loop iteration
        print("---------------------------------------------------------------")
        print("Analyze Pool Subcommands:")
        for i, cmd in enumerate(subcommands, 1):
            if cmd != "help" and cmd != "back":
                print(f"{i}. {cmd} - Show {cmd.replace('-', ' ').replace('norm', 'normalized')} for the pool")
            else:
                print(f"{i}. {cmd}")

        # Get user input for subcommand
        subcommand = input("Enter a subcommand (type 'help' to see list of commands again): ").strip().lower()
        mapped_subcommand = map_input_to_string(subcommand, subcommands)

        if mapped_subcommand == "current-max":
            pool_stat = functions.calculate_pool_wr_stats(data, config["current_champion_pool"]) # Changed to use data instead of normalized data
            if pool_stat:
                print(f"Current maximum winrate average for the pool: {pool_stat:.2f}")
            else:
                print("Current pool is empty or no data available.")

        elif mapped_subcommand == "current-max-norm":
            pool_stat = functions.calculate_pool_wr_stats(normalized_data, config["current_champion_pool"]) # Changed to use normalized data
            if pool_stat:
                print(f"Normalized maximum winrate average for the pool: {pool_stat:.2f}")
            else:
                print("Current pool is empty or no data available.")

        elif mapped_subcommand == "best-add":
            recommendations = functions.calculate_best_champ_additions(data, config["current_champion_pool"]) # Changed to use data
            if recommendations:
                print("Champs that would increase the pool's max winrate average (most to least):")
                for i, (champ, value) in enumerate(recommendations, 1):
                    print(f"{i}. {champ.title()}: {value:.2f}")
            else:
                print("No champs to recommend.")

        elif mapped_subcommand == "best-add-norm":
            recommendations = functions.calculate_best_champ_additions(normalized_data, config["current_champion_pool"]) # Changed to use normalized data
            if recommendations:
                print("Champs that would increase the pool's normalized max winrate average (most to least):")
                for i, (champ, value) in enumerate(recommendations, 1):
                    print(f"{i}. {champ.title()}: {value:.2f}")
            else:
                print("No champs to recommend.")

        elif mapped_subcommand == "worst-remove":
            removals = functions.calculate_worst_champ_removals(data, config["current_champion_pool"]) # Changed to use data
            if removals:
                print("Champs that would decrease the pool's max winrate average (most to least):")
                for i, (champ, value) in enumerate(removals, 1):
                    print(f"{i}. {champ.title()}: {value:.2f}")
            else:
                print("No champs to remove.")

        elif mapped_subcommand == "worst-remove-norm":
            removals = functions.calculate_worst_champ_removals(normalized_data, config["current_champion_pool"]) # Changed to use normalized data
            if removals:
                print("Champs that would decrease the pool's normalized max winrate average (most to least):")
                for i, (champ, value) in enumerate(removals, 1):
                    print(f"{i}. {champ.title()}: {value:.2f}")
            else:
                print("No champs to remove.")
        elif mapped_subcommand == "help":
            for i, cmd in enumerate(subcommands, 1):
                 if cmd != "help" and cmd != "back":
                     print(f"{i}. {cmd} - Show {cmd.replace('-', ' ').replace('norm', 'normalized')} for the pool")
                 else:
                     print(f"{i}. {cmd}")

        elif mapped_subcommand == "back":
            break

        else:
            print(f"Invalid subcommand. Use the listed options, or the corresponding numbers.")


def champ_pick():
    """Analyze matchups for specific input champions against your current pool."""
    input_champs = input("Enter the champions to analyze (separated by commas): ").strip().split(",")
    input_champs = [functions.normalize_champ_name(champ) for champ in input_champs]

    # Map input champs to dataset champs using name mapping
    mapped_input_champs = [name_mapping.get(champ, None) for champ in input_champs]
    mapped_input_champs = [champ for champ in mapped_input_champs if champ is not None]

    # Warn about unmapped champs
    unmapped_input_champs = [champ for champ, mapped in zip(input_champs, mapped_input_champs) if mapped is None]
    if unmapped_input_champs:
        print(f"Warning: The following input champs do not exist in the dataset: {', '.join(unmapped_input_champs)}")
        if not mapped_input_champs:
            print("No valid input champs provided. Returning to the main menu.")
            return

    # Analyze picks
    table = functions.analyze_champ_picks(data, config["current_champion_pool"], mapped_input_champs)
    print(table)

def settings():
    """Subcommands for editing settings."""
    subcommands = [
        "edit-heatmap",
        "edit-name",
        "edit-file",
        "edit-pool",
        "help",
        "back"
    ]

    while True:
        print("Settings Subcommands:")
        for i, cmd in enumerate(subcommands, 1):
             if cmd != "help" and cmd != "back":
                print(f"{i}. {cmd} - {cmd.replace('-', ' ').replace('heatmap', 'heatmap settings').replace('name', 'your name').replace('file', 'the input file path').replace('pool', 'your current champ pool')}")
             else:
                  print(f"{i}. {cmd}")

        subcommand = input("Enter a subcommand (type 'help' to see list of commands again): ").strip().lower()
        mapped_subcommand = map_input_to_string(subcommand, subcommands)

        if mapped_subcommand == "edit-heatmap":
            functions.print_break()
            functions.edit_heatmap_settings(config, config_file_path)

        elif mapped_subcommand == "edit-name":
            functions.print_break()
            new_name = input("Enter your new name: ").strip()
            config["your_name"] = new_name
            functions.save_config(config, config_file_path)
            print(f"Your name has been updated to {new_name}.")
            functions.print_break()

        elif mapped_subcommand == "edit-file":
            functions.print_break()
            new_file = input("Enter the new input file path: ").strip()
            config["input_file"] = new_file
            functions.save_config(config, config_file_path)
            print(f"Input file has been updated to {new_file}.")
            functions.print_break()

        elif mapped_subcommand == "edit-pool":
            functions.print_break()
            new_pool = input("Enter your new champ pool, separated by commas: ").strip().split(",")
            config["current_champion_pool"] = [champ.strip() for champ in new_pool]
            functions.save_config(config, config_file_path)
            print(f"Current champ pool has been updated to {', '.join(config['current_champion_pool'])}.")
            functions.print_break()
        elif mapped_subcommand == "help":
            for i, cmd in enumerate(subcommands, 1):
                if cmd != "help" and cmd != "back":
                   print(f"{i}. {cmd} - {cmd.replace('-', ' ').replace('heatmap', 'heatmap settings').replace('name', 'your name').replace('file', 'the input file path').replace('pool', 'your current champ pool')}")
                else:
                   print(f"{i}. {cmd}")

        elif mapped_subcommand == "back":
            break

        else:
            print(f"Invalid subcommand. Use the listed options, or the corresponding numbers.")

# Command handling loop
while True:
    functions.print_break()
    command_options = list(commands.keys())
    command = input("Enter a command (type 'help' to see list of commands again): ").strip().lower()
    
    
    if command == "help":
      show_help()
    else:
      for i, cmd in enumerate(command_options, 1):
        print(f"{i}. {cmd} - {commands[cmd]}")
      mapped_command = map_input_to_string(command, command_options)

      if mapped_command == "analyze-champs":
          functions.print_break()
          analyze_champs()

      elif mapped_command == "analyze-pool":
          analyze_pool()

      elif mapped_command == "champ-pick":
          functions.print_break()
          champ_pick()

      elif mapped_command == "generate-pools":
          functions.print_break()

          filtered_data = functions.select_champs(data, config["current_champion_pool"], "Choose champion selection option for generate-pools")

          num_champs = int(input("Enter the number of champs per pool: ").strip())
          top_pools = int(input("Enter the number of top pools to display: ").strip())
          pool_limit = int(input("Enter the max appearances of any champ per pool: ").strip())
          valid_combinations = functions.generate_pool_stats(
              filtered_data.corr(),
              num_champs,
              top_pools,
              pool_limit
          )
          print("Best pools:")
          for i, combo in enumerate(valid_combinations, 1):
              print(f"Pool {i}: {combo}")

      elif mapped_command == "heatmap":
          filtered_data = functions.select_champs(data, config["current_champion_pool"], "Choose champion selection option for heatmap")
          functions.generate_heatmap(filtered_data.corr(), config["heatmap_settings"])

      elif mapped_command == "settings":
          functions.print_break()
          settings()

      elif mapped_command == "exit":
          functions.print_break()
          print("Goodbye!")
          break

      else:
          functions.print_break()
          print("Invalid command. Type 'help' to see available commands.")