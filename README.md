# League of Legends Top Lane Matchup Generator

The **League of Legends Top Lane Matchup Generator** dataset helps you build an optimized champion pool by minimizing counterpicks while maximizing coverage for top lane matchups.

![Heatmap](heatmap.png)

- **Enlarge image here:** [Enlarged image](https://raw.githubusercontent.com/gabetucker2/lolMatchupGenerator/refs/heads/main/heatmap.png)
- **Winrate Data Source:** Extracted from [www.op.gg](https://www.op.gg)  
- **Patch Version:** **14.23**

*This heatmap is meta-agnostic, meaning that how meta a champion is has no impact on the champion's matchup correlations with other champs.*

Ambessa and Yone have a `0.76` correlation. Since this correlation coefficient is so close to `1`, this means they tend to counter similar champs, and they tend to be countered by similar champs. Therefore, Ambessa and Yone would be bad champs to have in the same champ pool.

Conversely, Warwick and Dr. Mundo have a `-0.27` correlation. Since this correlation coefficient is much closer to `-1` than most other correlation coefficients, this means they tend to counter different champs, and they tend to be countered by different champs. Therefore, Warwick and Dr. Mundo would be exceptional champs to have in the same champ pool.

---

## Installation Guide (Windows)

### Step 1: Install Git
1. Download Git from [https://git-scm.com](https://git-scm.com).
2. Run the installer and follow the setup wizard with default settings.

### Step 2: Clone the Repository
1. Open Command Prompt:
   - Press `Win + R`, type `cmd`, and press Enter.
2. Navigate to the folder where you want to save the project:
   ```bash
   cd <desired-folder-path>
   ```
   Replace `<desired-folder-path>` with the folder path where you want to clone the project.
3. Clone the repository:
   ```bash
   git clone https://github.com/gabetucker2/lolMatchupGenerator.git
   ```

### Step 3: Navigate to the Project Directory
1. Change to the project folder:
   ```bash
   cd lolMatchupGenerator
   ```

---

## Installing Python on Windows (if not already installed)

### Step 1: Download Python
1. Visit [https://www.python.org/downloads/](https://www.python.org/downloads/) and download the latest version of Python.

### Step 2: Install Python
1. Run the installer.
2. **Important:** Ensure you check the box for **"Add Python to PATH"** during the installation process.
3. Complete the installation by following the wizard prompts.

### Step 3: Verify Installation
1. Open Command Prompt.
2. Type the following command to check the Python version:
   ```bash
   python --version
   ```
   If installed correctly, you should see the version number (e.g., `Python 3.x.x`).

---

## Running the Matchup Generator

### Step 1: Configure Settings
- Open the `config.json` file in the project folder.
- Adjust the settings to your preference, such as adding champions to the `exclude_champions` or `only_include_champions` lists if you don't want certain champions to be included, or if you only want certain champions to be included. If `only_include_champions` has at least one element, the script will ignore `exclude_champions`.

### Step 2: Run the Program
1. Double-click the `run.bat` file in the project folder.
2. A Command Prompt will open, displaying the best top lane champ pools based on the provided data.
3. A heatmap will open, visualizing all champ matchup correlations.

---

Email [gabeqtucker@gmail.com](mailto:gabeqtucker@gmail.com) with any questions!
