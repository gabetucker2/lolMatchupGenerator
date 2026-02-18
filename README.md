I retrieved every lane matchup for every meta champion from webscraped Emerald+ global patch 16.03 [op.gg](http://op.gg/) counterpick data on 2/13/2026 (e.g., https://op.gg/lol/champions/elise/counters/jungle).

For your pool to be theoretically optimal if you only care about lane matchups, have one champion who's a strong blindpick and can fall back to when you have no good counterpicks available, and the rest of the champions in your pool should be good counterpicks from the list. Due to constraints in the model discussed below, low-playrate/off-meta champions are made to look like overpowered blindpicks, so be skeptical of low-playrate/off-meta champions on the lists (like Anivia top).

The 95% confidence interval of a champ you'll play against being within that range is in parentheses after each percentage. For example,

Zac: 5.13 (41.48% - 63.33%) means if you lock Zac, you'll most likely play against a champ you counter with a 5% matchup winrate advantage or a 5% matchup win rate disadvantage with respect to your current champ's win rate. And there's a 2.5% chance you will play against a champ against whom your winrate is below 41.48% and a 2.5% chance you will play against a champ against whom your winrate is above 63.33%. So those are the "outsider" matchup bounds you are expected to only have to face in extreme counterpick scenarios, but picks more extreme than those are extremely implausible.

**TOP LANE**

1. Zac: 5.13% (41.48% - 63.33%) (best counterpick, worst blindpick)
2. Master Yi: 4.94% (32.44% - 58.50%)
3. Vayne: 4.86% (38.88% - 58.87%)
4. Anivia: 4.78% (40.05% - 64.04%)
5. Naafiri: 4.69% (40.00% - 61.86%)
6. Heimerdinger: 4.62% (39.55% - 61.82%)
7. Maokai: 4.52% (39.25% - 61.07%)
8. Sylas: 4.10% (36.28% - 55.72%)
9. Quinn: 3.95% (41.36% - 61.15%)
10. Cassiopeia: 3.86% (41.58% - 62.58%)
11. Warwick: 3.58% (37.92% - 60.47%)
12. Varus: 3.42% (39.85% - 57.01%)
13. Swain: 3.37% (42.16% - 61.72%)
14. Aurora: 3.26% (39.33% - 58.36%)
15. Dr. Mundo: 3.22% (40.89% - 56.55%)
16. Irelia: 3.21% (43.51% - 58.67%)
17. Teemo: 3.20% (42.01% - 58.18%)
18. Yasuo: 3.18% (42.48% - 58.80%)
19. Malphite: 3.18% (40.77% - 57.84%)
20. Wukong: 3.14% (34.40% - 57.26%)
21. Poppy: 3.12% (41.39% - 59.40%)
22. Tahm Kench: 3.03% (41.81% - 56.50%)
23. Cho'Gath: 2.94% (43.75% - 59.30%)
24. Singed: 2.90% (45.97% - 60.09%)
25. Ryze: 2.87% (35.81% - 55.89%)
26. Yorick: 2.84% (42.65% - 57.03%)
27. Akali: 2.82% (39.13% - 56.23%)
28. Renekton: 2.75% (42.52% - 55.74%)
29. Volibear: 2.75% (40.76% - 58.45%)
30. Jayce: 2.74% (42.98% - 55.52%)
31. Kennen: 2.74% (45.77% - 61.95%)
32. Vladimir: 2.72% (41.18% - 55.53%)
33. Trundle: 2.72% (37.14% - 55.35%)
34. Kayle: 2.68% (46.61% - 60.10%)
35. Urgot: 2.61% (45.82% - 64.41%)
36. Illaoi: 2.60% (38.11% - 57.89%)
37. Sion: 2.60% (46.06% - 56.34%)
38. Yone: 2.59% (41.90% - 56.62%)
39. Garen: 2.51% (45.24% - 55.28%)
40. Gwen: 2.51% (43.43% - 59.71%)
41. Fiora: 2.50% (42.33% - 56.89%)
42. Jax: 2.48% (43.89% - 55.08%)
43. Pantheon: 2.48% (40.65% - 60.30%)
44. Riven: 2.48% (43.88% - 58.64%)
45. Kled: 2.48% (43.62% - 61.10%)
46. K'Sante: 2.48% (39.42% - 54.88%)
47. Olaf: 2.42% (43.84% - 56.92%)
48. Sett: 2.38% (45.28% - 60.24%)
49. Tryndamere: 2.32% (42.73% - 58.67%)
50. Gragas: 2.32% (41.95% - 58.27%)
51. Nasus: 2.30% (44.05% - 56.00%)
52. Gnar: 2.24% (43.50% - 55.99%)
53. Camille: 2.23% (42.87% - 60.39%)
54. Rumble: 2.08% (44.22% - 54.69%)
55. Gangplank: 2.08% (44.53% - 55.09%)
56. Darius: 2.03% (42.76% - 58.31%)
57. Ambessa: 2.01% (44.05% - 53.19%)
58. Shen: 1.80% (47.70% - 58.63%)
59. Zaahen: 1.74% (46.17% - 56.25%)
60. Mordekaiser: 1.72% (43.78% - 55.20%)
61. Aatrox: 1.65% (44.91% - 55.06%)
62. Ornn: 1.57% (48.25% - 57.14%) (best blindpick, worst counterpick)

**JUNGLE**

1. Skarner: 3.82% (37.02% - 57.19%) (best counterpick, worst blindpick)
2. Rammus: 3.43% (44.29% - 60.46%)
3. Gragas: 3.30% (39.26% - 62.86%)
4. Malphite: 3.00% (38.81% - 52.66%)
5. Nasus: 2.96% (43.94% - 60.54%)
6. Sejuani: 2.95% (42.80% - 57.49%)
7. Zyra: 2.89% (44.32% - 59.88%)
8. Aatrox: 2.85% (36.78% - 57.24%)
9. Shyvana: 2.85% (41.01% - 57.53%)
10. Ivern: 2.79% (46.72% - 65.74%)
11. Pantheon: 2.70% (40.94% - 56.99%)
12. Amumu: 2.64% (41.18% - 56.38%)
13. Trundle: 2.62% (35.85% - 60.07%)
14. Naafiri: 2.59% (43.87% - 54.54%)
15. Warwick: 2.52% (41.59% - 54.60%)
16. Wukong: 2.45% (43.40% - 56.34%)
17. Dr. Mundo: 2.45% (44.51% - 56.97%)
18. Evelynn: 2.41% (45.24% - 56.77%)
19. Udyr: 2.38% (43.10% - 58.60%)
20. Nidalee: 2.37% (41.67% - 57.27%)
21. Sylas: 2.30% (42.77% - 55.92%)
22. Elise: 2.30% (41.88% - 56.52%)
23. Nunu & Willump: 2.28% (45.82% - 57.35%)
24. Gwen: 2.27% (46.02% - 57.54%)
25. Bel'Veth: 2.26% (44.97% - 58.05%)
26. Karthus: 2.24% (42.41% - 56.14%)
27. Qiyana: 2.23% (39.58% - 53.77%)
28. Taliyah: 2.21% (44.17% - 56.44%)
29. Rengar: 2.21% (43.94% - 55.04%)
30. Zed: 2.21% (39.42% - 52.13%)
31. Jax: 2.20% (44.64% - 55.60%)
32. Master Yi: 2.11% (45.56% - 55.23%)
33. Jayce: 2.09% (42.15% - 53.56%)
34. Zac: 2.04% (47.41% - 57.21%)
35. Lillia: 2.02% (45.92% - 56.88%)
36. Volibear: 1.97% (45.01% - 56.75%)
37. Shaco: 1.95% (47.58% - 57.47%)
38. Kindred: 1.94% (44.18% - 54.01%)
39. Fiddlesticks: 1.93% (46.50% - 56.83%)
40. Graves: 1.90% (43.90% - 53.94%)
41. Viego: 1.86% (45.27% - 53.10%)
42. Kha'Zix: 1.82% (47.51% - 55.68%)
43. Diana: 1.81% (45.84% - 56.39%)
44. Briar: 1.78% (47.16% - 55.88%)
45. Rek'Sai: 1.75% (47.60% - 57.63%)
46. Nocturne: 1.72% (45.47% - 55.59%)
47. Hecarim: 1.72% (45.62% - 55.06%)
48. Ambessa: 1.71% (43.44% - 53.39%)
49. Talon: 1.70% (45.66% - 53.70%)
50. Lee Sin: 1.69% (44.98% - 53.86%)
51. Ekko: 1.67% (46.58% - 55.15%)
52. Zaahen: 1.62% (45.66% - 54.75%)
53. Kayn: 1.54% (46.02% - 53.97%)
54. Jarvan Iv: 1.40% (46.58% - 54.83%)
55. Vi: 1.33% (44.59% - 54.99%)
56. Xin Zhao: 1.24% (46.37% - 54.04%) (best blindpick, worst counterpick)

**MID LANE**

1. Malphite: 4.45% (34.01% - 59.77%) (best counterpick, worst blindpick)
2. Smolder: 3.94% (32.82% - 52.82%)
3. Kennen: 3.65% (32.01% - 62.42%)
4. Jayce: 3.63% (39.41% - 64.01%)
5. Vel'Koz: 3.63% (43.66% - 64.27%)
6. Sion: 3.61% (40.39% - 61.69%)
7. Aurelion Sol: 3.52% (43.05% - 61.20%)
8. Pantheon: 3.33% (38.39% - 59.92%)
9. Taliyah: 3.32% (41.69% - 56.83%)
10. Brand: 3.29% (38.10% - 64.89%)
11. Annie: 3.22% (41.11% - 57.47%)
12. Ziggs: 3.04% (42.95% - 57.53%)
13. Cho'Gath: 2.95% (44.22% - 61.35%)
14. Cassiopeia: 2.87% (40.92% - 56.96%)
15. Mel: 2.86% (38.74% - 52.92%)
16. Vex: 2.84% (43.24% - 59.11%)
17. Naafiri: 2.80% (44.79% - 60.39%)
18. Kayle: 2.75% (41.63% - 63.36%)
19. Xerath: 2.72% (45.02% - 61.48%)
20. Irelia: 2.70% (40.13% - 59.83%)
21. Vladimir: 2.70% (42.47% - 56.70%)
22. Syndra: 2.70% (45.26% - 57.80%)
23. Malzahar: 2.67% (43.47% - 55.81%)
24. Veigar: 2.60% (45.46% - 57.58%)
25. Anivia: 2.58% (44.75% - 57.60%)
26. Yone: 2.57% (42.66% - 55.55%)
27. Hwei: 2.50% (44.80% - 56.29%)
28. Galio: 2.49% (44.90% - 55.28%)
29. Zoe: 2.47% (44.72% - 57.16%)
30. Swain: 2.45% (44.91% - 61.15%)
31. Azir: 2.39% (36.65% - 52.54%)
32. Fizz: 2.35% (46.31% - 57.04%)
33. Ryze: 2.34% (41.03% - 52.33%)
34. Katarina: 2.32% (44.98% - 56.85%)
35. Lux: 2.32% (46.86% - 55.80%)
36. Talon: 2.30% (44.11% - 57.20%)
37. Akali: 2.30% (44.28% - 55.09%)
38. Aurora: 2.29% (43.45% - 56.31%)
39. Akshan: 2.25% (44.50% - 57.22%)
40. Diana: 2.21% (45.31% - 59.38%)
41. Twisted Fate: 2.18% (45.86% - 56.82%)
42. Lissandra: 2.15% (42.86% - 55.67%)
43. Leblanc: 2.11% (40.64% - 56.47%)
44. Zed: 2.08% (45.89% - 55.55%)
45. Viktor: 2.08% (45.94% - 57.45%)
46. Yasuo: 2.08% (45.08% - 55.61%)
47. Ekko: 1.98% (44.23% - 53.95%)
48. Sylas: 1.98% (42.52% - 57.31%)
49. Qiyana: 1.96% (47.55% - 59.67%)
50. Kassadin: 1.89% (45.80% - 56.22%)
51. Orianna: 1.88% (43.16% - 56.97%)
52. Ahri: 1.69% (47.55% - 57.38%) (best blindpick, worst counterpick)

**ADC**

1. Seraphine: 3.23% (44.23% - 58.75%) (best counterpick, worst blindpick)
2. Senna: 2.99% (35.62% - 56.82%)
3. Yasuo: 2.82% (44.69% - 57.54%)
4. Nilah: 2.76% (43.32% - 57.19%)
5. Kalista: 2.76% (38.67% - 52.45%)
6. Veigar: 2.60% (46.66% - 62.50%)
7. Kog'Maw: 2.27% (46.05% - 61.84%)
8. Mel: 2.22% (40.23% - 56.14%)
9. Brand: 2.18% (46.72% - 60.04%)
10. Samira: 2.06% (42.52% - 57.59%)
11. Xayah: 1.95% (41.85% - 54.59%)
12. Sivir: 1.90% (44.56% - 54.27%)
13. Vayne: 1.87% (45.98% - 55.52%)
14. Aphelios: 1.84% (41.49% - 52.73%)
15. Kai'Sa: 1.72% (44.26% - 53.46%)
16. Zeri: 1.72% (42.08% - 53.67%)
17. Draven: 1.65% (43.60% - 54.69%)
18. Smolder: 1.63% (45.08% - 53.67%)
19. Swain: 1.61% (47.00% - 58.50%)
20. Tristana: 1.57% (45.25% - 56.76%)
21. Yunara: 1.57% (41.81% - 52.03%)
22. Miss Fortune: 1.53% (42.59% - 53.46%)
23. Ziggs: 1.49% (45.24% - 59.91%)
24. Twitch: 1.48% (44.51% - 53.96%)
25. Ashe: 1.42% (44.90% - 53.48%)
26. Lucian: 1.39% (42.37% - 52.86%)
27. Caitlyn: 1.37% (42.96% - 52.21%)
28. Varus: 1.33% (43.91% - 52.88%)
29. Jhin: 1.28% (45.91% - 52.39%)
30. Jinx: 1.24% (46.01% - 55.56%)
31. Ezreal: 1.16% (43.01% - 51.43%)
32. Corki: 1.05% (43.19% - 51.60%) (best blindpick, worst counterpick)

**SUPPORT**

1. Ashe: 3.59% (33.99% - 52.96%) (best counterpick, worst blindpick)
2. Leblanc: 3.45% (38.68% - 54.38%)
3. Mel: 3.12% (37.42% - 57.58%)
4. Elise: 3.08% (42.86% - 60.19%)
5. Zoe: 3.05% (39.64% - 56.44%)
6. Swain: 3.02% (42.39% - 54.18%)
7. Shaco: 2.97% (43.48% - 57.24%)
8. Fiddlesticks: 2.87% (39.38% - 57.35%)
9. Xerath: 2.63% (44.89% - 55.56%)
10. Brand: 2.51% (46.70% - 57.73%)
11. Taric: 2.51% (46.03% - 62.50%)
12. Pantheon: 2.42% (42.63% - 56.14%)
13. Yuumi: 2.37% (43.46% - 58.74%)
14. Morgana: 2.33% (45.39% - 55.95%)
15. Tahm Kench: 2.26% (45.13% - 55.84%)
16. Zyra: 2.20% (43.75% - 57.35%)
17. Rell: 2.19% (43.75% - 56.25%)
18. Leona: 2.17% (46.88% - 59.48%)
19. Poppy: 2.17% (44.00% - 54.17%)
20. Renata Glasc: 2.14% (44.12% - 54.69%)
21. Rakan: 2.13% (45.79% - 56.74%)
22. Nautilus: 2.08% (45.04% - 57.47%)
23. Braum: 1.99% (48.29% - 60.77%)
24. Lux: 1.98% (44.77% - 53.97%)
25. Seraphine: 1.97% (46.46% - 56.71%)
26. Blitzcrank: 1.96% (45.85% - 54.61%)
27. Alistar: 1.95% (45.55% - 56.54%)
28. Senna: 1.92% (45.37% - 53.81%)
29. Vel'Koz: 1.92% (46.15% - 55.06%)
30. Maokai: 1.89% (46.22% - 57.75%)
31. Soraka: 1.89% (48.25% - 57.38%)
32. Pyke: 1.78% (45.15% - 56.76%)
33. Karma: 1.76% (46.78% - 58.11%)
34. Sona: 1.73% (49.18% - 60.93%)
35. Janna: 1.71% (47.77% - 57.30%)
36. Neeko: 1.68% (44.94% - 56.25%)
37. Zilean: 1.66% (48.07% - 56.90%)
38. Milio: 1.66% (48.47% - 58.70%)
39. Lulu: 1.51% (47.62% - 56.97%)
40. Bard: 1.46% (46.77% - 57.50%)
41. Nami: 1.40% (48.83% - 58.01%)
42. Thresh: 1.31% (48.37% - 54.97%) (best blindpick, worst counterpick)

From this we can determine the most and least matchup-dependent roles. Top's highest wr champ gives a 5.13% win rate boost if counterpicking, mid 4.45%, jungle 3.82%, support 3.59%, and ADC 3.23%. So support and ADC should first pick, jungle and mid should second pick, and top should last pick. Unless you have a blindable champ in your role you play no matter what, in which case you should ideally pick first.

How was this caclulated? I took the mean absolute error (MAE) of every champion’s matchup winrate list: mean ( | wr vs X matchup - baseline champ wr | , | wr vs Y matchup - baseline champ wr | , … , | wr vs Z matchup - baseline champ wr | ). Given we are considering all of a champion’s matchups, this calculates a percentage point representing the expected deviation of that champion’s winrate in their average matchup from their base winrate.

For example, if Zaahen has a 52% win rate and two matchups with the winrates [47%, 57%], then 5% is the MAE for this champion because the average deviation of his matchup winrates is 5%, meaning in lane he will have either a 5% advantage, or a 5% disadvantage, based on the champion he is laning against. If his matchups are [47%, 52%, 52%, 57%], then his MAE will be 2.5% and he will on average have a 2.5% advantage or disadvantage against his lane opponent.

MAE is automatically a meta-agnostic calculation: if your champ’s winrate is 50% and your matchup winrates are [45%, 55%], your MAE is 5%; if your champ’s winrate is 55% and your matchup winrates are [50%, 60%], your MAE is still 5%.

We use MAE instead of the root mean squared error (RMSE) because RMSE upscales outliers, which is not helpful unless you’re looking to have as few horrible matchups as possible rather than the best average matchup possible.

Champions with lower MAEs are better blindpicks because they have fewer unplayable matchups, but fewer matchups where they stomp; when you are blindpicking, you do not expect your opponent to counterpick themselves so it is better to play a champion with fewer good and bad matchups and more matchups in the middle of the road, i.e., it is better to play a champion with a lower MAEs.

Champions with higher MAEs are better counterpicks for the same reason: they have more unplayable matchups, but more matchups where they stomp. If you are counterpicking your opponent, e.g., in top lane, and you have a champion with a high MAEs, then this champion is statistically likely not to have a 52% win rate in a good matchup, but something more ridiculous like a 55% win rate which will let you stomp your lane. If your counterpick champion instead has a 45% win rate against your lane opponent who already picked, then you can simply default to your pool’s blind pick champion as a backup scenario and not have to worry about being counterpicked.

However, there is one issue remaining: if Zaahen (52% wr) has two matchups, Heimerdinger (80% matchup wr) and Aatrox (52% matchup wr), and Heimerdinger has been chosen top lane in 5 games this season and Aatrox has been chosen top 500 games this season (using season playrate rather than champ v champ playrate to militate counterpick and dodging effects on playrate within a matchup), then the MAE will be 14%: ( |52% - 52%| + |80% - 52%| ) / 2 = 14%. HOWEVER, this paints an incomplete picture because we are not accounting for the likelihood of a champion to be picked in our calculations. Aatrox is 100 times more likely to be picked than Heimerdinger, so therefore Zaahen’s winrate against Aatrox should be weighed 100 times more heavily than Heimerdinger, and the calculation with respect to the total weights would look like this: ( |52% - 52%|*.99 + |80% - 52%|*.01 ) / (0.01 + 0.99) = 0.28%. This better reflects how Zaahen in this hypothetical scenario is basically guaranteed to have an even winrate against his lane opponent. We call this process normalization.

A final important consideration is that 0.28% is calculated based on weights, which are based on # games played in the season paired with the winrate in that matchup, and some champions have higher playrates than others and therefore more games played across all matchups, so we have to ensure that Zaahen having a higher playrate than other champs doesn’t cause his normalized MAE score to be higher or lower than it should be. This allows us to upscale the importance of within-champion frequently-played matchups and remove between-champion playrate effects on how strong of a blindpick/counterpick they are more broadly. We’ll call this equalization.

One constraint is meta-agnosticism. Meta agnosticism is the assumption that all champions are equally powerful, i.e., the isolation of matchup volatility from champion strength. Therefore, the model assumes that those with low overall winrates but high MAE scores are still great blindpicks against champs they have higher winrates against than their average. It conversely assumes that those with high overall winrates and low MAE scores are bad counterpick champions because their capacity to counterpick is a result of them being strong right now rather than them having some matchups they’re way better against than others. If a champion has an awful winrate, their MAE score might still be high, which will make them look like an amazing counterpick while in actuality if you only look at their stats they only win against a handful of champions and lose to the rest. The statistics are basically assuming that Zed jungle’s best matchup — Lillia — is a fantastic auto-win counterpick even though his winrate against her is only 52.5% because Zed jungle is an off-meta champion and generally just sucks right now. All of this is to say that using this data will let you choose a champion whose identity as a blindpicker/counterpicker will not significantly change with meta shifts, even after the champion gets buffs or nerfs to their kit in future patches. However, this makes some off-meta champs who are performing horribly in their current roles look like OP blindpicks, which can be misleading, so take that with a grain of salt.

Another constraint is how due to equalization in the model which puts all champions on the same playing field regardless of their playrates, champions with low playrates are likely to have less accurate data due to having played fewer games and therefore are likely to have higher MAE scores since matchup winrates haven’t had as much time to converge to the mean per the law of large numbers.

Low winrate champs are also likely to have auto-lose matchups that cause low-winrate champs to erroneously appear as better blindpicks as well.

I hope this helps other stat nerds create better champ pools and have a numbers-based approach to selecting their mains!

# Installation Guide (Windows only)

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


**This is all outdated information.**

# Heatmap

The **League of Legends Top Lane Matchup Generator** helps you build an optimized champion pool by minimizing counterpicks while maximizing coverage for top lane matchups.

![Heatmap](heatmap.png)

- **Enlarge image here:** [Enlarged image](https://raw.githubusercontent.com/gabetucker2/lolMatchupGenerator/refs/heads/main/heatmap.png)
- **Winrate Data Source:** Extracted from [www.op.gg](https://www.op.gg)  
- **Patch Version:** **14.23**
- **Rank:** **Emerald +**

*This heatmap is meta-agnostic, meaning that how meta a champion is has no impact on the champion's matchup correlations with other champs.*

Ambessa and Yone have a `0.76` correlation. Since this correlation coefficient is so close to `1`, this means they tend to counter similar champs, and they tend to be countered by similar champs. Therefore, Ambessa and Yone would be bad champs to have in the same champ pool.

Conversely, Warwick and Dr. Mundo have a `-0.27` correlation. Since this correlation coefficient is much closer to `-1` than most other correlation coefficients, this means they tend to counter different champs, and they tend to be countered by different champs. Therefore, Warwick and Dr. Mundo would be exceptional champs to have in the same champ pool.

---

# Superlatives

## 15 most blind-pickable top-laners in the game
1. **K'Sante** (best blind pick in top lane, worst counterpick in top lane)
2. Gnar
3. Aatrox
4. Volibear
5. Aurora
6. Renekton
7. Gragas
8. Urgot
9. Sett
10. Ornn
11. Shen
12. Darius
13. Pantheon
14. Ambessa
15. Fiora

## 15 least blind-pickable top-laners in the game
1. **Galio** (worst blind pick in top lane, best counterpick in top lane)
2. Vayne
3. Trundle
4. Wukong
5. Udyr
6. Heimerdinger
7. Quinn
8. Malphite
9. Tryndamere
10. Cassiopeia
11. Rumble
12. Irelia
13. Yorick
14. Akali
15. Ryze

## 15 champs most likely to be strong counterpick additions for your champ pool
1. **Smolder** (best average counterpick addition in top lane)
2. Dr. Mundo
3. Cassiopeia
4. Akali
5. Yorick
6. Heimerdinger
7. Fiora
8. Vayne
9. Galio
10. Urgot
11. Olaf
12. Volibear
13. Swain
14. Singed
15. Ryze

## 15 champs least likely to be strong counterpick additions for your champ pool
1. **Gragas** (worst average counterpick addition in top lane) 
2. Kled
3. Camille
4. Jax
5. Ambessa
6. Aatrox
7. Gangplank
8. Poppy
9. Tahm Kench
10. Mordekaiser
11. Irelia
12. Gnar
13. Tryndamere
14. Darius
15. Garen

### 5 best top lane two-tricks combos
1. Dr. Mundo / Smolder (best two-trick combo in the game)
2. Cassiopeia / Dr. Mundo
3. Akali / Smolder
4. Cassiopeia / Dr. Mundo
5. Smolder / Yorick

The rest of this dataset is unfortunately not very interesting because the **15 champs most likely to be strong counterpick additions for your champ pool** list are repeating so frequently.  To solve this and make a more interesting best two-trick list, let's add a rule saying that no champs can be in more than N pools.  This will add more variation:

### 15 best top lane two-trick combos (exclusion after 1 appearance)
1. Dr. Mundo / Smolder
2. Akali / Cassiopeia
3. Heimerdinger / Yorick
4. Fiora / Vayne
5. Galio / Urgot
6. Olaf / Volibear
7. Singed / Swain
8. Ryze / Vladimir
9. Nasus / Ornn
10. Cho'Gath / Illaoi
11. Malphite / Maoki
12. Aurora / Udyr
13. Warwick / Yone
14. K'Sante / Trundle
15. Kennen / Shen

### 15 best top lane two-trick combos (exclusion after 2 appearances)
1. Dr. Mundo / Smolder
2. Cassiopeia / Smolder
3. Cassiopeia / Dr. Mundo
4. Akali / Yorick
5. Akali / Heimerdinger
6. Heimerdinger / Yorick
7. Fiora / Vayne
8. Fiora / Galio
9. Galio / Vayne
10. Olaf / Urgot
11. Urgot / Volibear
12. Olaf / Volibear
13. Singed / Swain
14. Ryze / Swain
15. Ryze / Singed

### 15 best top lane three-tricks combos (exclusion after 1 appearance)
1. Cassiopeia / Dr. Mundo / Smolder
2. Akali / Heimerdinger / Yorick
3. Fiora / Galio / Vayne
4. Olaf / Urgot / Volibear
5. Ryze / Singed / Swain
6. Nasus / Ornn / Vladimir
7. Cho'Gath / Illaoi / Malphite
8. Aurora / Maokai / Udyr
9. Trundle / Warwick / Yone
10. K'Sante / Kennen / Shen
11. Rumble / Sion / Teemo
12. Gwen / Kayle / Sylas
13. Quinn / Riven / Yasuo
14. Pantheon / Renekton / Sett
15. Jayce / Wukong / Zac

### 15 best top lane three-tricks combos (exclusion after 2 appearances)
1. Cassiopeia / Dr. Mundo / Smolder
2. Akali / Dr. Mundo / Smolder
3. Akali / Cassiopeia / Yorick
4. Fiora / Heimerdinger / Yorick
5. Fiora / Heimerdinger / Vayne
6. Galio / Urgot / Vayne
7. Galio / Olaf / Urgot
8. Olaf / Swain / Volibear
9. Singed / Swain / Volibear
10. Ryze / Singed / Vladimir
11. Ornn / Ryze / Vladimir
12. Cho'Gath / Nasus / Ornn
13. Cho'Gath / Illaoi / Nasus
14. Illaoi / Malphite / Maokai
15. Malphite / Maokai / Udyr

### 15 best top lane four-tricks combos (exclusion after 1 appearance)
1. Akali / Cassiopeia / Dr. Mundo / Smolder
2. Fiora / Heimerdinger / Vayne / Yorick
3. Galio / Olaf / Urgot / Volibear
4. Ryze / Singed / Swain / Vladimir
5. Cho'Gath / Illaoi / Nasus / Ornn
6. Aurora / Malphite / Maokai / Udyr
7. K'Sante / Trundle / Warwick / Yone
8. Kennen / Rumble / Shen / Teemo
9. Gwen / Kayle / Sion / Sylas
10. Quinn / Renekton / Riven / Yasuo
11. Jayce / Pantheon / Sett / Wukong
12. Darius / Garen / Tryndamere / Zac
13. Gnar / Irelia / Mordekaiser / Tahm Kench
14. Aatrox / Ambessa / Gangplank / Poppy
15. Camille / Gragas / Jax / Kled

### 15 best top lane four-tricks combos (exclusion after 2 appearances)
1. Akali / Cassiopeia / Dr. Mundo / Smolder
2. Cassiopeia / Dr. Mundo / Smolder / Yorick
3. Akali / Fiora / Heimerdinger / Yorick
4. Fiora / Galio / Heimerdinger / Vayne
5. Galio / Olaf / Urgot / Vayne
6. Olaf / Swain / Urgot / Volibear
7. Ryze / Singed / Swain / Volibear
8. Ornn / Ryze / Singed / Vladimir
9. Cho'Gath / Nasus / Ornn / Vladimir
10. Cho'Gath / Illaoi / Malphite / Nasus
11. Illaoi / Malphite / Maokai / Udyr
12. Aurora / Maokai / Udyr / Warwick
13. Aurora / Trundle / Warwick / Yone
14. K'Sante / Shen / Trundle / Yone
15. K'Sante / Kennen / Shen / Teemo
