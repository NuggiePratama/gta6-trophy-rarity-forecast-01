# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/NuggiePratama/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/NuggiePratama/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every Grand Theft Auto VI trophy on PlayStation 5, the percent
of the players Exophase tracks who will have earned it at 2026-11-26T00:00Z,
as five quantiles per trophy, from the house's recording of the trophy page
up to the case checkpoint.

## Entry

`main.py` (Python 3.13, standard library only) reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `*.out` per case to
`$MAKEFUN_OUTPUT_DIR` (else `/output`):

```json
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
```

Run locally:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic (two runs byte-identical), offline, ~0.1 s for two cases.

## Model

Per trophy, a trend in logit space fitted on the case's own history, shrunk
toward the field, with a spread that prices the two things the house
baselines ignore: the fit's own uncertainty, and Exophase's discrete
refreshes.

Exophase does not update continuously. It re-reads the page on its own
schedule (hours apart) and the rates jump at each refresh: on launch night
the page sat unchanged at 638 players tracked from 23:48Z to 04:00Z, then
jumped to 3,868 players and every rate but the one at 100% fell, because
the newly tracked players held fewer trophies. A rate's path is flat
stretches plus dilution jumps, and the final week's path is the same shape
over 4.5 days: new players keep joining (dilution, rates down) while the
players already tracked keep earning (progression, rates up, fastest for
the early story trophies).

1. **Distinct states.** Consecutive reads with an identical rate vector
   collapse into one state. A 5-hour stale page counts once, not many
   times.
2. **Weighted trend.** Least squares of logit(rate) on ln(1 + hours since
   release), weighted by earned counts (rate x players / 100): a rate
   built on 7 players is far noisier in logit terms than one built on
   7,000. Log time because earn rates saturate.
3. **Shrinkage.** Each slope is shrunk toward the median slope of all
   trophies, with prior strength 3 states. A trophy with two distinct
   states has a line through two points and no residual; the field's
   slope is a better prior than its own noise.
4. **Median.** The shrunk line read at the target, clamped to move at
   most 2.5 logit from the last read. When the whole history is one flat
   state the page is stale and a refresh is due, so the median shifts
   down 0.9 logit (half the first documented refresh's ~1.8 logit
   dilution of rare trophies, scaled by how much of a refresh interval
   the staleness plus the horizon covers, tapering to zero at 100%).
   When only one trophy is flat while the page moved, that is an
   equilibrium and its median holds the last rate.
5. **Spread.** sigma^2 = random-walk^2 + fit^2 + refresh^2. The random
   walk is the house baseline's 0.5 x sqrt(days). The fit term is the
   line's residual plus its slope uncertainty stretched to the target.
   The refresh term is the median |logit jump| between distinct states
   in the case's history (0.75 logit when the history shows no jump),
   grown with the square root of the expected refresh count and capped
   at 0.6, because refreshes average out over a long horizon. Sigma is
   floored at 0.5 and capped at 1.6 logit.

Quantiles are the median shifted by the normal z for each level, back
through the inverse logit, rounded half-up to 4 decimals.

## Validation (local; the server grades the real score)

Practice cases p00/p01 have completely flat histories, so no exact
practice loss can be computed locally (the answers are the refreshed
reads the house holds). Instead the forecast was scored against three
synthetic answers, all consistent with the house's account of the first
refresh (players 638 -> 3,868, trophies earned 943 -> 4,276, the 100%
trophy unchanged): new players all hold the first-mission trophy with
the rest on early trophies; new trophies proportional to old counts;
new trophies only on early trophies. Mean pinball loss:

| scenario | this entry | house baseline |
| --- | --- | --- |
| fresh-heavy | 0.45 | 1.57 |
| proportional | 0.33 | 1.40 |
| early-only | 0.46 | 1.60 |

House baseline practice loss per the kit: ~1.51. Typical miss
0.17-0.23 points vs 0.68-0.70 for the baseline.

A synthetic final-like case (120 reads over 2 days, 15 refresh states,
tracked players 638 -> 34,761, logit-linear true paths): loss 0.23 vs
0.25 for the baseline, typical miss 0.25 vs 1.06 points.

Edge cases (empty history, all-null rates, single read, zero horizon,
empty trophy list, malformed case file) all exit 0 with valid output.
Grader simulation (clean read-only copy, `python3 -P -s`, fresh output
dir with a pre-existing file): exit 0, byte-identical outputs, the
pre-existing file untouched. The program never deletes the output
directory; it only creates it and writes files.

## Limits

- The practice answers are not public, so the practice figures above
  are scenario ranges, not the real practice score.
- The final week's true paths are unknown; the trend is fitted on two
  days of history and log time saturates it, but a trend can bend.
- The flat-history dilution prior (0.9 logit) is a judgment call from
  the one documented refresh, not a fitted parameter; it only fires
  when the whole history is one state.

---

<sub>Entered on [make.fun](https://make.fun/?ref=e7kjxsk8&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
