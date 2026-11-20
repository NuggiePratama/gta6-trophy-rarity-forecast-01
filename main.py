"""Forecast each GTA VI trophy's Exophase earn rate one week after launch.

Model: a per-trophy trend in logit space, fitted on the case's own history,
shrunk toward the field, with a spread that prices the two things the house
baselines ignore: the fit's own uncertainty, and Exophase's discrete refreshes.

Why this shape
--------------
Exophase does not update continuously. It re-reads the game's page on its own
schedule (hours apart) and the rates jump at each refresh: on launch night the
page sat unchanged at 638 players tracked from 23:48Z to 04:00Z, then jumped
to 3,868 players and every rate but the one at 100% fell, because the newly
tracked players held fewer trophies. So a rate's path is flat stretches plus
dilution jumps, and the final week's path is the same shape stretched over
4.5 days: new players keep joining (dilution, rates down) while the players
already tracked keep earning (progression, rates up, fastest for the early
story trophies). A persistence forecast misses both forces; a curve fitted
on the raw reads overweights the flat stretches, because a stretch of
identical reads is one observation of a stale page, not many observations
of a steady rate.

What the program does, per trophy
---------------------------------
1. Collapse the history into distinct states: consecutive reads with an
   identical rate vector are one state. The slope is fitted on states, so a
   5-hour stale page counts once.
2. Fit a weighted line of logit(rate) on ln(1 + hours since release).
   Weights are the earned counts (rate * players / 100): a rate built on 7
   players is far noisier in logit terms than one built on 7,000, and the
   weights say so. Log time, not clock time, because earn rates saturate:
   the first hours move a rate far more than the fiftieth.
3. Shrink the slope toward the median slope of all trophies. A trophy with
   two distinct states has a line through two points and no residual; the
   field's slope is a better prior for it than its own noise.
4. Read the shrunk line at the target for the median, clamped so it moves
   at most 2.5 logit from the last read. When the history holds no trend
   for a trophy, what happens depends on the page: if the whole history is
   one flat state, the page is stale and a refresh is due, so the median
   is shifted down by a dilution prior (0.9 logit, half the first
   documented refresh's ~1.8 logit dilution of rare trophies, scaled by
   how much of a refresh interval the staleness plus the horizon covers,
   tapering to zero at 100% where no dilution is possible) and the spread
   is widened; if only that trophy is flat while the page moved, it is an
   equilibrium and the median holds the last rate.
5. Spread: sigma^2 = random-walk^2 + fit^2 + refresh^2. The random walk
   is the house baseline's 0.5 * sqrt(days). The fit term is the line's
   residual plus its slope uncertainty stretched to the target. The refresh
   term is the median |logit jump| between distinct states in this case's
   history (0.75 logit when the history shows no jump at all), grown with
   the square root of the expected number of refreshes and capped, because
   refreshes average out over a long horizon instead of accumulating
   without bound. Sigma is floored at 0.5 and capped at 1.6 logit.

Quantiles are the median shifted by the normal z for each level, back
through the inverse logit, rounded half-up to 4 decimals.

Determinism: pure float arithmetic on the case file, no clock, no random
numbers, no network, standard library only. Two runs write identical bytes.

Robustness: a trophy with no usable rate falls back to 1% with a wide
spread; a case that fails to parse still gets a forecast file, so the run
never crashes on one bad case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Random-walk spread of a rate, logit units per square-root day (house baseline).
WALK = 0.5
# Median shift applied when the whole history is one flat state (the page
# never refreshed): the next refresh is due, and the house's account of the
# first refresh puts a rare trophy's dilution near 1.8 logit (tracked
# players 638 -> 3,868, new players holding ~none of the rare trophies).
# 0.9 is half that: a conservative prior that the next refresh roughly
# doubles the tracked base with fresh players. It tapers to zero at 100%,
# where no dilution is possible, and it never fires for a trophy that is
# flat while the rest of the page moved: that is an equilibrium, not a
# stale page, and its median stays at the last rate.
FLAT_SHIFT = 0.9
# Hours between Exophase refreshes assumed when judging whether one is due.
REFRESH_HOURS = 4.5
# Jump scale used when the history shows no refresh jump at all.
JUMP_PRIOR = 0.75
# Slope prior strength, in distinct states, for the field-median shrinkage.
SHRINK_STATES = 3.0
# Caps that keep an extrapolation inside what the data can support.
MAX_SLOPE = 1.5
MAX_MOVE = 2.5
SIGMA_FLOOR = 0.5
SIGMA_CAP = 1.6
DEFAULT_FIT_SIGMA = 0.35
RESID_FLOOR = 0.15


def pct_clamp(x):
    """Keep a percent inside the range the logit can take."""
    return min(99.99, max(0.01, x))


def to_logit(pct):
    p = pct_clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round_half_up(x, places=4):
    """Round half up to `places` decimals; whole numbers come back as int."""
    scaled = x * (10 ** places)
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / (10 ** places)
    return int(value) if float(value).is_integer() else value


def read_time(text):
    """Parse an ISO 8601 timestamp into an aware UTC datetime."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def release_midnight(game):
    """Midnight UTC of the game's release date, or None if unparseable."""
    raw = (game or {}).get("release_date")
    if not isinstance(raw, str):
        return None
    day = read_time(raw[:10] + "T00:00:00Z")
    return day


def hours_since(moment, start):
    if moment is None or start is None:
        return 0.0
    return max(-0.9, (moment - start).total_seconds() / 3600.0)


def log_clock(hours):
    return math.log1p(max(0.0, hours))


def distinct_states(history, start, count):
    """Collapse consecutive identical rate vectors into one state each.

    Returns a list of (log-time, players, rates-tuple), oldest first. A
    read that repeats the previous vector adds no information: the page
    simply had not refreshed yet.
    """
    states = []
    previous = None
    for read in history or []:
        rates = read.get("rates") if isinstance(read, dict) else None
        if not isinstance(rates, list) or len(rates) != count:
            continue
        key = tuple(rates)
        if key == previous:
            continue
        previous = key
        moment = read_time(read.get("at"))
        players = read.get("players_tracked")
        if not isinstance(players, (int, float)) or players <= 0:
            players = 0
        states.append((log_clock(hours_since(moment, start)), players, key))
    return states


def weighted_line(xs, ys, ws):
    """Weighted least squares of y on x. Returns (intercept, slope, resid).

    resid is the weighted RMS residual, floored: two points determine a
    line exactly, but that says nothing about how well the line will
    extrapolate, so a perfect fit still carries the floor.
    """
    total = sum(ws)
    if total <= 0 or len(xs) < 2:
        return None
    xbar = sum(w * x for x, w in zip(xs, ws)) / total
    ybar = sum(w * y for y, w in zip(ys, ws)) / total
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 0:
        return None
    slope = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws)) / sxx
    intercept = ybar - slope * xbar
    resid = math.sqrt(
        sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / total
    )
    resid = max(resid, RESID_FLOOR)
    slope_se = resid / math.sqrt(sxx)
    return intercept, slope, resid, slope_se, xbar


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def forecast_trophy(states, index, last_rate, pool_slope, jump_scale,
                    horizon_days, horizon_hours, x_target, globally_flat,
                    staleness_hours):
    """Five quantiles for one trophy, from its distinct states."""
    points = []
    for x, players, rates in states:
        rate = rates[index]
        if rate is None:
            continue
        points.append((x, to_logit(rate), max(1.0, rate * players / 100.0)))
    last_logit = to_logit(last_rate if last_rate is not None else 1.0)

    distinct_rates = {round(p[1], 9) for p in points}
    if len(points) >= 2 and len(distinct_rates) >= 2:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ws = [p[2] for p in points]
        fit = weighted_line(xs, ys, ws)
        if fit is not None:
            _, slope, resid, slope_se, xbar = fit
            n = len(points)
            shrunk = (n * slope + SHRINK_STATES * pool_slope) / (n + SHRINK_STATES)
            shrunk = max(-MAX_SLOPE, min(MAX_SLOPE, shrunk))
            centre = sum(w * y for _, y, w in points) / sum(ws)
            xmean = sum(w * x for x, _, w in points) / sum(ws)
            median_logit = centre + shrunk * (x_target - xmean)
            fit_sigma = math.sqrt(
                resid ** 2 + (slope_se * (x_target - xbar)) ** 2
            )
            fit_sigma = min(fit_sigma, 1.0)
            if n < 3:
                # Two points determine a line exactly; that precision is
                # spurious, so a short fit still carries real uncertainty.
                fit_sigma = max(fit_sigma, 0.5)
        else:
            median_logit = last_logit
            fit_sigma = DEFAULT_FIT_SIGMA
    else:
        # No trend to fit. When the whole page is one stale state, a
        # refresh is due: shift the median down for the dilution it brings,
        # scaled by how much of a refresh interval the staleness plus the
        # horizon covers, and tapering to nothing at 100%. When only this
        # trophy is flat while the page moved, it is an equilibrium: hold
        # the last rate with a narrow spread.
        if globally_flat:
            chance = min(1.0, (staleness_hours + horizon_hours) / REFRESH_HOURS)
            taper = 1.0 - pct_clamp(last_rate if last_rate is not None else 1.0) / 100.0
            median_logit = last_logit - FLAT_SHIFT * chance * taper
            fit_sigma = 0.9
        else:
            median_logit = last_logit
            fit_sigma = DEFAULT_FIT_SIGMA

    median_logit = max(last_logit - MAX_MOVE, min(last_logit + MAX_MOVE, median_logit))
    median_logit = max(to_logit(0.01), min(to_logit(99.99), median_logit))

    walk = WALK * math.sqrt(max(0.0, horizon_days))
    refreshes = max(0.5, horizon_hours / 8.0)
    refresh = min(0.6, jump_scale * math.sqrt(refreshes))
    sigma = math.sqrt(walk ** 2 + fit_sigma ** 2 + refresh ** 2)
    sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))

    quantiles = []
    for z in Z:
        q = round_half_up(pct_clamp(from_logit(median_logit + z * sigma)))
        quantiles.append(q)
    for k in range(1, len(quantiles)):
        if quantiles[k] < quantiles[k - 1]:
            quantiles[k] = quantiles[k - 1]
    return quantiles


def fallback_rows(trophies, last_rates):
    """Last rates with a wide spread, used when a case cannot be fitted."""
    rows = []
    for pos, trophy in enumerate(trophies):
        rate = None
        if pos < len(last_rates):
            rate = last_rates[pos]
        centre = to_logit(rate if rate is not None else 1.0)
        quantiles = []
        for z in Z:
            quantiles.append(round_half_up(pct_clamp(from_logit(centre + z * 1.0))))
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return rows


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_minutes = case.get("horizon_minutes") or 0
    horizon_days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0
    start = release_midnight(case.get("game") or {})
    if start is None and history:
        start = read_time(history[0].get("at"))
    target = read_time(case.get("target"))
    x_target = log_clock(hours_since(target, start))

    states = distinct_states(history, start, len(trophies))
    last_rates = list(states[-1][2]) if states else [None] * len(trophies)
    globally_flat = len(states) <= 1
    if history:
        first = read_time(history[0].get("at"))
        checkpoint = read_time(case.get("at"))
        staleness_hours = 0.0
        if first is not None and checkpoint is not None:
            staleness_hours = max(0.0, (checkpoint - first).total_seconds() / 3600.0)
    else:
        staleness_hours = 0.0

    # The field's slope and the typical refresh jump, pooled over trophies.
    slopes = []
    jumps = []
    for index in range(len(trophies)):
        points = []
        for x, players, rates in states:
            rate = rates[index]
            if rate is not None:
                points.append((x, to_logit(rate), max(1.0, rate * players / 100.0)))
        if len(points) >= 2:
            fit = weighted_line([p[0] for p in points],
                                [p[1] for p in points],
                                [p[2] for p in points])
            if fit is not None:
                slopes.append(fit[1])
            trail = [p[1] for p in points]
            jumps.extend(abs(b - a) for a, b in zip(trail, trail[1:]) if b != a)
    pool_slope = median_of(slopes) if slopes else 0.0
    pool_slope = max(-MAX_SLOPE, min(MAX_SLOPE, pool_slope))
    jump_scale = median_of(jumps) if jumps else JUMP_PRIOR
    jump_scale = max(0.1, min(1.5, jump_scale))

    rows = []
    for index, trophy in enumerate(trophies):
        rate = last_rates[index] if index < len(last_rates) else None
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(states, index, rate, pool_slope, jump_scale,
                                 horizon_days, horizon_hours, x_target,
                                 globally_flat, staleness_hours),
        })
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                trophies = case.get("trophies") or []
                last = case.get("history") or []
                rates = list(last[-1].get("rates") or []) if last else []
                forecast = {"trophies": fallback_rows(trophies, rates)}
            except Exception:
                forecast = {"trophies": []}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
