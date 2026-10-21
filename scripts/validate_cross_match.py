"""Descriptive held-out Metrica validation (local source files only).

Run ``python scripts/validate_cross_match.py --metrica data/raw/metrica``.
Legacy placeholder/IDSSE numerical experiments remain available without --metrica.
"""
from pathlib import Path

import pandas as pd


def _shot_rate(rows):
    numerator = int((rows.shots > 0).sum())
    return dict(numerator=numerator, denominator=len(rows), rate=numerator / len(rows) if len(rows) else None)


def summarize_v4_fold(report, other_defense):
    """Compare selected-team sequences and pooled other-match team exposure.

    Home/away identities are not linked between anonymized matches. Pool *both*
    teams' action counts and observed team-minutes, separately for each action.
    """
    other_defense = list(other_defense)
    ids = [match_id for match_id, _ in other_defense]
    if report.match_id in ids or len(ids) != len(set(ids)):
        raise ValueError('held-out match and duplicate matches cannot enter dataset baseline')
    if not ids:
        raise ValueError('dataset baseline needs another match')
    baseline = pd.concat([rows for _, rows in other_defense], ignore_index=True)
    defense = []
    for row in report.defensive_actions.query('team == @report.team').itertuples():
        pooled = baseline[baseline.action_type == row.action_type]
        minutes = float(pooled.observed_tracking_minutes.sum())
        actions = int(pooled.total_actions.sum())
        defense.append(dict(action_type=row.action_type, total_actions=int(row.total_actions),
            observed_tracking_minutes=float(row.observed_tracking_minutes), actions_per_90=row.actions_per_90,
            baseline_actions=actions, baseline_observed_team_minutes=minutes,
            dataset_baseline_per_90=actions*90/minutes if minutes else None))
    shots = report.shot_summaries.query('team == @report.team')
    return dict(match_id=report.match_id, team=report.team, baseline_match_ids=ids,
        baseline_label='Cross-match dataset baseline: pooled anonymous teams in the other matches; not the same team',
        transition_shot_rate=_shot_rate(report.transitions.query('team == @report.team')),
        possession_shot_rate=_shot_rate(report.possessions.query('team == @report.team')),
        defense=defense,
        shots_by_channel=[dict(channel=row.channel, shots=int(row.shots), goals=int(row.goals),
            team_shots=int(shots.shots.sum()), team_goals=int(shots.goals.sum())) for row in shots.itertuples()],
        interpretation='Descriptive counts only; no significance, causal effect, or learned shot-quality estimate.')


def run_metrica(root, output):
    # Use exactly the same audited reports and folds as the release gate.
    if __package__:
        from scripts.validate_release import run
    else:
        from validate_release import run
    run(Path(root), Path(output))


def main():
    import sys
    args = sys.argv[1:]
    if args and args[0] == '--metrica':
        import argparse
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--metrica', type=Path, required=True)
        parser.add_argument('--output', type=Path, default=Path('docs/validation/v4'))
        parsed = parser.parse_args()
        run_metrica(parsed.metrica, parsed.output)
        return
    from soccer_tactics.analytics.cross_match import leave_one_match_out, summarize_cross_match
    from soccer_tactics.loaders import load_idsse_game, load_placeholder_matches
    if not args:
        matches = load_placeholder_matches()
    else:
        root, *match_ids = args
        if len(match_ids) < 2:
            raise SystemExit('provide an IDSSE directory and at least two match ids')
        matches = [load_idsse_game(root, match_id) for match_id in match_ids]
    passes, players = leave_one_match_out(matches)
    print(summarize_cross_match(passes, players))


if __name__ == '__main__':
    main()
