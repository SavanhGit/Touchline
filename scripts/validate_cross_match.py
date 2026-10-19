"""Leave-one-match-out evaluation over canonical tracking/event pairs.

Usage:
    python scripts/validate_cross_match.py
    python scripts/validate_cross_match.py <idsse-directory> <match-id> [...]

With no arguments, the script runs against two deterministic placeholder matches.
With arguments, it loads the requested IDSSE matches. The script intentionally
does not download the 2.63 GB public archive.
"""
import sys

from soccer_tactics.analytics.cross_match import leave_one_match_out, summarize_cross_match
from soccer_tactics.loaders import load_idsse_game, load_placeholder_matches


args = sys.argv[1:]
if not args:
    matches = load_placeholder_matches()
else:
    root, *match_ids = args
    if len(match_ids) < 2:
        raise SystemExit("provide an IDSSE directory and at least two match ids")
    matches = [load_idsse_game(root, match_id) for match_id in match_ids]

passes, players = leave_one_match_out(matches)
print(summarize_cross_match(passes, players))
