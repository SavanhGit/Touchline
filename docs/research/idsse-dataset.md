# IDSSE dataset verification

Verified 2026-10-18 against the dataset's official repository, public
Figshare API record, and the linked publication.

## Recommendation

Use **IDSSE: An integrated dataset of spatiotemporal and event data in elite
soccer** as the second source after Metrica. It contains both event and
observed player/ball position data from seven full Bundesliga matches, which
allows same-match tracking/event alignment and leave-one-match-out evaluation.

## Primary sources

- Official repository and processing code:
  https://github.com/spoho-datascience/idsse-data
- Public dataset record and file manifest:
  https://api.figshare.com/v2/articles/28196177
- Publication:
  https://doi.org/10.1038/s41597-025-04505-y
- License:
  https://creativecommons.org/licenses/by/4.0/

## Facts used by the implementation

- The repository identifies the source as official German Bundesliga match
  information, event data, and TRACAB position data.
- The dataset record is public, version 1, and licensed CC BY 4.0. Attribution
  must name the Deutsche Fussball Liga (DFL) and cite the publication.
- The record exposes three XML file families per match:
  `matchinformation`, `events_raw`, and `positions_raw_observed`.
- The public record reports seven full matches and a total download size of
  about 2.63 GB.
- The official repository loads the XML through Floodlight's
  `read_event_data_xml` and `read_position_data_xml` readers.

## Local usage

Download the files from the Figshare record, place them in one directory, and
install the optional parser:

```text
pip install -e ".[idsse]"
```

Then:

```python
from soccer_tactics.loaders import list_idsse_matches, load_idsse_game

match_ids = list_idsse_matches("data/raw/idsse")
tracking, events = load_idsse_game("data/raw/idsse", match_ids[0])
```

The loader keeps the source match id, maps first and second halves to periods
1 and 2, preserves observed positions in metres, and emits the repository's
canonical tracking/event schemas. Event conversion is intentionally
conservative: only pass/cross and shot events with usable coordinates are
included.

## Limitations

The dataset is large, the XML reader is optional, and source synchronization
must be checked before treating event-frame joins as exact. The loader is
therefore separate from Metrica and the evaluation path scores each held-out
match using an xT surface fitted on the other matches.
