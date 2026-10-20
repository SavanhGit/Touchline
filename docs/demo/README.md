# Touchline demo

Start with [the featured game 1 away analysis](featured-analysis.md), then [watch the walkthrough](walkthrough.mp4). Data is from [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data); team/player identities are anonymous. Source CSVs remain external to Git.

## Screenshots

The controller captured and visually inspected these real dashboard views using game 1, away perspective, at readable desktop size:

- [Overview](overview.png): 3–0 result, measured summaries and hypotheses.
- [Risk & progression](risk-progression.png): loss locations and completed entry routes.
- [Players](players.png): selected-team pass metrics and eligibility context.
- [Evidence](evidence.png): player positions, fixed source arrow, timestamp and tracking reconstruction label.

## Walkthrough and manual checks

`walkthrough.mp4` is a **54-second captioned sampled browser walkthrough**, not seamless real-time screen recording or match video. It packages 54 real browser captures from an interactive local session, sampled roughly once per second during selected segments and encoded at 12 fps by repeating each capture for one second. Time between segments is omitted; there is no interpolated player motion. The same format can be described as a slideshow of sampled browser frames. Continuous screen recording remains unfinished.

The captured sequence visits Overview → Risk & progression → Players → Evidence, selects another supporting moment, moves the evidence slider to −3 and +3 seconds, toggles simplified control, and switches to game 2. Captions explain measured evidence versus hypotheses and identify Metrica Sports. Evidence is a tracking reconstruction. Control omits ball flight and acceleration and does not estimate pass success.

Verification on Windows with Python 3.12.14:

- Encoded H.264 MP4, 1440 × 1000, 12 fps, 648 frames, 54 seconds. Browser content occupies 1440 × 900; a separate caption band adds 100 pixels.
- Decoded and visually inspected MP4 frames at 0, 27 and 53 seconds. Opening shows game 1 away, midpoint shows the evidence selector/source ID/clock, ending shows game 2 away and its 3–2 result. Captions remain readable. The sampled evidence viewport clips the bottom of the pitch at some moments; use the full Evidence screenshot for detailed positions and arrow inspection.
- A clean ignored `.venv-check` was created from Python 3.12.14, installed with `-e ".[dev,dashboard,demo]" -c requirements-demo.txt`, and checked with `python -m pip check`: `No broken requirements found.`
- Local Markdown links and media paths were checked for existence; raw data, environments and capture scratch files were checked against Git ignore rules.
- The clean environment launched Streamlit on port 8517 and its HTTP endpoint returned 200. The controller's browser session reached the dashboard and supplied the real interaction captures; the normal launch command is documented in the [main README](../../README.md).

The demo makes no claim that a tactical adjustment would change the score. Five relevant events and three supporting events are display safeguards, not significance tests. Review the [validation notes](../VALIDATION.md) and [all four report exports](release-validation.json).
