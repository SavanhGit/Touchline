# Touchline demo

Start with [the featured game 1 away analysis](featured-analysis.md), then [watch the walkthrough](walkthrough.mp4). Data is from [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data); team/player identities are anonymous. Source CSVs remain external to Git.

## Screenshots

These real dashboard views were captured and visually inspected using game 1, away perspective, at readable desktop size:

- [Overview](overview.png): 3–0 result, measured summaries and hypotheses.
- [Risk & progression](risk-progression.png): loss locations and completed entry routes.
- [Players](players.png): selected-team pass metrics and eligibility context.
- [Evidence](evidence.png): player positions, fixed source arrow, timestamp and tracking reconstruction label.

## Walkthrough and manual checks

`walkthrough.mp4` is an **80-second captioned edited browser walkthrough**, encoded from 240 sequential real browser captures at 3 fps. It preserves live dashboard interaction at a low frame rate. Capture timing is normalized to one frame every third of a second, so playback is edited rather than a seamless high-frame-rate screen recording. It is not match video; pitch content is tracking reconstruction. No interpolated player motion is added.
The captured sequence visits Overview → Risk & progression → Players → Evidence, selects another supporting moment, moves the evidence slider to −3 and +3 seconds, toggles simplified control, and switches to game 2. Captions explain measured evidence versus hypotheses and identify Metrica Sports. Evidence is a tracking reconstruction. Control omits ball flight and acceleration and does not estimate pass success.

Verification on Windows with Python 3.12.14:

- Encoded H.264 MP4, 1440 × 1000, 3 fps, 240 frames, 80 seconds. Browser content occupies 1440 × 900; a separate caption band adds 100 pixels.
- Decoded the entire 240-frame MP4 without a decode failure and visually inspected frames at 0, 40 and 79.67 seconds. Opening shows game 1 away, midpoint shows the Players table, ending shows game 2 away and its 3–2 result. Captions remain readable. Additional sequence inspection confirms evidence selection, slider endpoints, full-pitch reconstruction, control and match switching. Some viewports clip lower content while scrolling; the dedicated Evidence screenshot provides a full-pitch reference.
- Native in-app browser playback over local HTTP ran from start to the 80-second ending. Visual checks during playback confirmed Overview, risk/channel chart, Players and the game-2 ending with readable captions.
- A clean ignored `.venv-check` was created from Python 3.12.14, installed with `-e ".[dev,dashboard,demo]" -c requirements-demo.txt`, and checked with `python -m pip check`: `No broken requirements found.` Installation succeeded despite a nonfatal pip latest-version-check warning.
- Local Markdown links and media paths were checked for existence; raw data, environments and capture scratch files were checked against Git ignore rules.
- The clean environment launched Streamlit on port 8517 and its HTTP endpoint returned 200. The recorded browser session reached the dashboard and captured real interactions; the normal launch command is documented in the [main README](../../README.md).

The demo makes no claim that a tactical adjustment would change the score. Five relevant events and three supporting events are display safeguards, not significance tests. Review the [validation notes](../VALIDATION.md) and [all four report exports](release-validation.json).
