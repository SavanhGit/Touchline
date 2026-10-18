# soccer-tactics

Tracking-data analytics engine for soccer: movement prediction, zone and player risk, defensive weakness detection, with an LLM agent on top.

## Layout

```
src/soccer_tactics/
  schema.py       # the one tracking format every source converts into
  loaders/        # dataset -> schema (Metrica sample data first)
  analytics/      # features and models (velocity, pitch control, xT, trajectory forecasting)
  agent/          # LLM tool-calling layer over analytics
  cv/             # phase 2: video -> tracking, must emit the same schema
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
git clone --depth 1 https://github.com/metrica-sports/sample-data data/raw/metrica
pytest
```

## Roadmap

1. Schema + loaders (done: Metrica sample games 1-2)
2. Features: velocity, acceleration, possession phases
3. Models: pitch control, expected threat, trajectory forecasting
4. Player impact: build-up detriment and on-field impact scores
5. Agent layer: natural-language queries backed by the analytics
6. CV pipeline: roboflow/sports -> same tracking schema
