from pathlib import Path

import pytest

from soccer_tactics.loaders import load_metrica_game

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "metrica"
GAMES = [g for g in (1, 2) if (ROOT / "data" / f"Sample_Game_{g}").exists()]


@pytest.fixture(scope="session")
def games():
    """{game: (tracking, events)}, loaded once per test session."""
    return {g: load_metrica_game(ROOT, g) for g in GAMES}
