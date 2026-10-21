"""Explicit opt-in download of only the Roboflow player and pitch checkpoints."""
import argparse
from pathlib import Path


MODEL_IDS = {
    'football-player-detection.pt': '17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q',
    'football-pitch-detection.pt': '1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/models/video'))
    args = parser.parse_args()
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit('Install soccer-tactics[cv] before downloading the two weights.') from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, file_id in MODEL_IDS.items():
        target = args.output_dir / name
        if target.is_file():
            print(f'Already present: {target}')
            continue
        temporary = target.with_suffix('.pt.part')
        try:
            downloaded = gdown.download(id=file_id, output=str(temporary), quiet=False)
            if not downloaded or not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f'Could not download {name}; retry or provide the local checkpoint manually.')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
