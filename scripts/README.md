# scripts/ — everyday tools

| File | Command | What it does |
|---|---|---|
| `run_video.py` | `python scripts/run_video.py --video data/videos/22.mp4 --signal AUTO` | Runs the full pipeline on a video and prints a summary |
| `seed_demo_data.py` | `python scripts/seed_demo_data.py` (`--remove` to undo) | Adds watermarked DEMO records so every dashboard page has data |
| `clean_test_data.py` | `python scripts/clean_test_data.py --apply` | Removes rows that old test runs wrote into the real database |
