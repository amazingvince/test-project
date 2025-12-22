# Scripts

This folder contains small, user-facing utilities and one-off workflows.

**Common utilities**
- `scripts/tokenizer_probe.py`: Inspect tokenization of `<think>`/`<uci_move>` tags and sample streamed data.
- `scripts/gut_check_report.py`: End-to-end sanity check; writes a Markdown report with prompt + trace examples and can auto-preprocess a small dataset.
- `scripts/download_openings.py`: Download opening TSVs into `./data/openings`.
- `scripts/download_tablebases.py`: Download Syzygy tablebases into `./data/syzygy`.
- `scripts/global_chess_challenge_2025_setup.py`: Clone the AIcrowd starter kit into `./tmp/`, copy prompt templates, and print local-eval commands.

**Convenience wrappers**
- `scripts/train_sft.sh`, `scripts/train_distill.sh`, `scripts/evaluate.sh`: Bash helpers for common runs.
