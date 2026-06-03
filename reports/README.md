# Reports

`final_results/` contains local validation tables and figures used for the
project report. These files are generated from executed notebook outputs by:

```bash
python3 scripts/build_report_artifacts.py --skip-patch
```

The notebook also writes a fresh report bundle to
`/kaggle/working/report_artifacts/` when it is run on Kaggle.
