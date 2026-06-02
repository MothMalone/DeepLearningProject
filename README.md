# Vietnamese GPT-2 Math Word Problem Fine-Tuning

This repository contains our deep-learning project for fine-tuning the fixed
`NlpHUST/gpt2-vietnamese` backbone on Vietnamese math word problems.

The project started as Kaggle notebooks and was later organized into reusable
modules, experiment configs, and reporting scripts. The final empirical core is
`rewind3.ipynb`, which produced the best local validation result among our
rule-compliant runs.

## Best Local Result

Validation subset: 500 examples from `valid.json`

| Metric | Value |
|---|---:|
| Raw score | 1682 / 5000 |
| Score percentage | 33.64% |
| Exact answers | 157 / 500 |
| Extractable answers | 491+ / 500 depending on decoding |
| Main failure mode | Arithmetic mistake |

The final stretch showed that the largest gain came from self-consistency
decoding on the best epoch checkpoint, not from changing target construction.

## Constraints

The official Kaggle notebook must satisfy:

- Internet OFF
- GPU accelerator
- fixed model backbone: `NlpHUST/gpt2-vietnamese`
- use only provided `train.json`, `valid.json`, and optional `test.json`
- no external APIs, online solvers, LoRA/PEFT, GRPO, teacher distillation, or extra data
- no test-time learning
- output `/kaggle/working/test_predictions.json` without gold-answer fields

## Repository Layout

```text
.
├── rewind3.ipynb                         # final/best Kaggle notebook core
├── kaggle_vietnamese_gpt2_math_notebook.py
├── run_vast_experiments.sh               # historical Vast.ai runner
├── analyze_experiment_results.py
├── src/vn_gpt2_math/                     # reusable research modules
├── train.py                              # modular training entrypoint skeleton
├── evaluate.py                           # evaluate prediction JSON against valid.json
├── infer.py                              # frozen checkpoint inference entrypoint
├── configs/                              # final and ablation configs
├── scripts/                              # preprocessing and result-inspection tools
├── tests/                                # parsing/metric smoke tests
└── docs/                                 # method, history, and submission notes
```

## Reproducing the Final Kaggle Run

1. Open `rewind3.ipynb` in Kaggle.
2. Attach the two Kaggle inputs:
   - `kimanh2002/dataset-math`
   - `kimanh2002/nlphustgpt2-vietnamese`
3. Set Internet OFF and GPU ON.
4. Run all cells.
5. Confirm these files exist:
   - `/kaggle/working/valid_output.json`
   - `/kaggle/working/valid_report.json`
   - `/kaggle/working/test_predictions.json`
   - `/kaggle/working/hpo_report.json`

## Local Smoke Checks

These checks do not train a model:

```bash
python3 -m unittest discover -s tests
python3 -m py_compile train.py evaluate.py infer.py analyze_experiment_results.py
```

## Research Summary

We tried several families of approaches:

- plain SFT baselines
- answer-only and answer-stub SFT
- equation-focused and short-solution SFT
- tagged `<think>/<answer>` warmups
- GRPO/RLVR preflight experiments
- local-LLM preprocessing/distillation workflow
- FDD/PoT-inspired compact arithmetic targets
- final rewind3 weighted-loss + self-consistency decoding

Most target rewrites improved extractability but did not reliably improve numeric
correctness. The best rule-compliant result came from preserving the rewind3
compact target/filter pipeline and adding better checkpoint/decoding selection.
