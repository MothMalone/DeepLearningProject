# Final Method

The final run is a full fine-tune of `NlpHUST/gpt2-vietnamese` on compact
Vietnamese math targets. The notebook builds each target from the provided
solution, keeps short arithmetic chains when they are extractable, and always
ends with one answer anchor:

```text
####đáp án là: <number>
```

## Training

The training loss masks the prompt and upweights two parts of the completion:

| Span | Purpose |
|---|---|
| arithmetic tokens near `=` | keep equations from collapsing into prose |
| final answer anchor | make answer extraction stable |
| EOS | stop after the answer rather than drifting |

The final configuration is stored in `configs/final/rewind3_best.json` and in
the first config cell of `notebooks/final_experiment.ipynb`.

## Model Selection

The notebook saves one checkpoint per epoch. Each checkpoint is evaluated by
generation on validation examples, then the best checkpoint is selected by raw
validation score. This matters because teacher-forced validation loss and
generated answer quality do not peak at the same time.

In the 500-example reference run, validation loss was already rising by epoch 7,
but generated score still peaked at epoch 7. For the 1,000-example reference
run, the best completed decoding result was:

```text
sc21_t05: 3285 / 10000
```

## Decoding

Self-consistency decoding outperformed beam search. The strongest completed
decoding comparison on 1,000 validation examples was:

| Config | Score |
|---|---:|
| `sc15_t04` | 3163 / 10000 |
| `sc21_t04` | 3144 / 10000 |
| `sc21_t05` | 3285 / 10000 |

The final notebook keeps `sc21_t05` as the fallback and also tests a nearby
larger sample configuration.

## Data Checks

The main data-centric changes were:

- preserve grouped LaTeX fractions as `((num)/(den))`;
- strip Asymptote diagram code before target construction;
- avoid treating common Vietnamese words such as `bay` as digits;
- keep validation/test records separate from training filters;
- deduplicate train queries after target filtering.

Dataset and result audits are implemented in:

```text
scripts/audit_dataset_quality.py
src/vn_gpt2_math/dedup.py
src/vn_gpt2_math/reporting.py
```

## Remaining Weakness

Most residual errors are arithmetic mistakes rather than formatting failures.
`GSM_Rephrased` and `GSM_AnsAug` are the strongest groups, while `SV` and
`FOBAR` remain weak because they require stable variable tracking under
perturbed wording.
