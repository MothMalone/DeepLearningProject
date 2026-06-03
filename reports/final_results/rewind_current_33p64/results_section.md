# Results: 500-Example Rewind Reference

This run scored **1682 / 5000** on the 500-example validation subset
(**33.64%**), with **157 exact answers**.

Epoch selection mattered. The generated validation score rose through epoch 7
and then dipped at epoch 8:

```text
epoch 6: 1095
epoch 7: 1170
epoch 8: 1136
```

The teacher-forced validation loss was already rising by epoch 7, so loss alone
would have selected a different checkpoint. For this task, generated numeric
score was the better checkpoint-selection signal.

The strongest sub-track was `GSM_Rephrased` with a mean score of 5.68 / 10.
The weakest groups were `GSM_SV` and `MATH_FOBAR`, which matches the qualitative
error analysis: the model handles paraphrased arithmetic better than variable
tracking or perturbed symbolic wording.

The dominant failure mode was `arithmetic_mistake` with 235 cases. Formatting
and extraction were no longer the main bottleneck.
