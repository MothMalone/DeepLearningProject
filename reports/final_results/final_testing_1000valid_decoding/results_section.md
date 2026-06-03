# Results: 1000-Example Decoding Reference

This reference run used 1,000 validation examples and scored **3285 / 10000**
(**32.85%**), with **303 exact answers**.

The decoding sweep was the main result from this run:

| Decode config | Score |
|---|---:|
| `sc15_t04` | 3163 / 10000 |
| `sc21_t04` | 3144 / 10000 |
| `sc21_t05` | 3285 / 10000 |

The best completed configuration was `sc21_t05`: 21 sampled completions,
temperature 0.5, `top_k=50`, and `top_p=0.95`.

The sub-track split stayed consistent with the smaller validation run.
`GSM_Rephrased` was strongest at 6.25 / 10, while `GSM_SV`, `GSM_FOBAR`, and
`MATH_FOBAR` remained the hardest groups. This suggests that the remaining
problem is not answer formatting, but unstable arithmetic and variable tracking.
