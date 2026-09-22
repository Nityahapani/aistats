# Causal Estimands and Theoretical Framing

## 1. Precisely what we claim causally

### Estimand 1 (ATE): does stability-weighted selection reduce the gap?

$$\tau = \mathbb{E}[G_{\mathrm{control}} - G_{\mathrm{centroid}}]$$

where:
- $G_{\mathrm{control}} = L_{\mathrm{test}}(\hat{h}^*) - \hat{L}_{\mathrm{val}}(\hat{h}^*)$ — gap from standard HPO winner
- $G_{\mathrm{centroid}} = L_{\mathrm{test}}(\hat{h}^{\dagger}) - \hat{L}_{\mathrm{val}}(\hat{h}^{\dagger})$ — gap from centroid of bootstrap winners
- Expectation is over datasets, searchers, and random seeds

**Estimate:** $\hat{\tau} = +0.0444$, 95% CI $[+0.0368, +0.0522]$, $n=180$ pairs.

This is an ATE in a **paired experiment** where the randomization unit is the run (not the arm),
and treatment assignment (centroid vs original) is deterministic given the bootstrap winners.
The causal identification rests on:
1. The centroid is computed from the same development data as the original winner
2. OSC is computed before any test evaluation
3. The test set is never seen during OSC computation or treatment selection

### Estimand 2 (Moderation): does OSC moderate the treatment effect?

$$\tau(\mathrm{OSC}) = \alpha + \beta \cdot \mathrm{OSC}$$

where $\tau(\mathrm{OSC}) = \mathbb{E}[G_{\mathrm{control}} - G_{\mathrm{centroid}} \mid \mathrm{OSC}]$.

**Estimate:** $\hat{\beta} = -0.092$ (paired model), 95% CI $[-0.123, -0.064]$,
equivalent to interaction $\hat{\beta}_3 = +0.081$ in the joint model.

**This is NOT claiming "OSC is causal."** What we claim:
- OSC, computed pre-treatment on dev data, *predicts* how much the treatment will help
- Experimentally induced variation in OSC (via val-set noise) produces corresponding variation in $\tau$
- The mechanism: high val noise → low OSC (unstable selection) → original winner unreliable → centroid more stable → larger gap reduction

The causal chain is: **evaluation noise → low OSC → large selection error → large treatment benefit**.
OSC is a *mediator* of noise, not an independent cause.

---

## 2. What we do NOT claim

- We do not claim OSC *causes* large generalization gaps. OSC is a diagnostic computed from
  the same noisy process that creates the gap. Correlation ≠ causation here.
- We do not claim centroid selection is universally optimal. In this experiment it always helps,
  but that is specific to the SVM-RBF search space where the landscape is smooth.
- We do not claim the bootstrap estimate of OSC is consistent for any specific population quantity
  without further assumptions about the data-generating process.

---

## 3. Three empirical claims (paper structure)

### Claim 1 — Near-optimal selection stability predicts post-selection gap
*Measurement* (observational, n=162)

OSC, not WS or SE, explains generalization gap after controlling for task, searcher,
dataset, and budget:
- Partial $R^2 = 0.078$, OLS coef $= -0.029$, $p = 0.0004$
- LMM with $(1|\mathrm{dataset})$: coef $= -0.035$, $z = -4.13$, $p < 0.001$
- Grouped CV: ΔMAE $= +0.00175$, 95% CI $[+0.00056, +0.00293]$, $P(\Delta > 0) = 0.998$

### Claim 2 — OSC is a calibrated, transferable diagnostic
*Prediction* (grouped cross-validation + calibration)

- Calibration bins (pre-specified): E[G|OSC<0.4] = 0.028 vs E[G|OSC≥0.8] = 0.006,
  Cohen $d = 1.29$, $p < 0.001$
- ε-sensitivity: signal flat from ε=0.005 to ε=0.30 (Spearman $r ≈ -0.49$ throughout)
- Cross-dataset prediction: OSC improves MAE in every held-out fold

### Claim 3 — OSC identifies where stability-weighted selection helps most
*Intervention* (randomized experiment, n=180 pairs)

- ATE: $\hat{\tau} = +0.044$, 95% CI $[+0.037, +0.052]$, permutation $p < 10^{-5}$
- Moderation: $\hat{\beta} = -0.092$, 95% CI $[-0.123, -0.064]$, $P(\beta < 0) = 1.000$
- Manipulation check: experimentally induced OSC shifts via val noise (F=25.4, p<0.001)
- Interaction: Treatment×OSC = $+0.081$, CI $[+0.051, +0.110]$, $P(>0) = 1.000$
- 5.1× variation in treatment benefit across OSC range [0, 1]

---

## 4. Theoretical contribution: near-optimal selection stability

### The central conceptual distinction

**Configuration-space stability (WS):** Does the HPO procedure repeatedly select the
same *configuration*?
$$\mathrm{WS} = P(\hat{h} \approx \hat{h}^* \text{ in } \ell_2\text{-config space})$$

**Performance-space stability (OSC):** Does the HPO procedure repeatedly select
configurations whose *performance* lies near the optimum?
$$\mathrm{OSC}_\varepsilon = P(\hat{h} \in \mathcal{H}_\varepsilon) \quad \text{where} \quad
\mathcal{H}_\varepsilon = \{h : L(h) \leq L(h^*) + \varepsilon\}$$

### Why this distinction matters

A procedure can have high WS and low OSC: it always selects the same configuration,
but that configuration is a consistent noise artifact — overfitting to the validation split.
Empirically: E[G | high WS, low OSC] = +0.020 ≈ E[G | low WS, low OSC] = +0.020.

A procedure can have low WS and high OSC: different configurations are selected each time,
but they all lie near the optimal region — the procedure is exploring near-optimal
alternatives, not failing. Empirically: E[G | high OSC, low WS] = +0.005 ≈
E[G | high OSC, high WS] = +0.004.

### Formal statement

Let $\hat{h}(\mathcal{D})$ be the configuration selected by an HPO procedure on dataset $\mathcal{D}$.
Define:

$$\mathrm{OSC}_\varepsilon(\hat{h}, \mathcal{D}) = \mathbb{P}_{\mathcal{D}^*}
\left( L(\hat{h}(\mathcal{D}^*)) \leq L(\hat{h}(\mathcal{D})) + \varepsilon \cdot \mathrm{range}(\mathcal{D})
\right)$$

where $\mathcal{D}^*$ is a bootstrap resample of the development set, and $\mathrm{range}(\mathcal{D})$
is the range of validation losses observed across bootstrap winners.

**Proposition (informal):** Under mild regularity conditions on the loss landscape,
$\mathrm{OSC}_\varepsilon \to 1$ as the validation set size $n_{\mathrm{val}} \to \infty$
for any fixed $\varepsilon > 0$. Moreover, $\mathbb{E}[G] \leq f(\mathrm{OSC}_\varepsilon)$
for a decreasing function $f$, where the relationship is empirically linear in our experiments.

**Bootstrap estimator:** We estimate $\mathrm{OSC}_\varepsilon$ by:
$$\widehat{\mathrm{OSC}}_\varepsilon = \frac{1}{B} \sum_{b=1}^B
\mathbf{1}\left[\hat{L}_{\mathrm{val}}^{(b)}(\hat{h}^{(b)}) \leq
\hat{L}_{\mathrm{val}}(\hat{h}) + \varepsilon \cdot \widehat{\mathrm{range}}\right]$$

where $\hat{h}^{(b)}$ is the winner on the $b$-th bootstrap resample.

### Connection to Rashomon sets

OSC is related to, but distinct from, the Rashomon set (Rudin et al.):
$$\mathcal{R}_\varepsilon = \{h : L(h) \leq L(h^*) + \varepsilon\}$$

The Rashomon set asks: what configurations are nearly as good as the best?
OSC asks: does the HPO procedure consistently land in the near-optimal set?
These are complementary: a large Rashomon set means many configurations are good;
high OSC means the procedure finds them reliably.
