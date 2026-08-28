# Step 25 Method Note: Regularized Tweedie Reserving

## Objective

Step 25 evaluates one Regularized Tweedie reserving model as a controlled comparison against the existing baseline Regularized Poisson model. The experiment tests whether changing the conditional variance assumption improves reserve accuracy while holding the data, predictors, validation design, regularisation framework, simulation seeds, and evaluation truth fixed.

Model accuracy is a research result, not a technical acceptance criterion. Technical acceptance concerns implementation correctness, reproducibility, reconciliation, pairing, and complete reporting.

## Tweedie model

For incremental paid loss \(Y\), the Tweedie model assumes

\[
\operatorname{E}(Y)=\mu, \qquad
\operatorname{Var}(Y)=\phi\mu^p,
\]

where \(\phi\) is a dispersion parameter and \(p\) is the Tweedie variance power. The fitted model is `sklearn.linear_model.TweedieRegressor` with the log link,

\[
\log(\mu)=X\beta.
\]

The frozen candidate variance-power grid is:

- \(p \in \{1.1, 1.3, 1.5, 1.7, 1.9\}\).

No alternative link functions or additional distributional variants are considered in Step 25.

## Regularisation

The model uses L2 regularisation. The dedicated Tweedie alpha grid is frozen to the existing Regularized Poisson alpha grid:

- `0.0001`
- `0.001`
- `0.01`
- `0.1`
- `1.0`
- `10.0`

Variance power and alpha are selected jointly.

## Predictors and preprocessing

The Regularized Tweedie model uses exactly the baseline Regularized Poisson feature set:

- `accident_year_centered`
- `relative_log_inflation`
- `structural_break_indicator`
- categorical `development_year`

Numeric predictors are standardised and development year is one-hot encoded through the existing pipeline conventions. Step 24 calendar-regime × development-year interaction features are deliberately excluded; no Tweedie break-interaction model is fitted.

The model reuses `triangle_to_cell_dataset()` and its existing observed/future cell definitions. Incremental paid amounts are divided by `1,000,000` before fitting and predictions are converted back to pounds afterwards. This scaling is for numerical stability and does not change the economic quantity being estimated.

## Hyperparameter validation

Hyperparameter selection uses the existing rolling calendar-year diagonal validation framework. For each candidate power × alpha pair:

1. training is restricted to cells observed before the held-out historical calendar-year diagonal;
2. the model is fitted only to those historical training cells;
3. predictions are evaluated on the held-out historical cells;
4. held-out mean absolute error (MAE) is recorded at fold level;
5. fold MAE is aggregated by power and alpha;
6. the existing minimum-successful-validation-fold rule is enforced; and
7. the pair with the lowest mean historical validation MAE is selected, using deterministic tie-breaking.

MAE is the common selection criterion across different Tweedie power values. The selected model is then refitted on all cells observed at the valuation date and used to predict future cells.

## Leakage safeguards

Evaluation reserve truth is never passed to the fitting API and is not used in feature construction, model fitting, validation-fold construction, or hyperparameter selection. It is accessed by the experiment runner only after the portfolio and triangles have been created, for evaluation and reporting of reserve errors.

The Expected Loss pricing prior and other dissertation model components are not used to tune this model. Hyperparameters are selected exclusively from historical observed triangle cells.

## Experimental design and pairing

The final experiment comprises:

- 9 frozen scenarios;
- 50 simulations per scenario;
- 2 bases (`gross` and `ceded`); and
- 2 models (`regularized_poisson` and `regularized_tweedie`).

This gives \(9 \times 50 \times 2 \times 2 = 1{,}800\) model attempts.

Within each scenario and simulation, Regularized Poisson and Regularized Tweedie use the same simulated portfolio, the same gross and ceded triangles, the same evaluation truth, and the same seed. Gross and ceded results are evaluated separately. Model fitting success and applicability are reported separately from conditional accuracy among successful fits.

Regularized Poisson is the sole comparator for Step 25. No statistical-significance claim is made here; paired statistical comparisons are reserved for Step 31.
