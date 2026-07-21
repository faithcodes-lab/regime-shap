# Concepts

This page sets out the method behind `regime-shap` in full. It is the reference for what the
package computes and why.

## The question

A model can achieve a good overall score while relying on features that are not dependable. If
the features a model leans on change from one period to the next, its reasoning is unstable even
when its accuracy looks fine, and that instability is a risk for anyone who has to trust the
model. `regime-shap` measures this directly. It asks whether a model relies on the same features
across different regimes, or whether it quietly reorganises which features matter.

## Regimes

A regime is a distinct time period, for example a market crisis, a policy change, or an economic
era. In this package regimes are contiguous periods in the spirit of structural breaks, not
recurring categories. The unit the package works with is a per-row label saying which regime each
row of the feature matrix belongs to.

You can supply regimes in three ways:

- **Per-row labels.** A label for every row, the most general and domain-agnostic form.
- **Date ranges.** For date-indexed data, a list of `(start, end)` ranges or a
  `{label: (start, end)}` mapping, which the package resolves to per-row labels.
- **Automatic detection.** When you have no regimes in mind, `detect_breaks` proposes change
  points from a series using the `ruptures` change-point library, which you then turn into labels.

## Feature importance from SHAP

For each row, SHAP assigns every feature a value that is its contribution to that row's
prediction. `regime-shap` computes these once with `shap.TreeExplainer`, which gives exact SHAP
values for tree models.

The importance of a feature is the **mean of the absolute SHAP values** for that feature across a
set of rows. Taking the absolute value first means a feature that pushes predictions up on some
rows and down on others is credited for the size of its influence, not cancelled out. Two views
follow from this:

- **Global importance** averages the absolute SHAP values over all rows.
- **Per-regime importance** averages them within each regime's rows, giving one importance profile
  per regime.

## Comparing regimes with Spearman correlation

Within each regime the features are ranked by their per-regime importance, most important first.
Two regimes are then compared by the **Spearman rank correlation** between their two rankings,
written rho. It ranges from -1 to 1: a value of 1 means the two regimes agree perfectly on the
order of the features, 0 means the orderings are unrelated, and -1 means they are reversed.

The comparison uses the rankings rather than the raw SHAP magnitudes on purpose. Different regimes
can have different overall SHAP scales (a more volatile regime tends to have larger SHAP values
throughout), so comparing magnitudes would confound a genuine change in which features matter with
a simple change in scale. Comparing the orderings avoids that.

The full result is a regime-by-regime matrix of rho, symmetric with ones on the diagonal.

## Stability bands (Akoglu, 2018)

Each correlation is classified into one of three bands, following the interpretation of
correlation strength in Akoglu (2018):

| Band | Condition |
| --- | --- |
| stable | rho greater than 0.6 |
| moderately stable | rho greater than 0.3 and at most 0.6 |
| unstable | rho at most 0.3 |

The two thresholds (0.3 and 0.6) are the defaults and can be changed. A stable band means the
model's drivers hold across the two regimes, and an unstable band means they have genuinely
reorganised.

Read a stable result carefully. A high stability score shows only that the ranking does not change
across regimes; it does not by itself show that the explanation is trustworthy. A model that relies
on very few features can be perfectly stable simply because there is little that could change.
Stability is therefore most informative when read alongside how concentrated the model's importance
is: a stable ranking spread across many features is reassuring, while a stable ranking resting on
one or two features mainly reflects that the model uses very little information.

## Short regimes and bootstrap confidence intervals

A regime with only a handful of rows produces a noisy ranking, so a single correlation for it can
be falsely precise. Regimes smaller than `small_sample_threshold` are flagged, and for any pair
that involves a flagged regime the package can attach a bootstrap confidence interval to the
correlation.

The bootstrap works as follows. It resamples the regime's rows with replacement, by default 1000
times, and for each resample it recomputes the SHAP values and the feature ranking. It then pairs
resample `i` of one regime with resample `i` of the other, computes the Spearman correlation for
each pair, and reports a percentile interval, by default the 95 percent interval. Below a small
floor of observations the package warns that the interval is unreliable.

A wide interval is the honest message that the ranking is uncertain because the data are limited.
The bootstrap quantifies that uncertainty; it does not remove it. The resampling treats the
regime's rows as independent, so for strongly autocorrelated data, such as consecutive time steps,
the interval can understate the true uncertainty and should be read as indicative rather than
exact. Because the bootstrap recomputes SHAP many times, the speed of the explainer matters, which
is one reason the package is currently tree-only.

## Supported models

`regime-shap` currently supports tree-based models (for example XGBoost, LightGBM, and
scikit-learn tree ensembles) through `shap.TreeExplainer`. This is a deliberate choice.
`TreeExplainer` computes exact SHAP values quickly, which is what makes the bootstrap feasible,
since it recomputes SHAP up to a thousand times per regime.

The stability methodology itself is model-agnostic. The importance, comparison, and stability
steps operate on SHAP values alone and never inspect the model, so only the SHAP computation is
tree-specific. Support for other model types through pluggable explainers is a possible future
extension, with the honest caveat that the bootstrap confidence intervals become expensive and
approximate for non-tree models.

## Limitations

Three limitations are worth keeping in mind:

- **Tree models only.** SHAP is computed with `shap.TreeExplainer`, so the package currently
  supports tree-based models only (see Supported models above). The stability method itself is
  model-agnostic; only the SHAP step is tree-specific.
- **Small regimes are uncertain.** A regime with few rows gives a noisy ranking. The bootstrap
  intervals report that uncertainty but do not remove it, and for autocorrelated data they are
  indicative rather than exact.
- **Stability is not the same as trustworthiness.** A high stability score can reflect a model
  that relies on very few features rather than one that is genuinely robust, so it should be read
  alongside how concentrated the importance is, not on its own.

## References

Akoglu, H. (2018) 'User's guide to correlation coefficients', *Turkish Journal of Emergency
Medicine*, 18(3), pp. 91-93.

Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions',
*Advances in Neural Information Processing Systems*, 30, pp. 4765-4774.

Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz, R., Himmelfarb,
J., Bansal, N. and Lee, S.-I. (2020) 'From local explanations to global understanding with
explainable AI for trees', *Nature Machine Intelligence*, 2(1), pp. 56-67.
