# Frozen Research Design

## Working title

Inflation-Aware Reserving for Excess-of-Loss Reinsurance:
A Monte Carlo Comparison of Classical and Machine-Learning
Methods under Structural Change

## Central research question

How materially do claims inflation, structural change and
reinsurance indexation affect reserve estimates for
Excess-of-Loss reinsurance portfolios, and which reserving
methods remain most reliable under these conditions?

## Objectives

1. Simulate reproducible claim-level and payment-level data
   for short-tail and long-tail insurance portfolios.

2. Apply Excess-of-Loss reinsurance using alternative
   indexation and stability-clause structures.

3. Construct incomplete paid-claims triangles at a fixed
   valuation date.

4. Compare classical, inflation-adjusted and
   machine-learning reserving methods.

5. Measure model accuracy under stable inflation,
   emerging inflation, shock inflation and structural change.

6. Assess whether the relative performance of the methods
   differs between gross and ceded reserves.

## Main hypotheses

H1: Under stable inflation and no structural change,
classical Chain Ladder should provide broadly reasonable
reserve estimates.

H2: Under emerging or shock inflation, methods that
explicitly incorporate calendar-year inflation should
produce lower reserve bias than unadjusted Chain Ladder.

H3: Structural changes in reporting or settlement patterns
should reduce the accuracy of methods that rely strongly on
historical development stability.

H4: Reinsurance indexation clauses should materially affect
ceded reserves, particularly for long-tail claims exposed to
high inflation.

H5: A regularised machine-learning model may improve
predictive accuracy in some non-stationary scenarios, but
its performance may be less stable or interpretable than
classical actuarial methods.

## Core reserving methods

1. Chain Ladder
2. Inflation-Adjusted Chain Ladder
3. Cashflow Uplift
4. Regularised Poisson or LASSO reserving model

## Optional extension

A neural-network model will only be implemented if the core
experiment and main dissertation draft are complete.

## Core inflation scenarios

1. Stable inflation
2. Emerging inflation
3. Temporary inflation shock

## Core portfolio types

1. Short-tail
2. Long-tail

## Structural-change scenarios

1. No structural break
2. Accelerated settlement following the structural-break year

## Reinsurance clauses

1. No indexation
2. Full indexation
3. Triggered or capped indexation clause

## Core frequency assumption

Constant claim frequency is the main experiment.

Increasing and decreasing frequency will be used only as
sensitivity tests.

## Primary evaluation measures

1. Signed reserve error
2. Absolute reserve error
3. Percentage reserve error where the true reserve is material
4. Portfolio-normalised error
5. Mean absolute error
6. Root mean squared error
7. Mean reserve bias

## Validation approach

1. Internal reconciliation tests
2. Hand-worked treaty examples
3. Retrospective diagonal validation
4. Paired Monte Carlo comparisons
5. Sensitivity analysis
6. Reproducibility checks

## Scope freeze date

1 August 2026