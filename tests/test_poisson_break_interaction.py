"""Tests for Poisson structural-break interactions."""

import pandas as pd

from src.ml_models import (
    add_break_development_interactions,
)


def test_no_break_has_zero_interactions() -> None:
    cells = pd.DataFrame(
        {
            "development_year": [
                1,
                2,
                3,
                4,
            ],
            "structural_break_indicator": [
                0,
                0,
                0,
                0,
            ],
        }
    )

    transformed, columns = (
        add_break_development_interactions(
            cells
        )
    )

    assert len(columns) == 3

    assert (
        transformed[
            columns
        ].to_numpy()
        == 0.0
    ).all()


def test_correct_development_interaction_is_active() -> None:
    cells = pd.DataFrame(
        {
            "development_year": [
                3,
            ],
            "structural_break_indicator": [
                1,
            ],
        }
    )

    transformed, columns = (
        add_break_development_interactions(
            cells
        )
    )

    assert (
        "structural_break_x_development_3"
        in columns
    )

    assert transformed.loc[
        0,
        "structural_break_x_development_3",
    ] == 1.0


def test_development_year_one_is_reference() -> None:
    cells = pd.DataFrame(
        {
            "development_year": [
                1,
                2,
                3,
            ],
            "structural_break_indicator": [
                1,
                1,
                1,
            ],
        }
    )

    _, columns = (
        add_break_development_interactions(
            cells
        )
    )

    assert (
        "structural_break_x_development_1"
        not in columns
    )


def test_only_matching_development_interaction_is_active() -> None:
    cells = pd.DataFrame(
        {
            "development_year": [
                3,
            ],
            "structural_break_indicator": [
                1,
            ],
        }
    )

    transformed, columns = (
        add_break_development_interactions(
            cells
        )
    )

    active_columns = [
        column
        for column in columns
        if transformed.loc[
            0,
            column,
        ] == 1.0
    ]

    assert active_columns == [
        "structural_break_x_development_3"
    ]


def test_interaction_features_preserve_required_columns() -> None:
    cells = pd.DataFrame(
        {
            "accident_year": [
                2020,
                2021,
            ],
            "development_year": [
                2,
                3,
            ],
            "calendar_year": [
                2021,
                2023,
            ],
            "accident_year_centered": [
                10,
                11,
            ],
            "relative_log_inflation": [
                -0.02,
                -0.01,
            ],
            "structural_break_indicator": [
                1,
                1,
            ],
        }
    )

    transformed, interaction_columns = (
        add_break_development_interactions(
            cells
        )
    )

    for column in [
        "accident_year_centered",
        "relative_log_inflation",
        "structural_break_indicator",
        "development_year",
    ]:
        assert column in transformed.columns

    assert len(
        interaction_columns
    ) > 0


def test_no_break_interaction_model_matches_original() -> None:
    import numpy as np
    import pandas as pd

    from src.ml_models import (
        fit_regularised_poisson_break_interaction_reserving_model,
        fit_regularised_poisson_reserving_model,
    )

    triangle = pd.DataFrame(
        [
            [10.0, 6.0, 3.0, 1.0],
            [12.0, 7.0, 4.0, np.nan],
            [11.0, 8.0, np.nan, np.nan],
            [13.0, np.nan, np.nan, np.nan],
        ],
        index=[
            2021,
            2022,
            2023,
            2024,
        ],
        columns=[
            1,
            2,
            3,
            4,
        ],
    )

    inflation_index = {
        2021: 1.00,
        2022: 1.02,
        2023: 1.04,
        2024: 1.06,
        2025: 1.08,
        2026: 1.10,
        2027: 1.12,
    }

    common_arguments = {
        "incremental_triangle": triangle,
        "inflation_index": inflation_index,
        "valuation_year": 2024,
        "basis": "gross",
        "alpha_grid": [
            0.01,
            0.1,
            1.0,
        ],
        "minimum_training_diagonals": 2,
        "minimum_validation_folds": 1,
        "amount_scale": 1.0,
        "structural_break_year": None,
    }

    original = (
        fit_regularised_poisson_reserving_model(
            **common_arguments
        )
    )

    interaction = (
        fit_regularised_poisson_break_interaction_reserving_model(
            **common_arguments
        )
    )

    original_reserve = float(
        original[
            "summary"
        ].iloc[0][
            "total_estimated_reserve"
        ]
    )

    interaction_reserve = float(
        interaction[
            "summary"
        ].iloc[0][
            "total_estimated_reserve"
        ]
    )

    assert np.isclose(
        original_reserve,
        interaction_reserve,
        rtol=1e-8,
        atol=1e-8,
    )

    original_alpha = float(
        original[
            "summary"
        ].iloc[0][
            "selected_alpha"
        ]
    )

    interaction_alpha = float(
        interaction[
            "summary"
        ].iloc[0][
            "selected_alpha"
        ]
    )

    assert np.isclose(
        original_alpha,
        interaction_alpha,
    )