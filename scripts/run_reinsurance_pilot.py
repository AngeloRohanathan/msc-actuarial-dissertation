"""Run the baseline fixed-term XoL reinsurance pilot."""

from __future__ import annotations

import pandas as pd

from config import (
    MASTER_RANDOM_SEED,
    PILOT_DATA_DIR,
    PILOT_XOL_ATTACHMENT,
    PILOT_XOL_LIMIT,
    ensure_directories,
    validate_config,
)

from src.reinsurance import (
    apply_xol_to_payments,
)

from src.simulation import simulate_portfolio


OUTPUT_DIR = (
    PILOT_DATA_DIR / "reinsurance"
)

PAYMENT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "reinsured_payments_pilot.parquet"
)

CLAIM_OUTPUT_PATH = (
    OUTPUT_DIR
    / "reinsured_claim_summary_pilot.parquet"
)


def main() -> None:
    """Generate and save the baseline XoL pilot."""

    ensure_directories()
    validate_config()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, payments = simulate_portfolio(
        simulation_id=1,
        frequency_scenario="constant",
        tail_type="long",
        inflation_scenario="stable",
        apply_structural_break=False,
        seed=MASTER_RANDOM_SEED,
    )

    reinsured_payments, claim_summary = (
        apply_xol_to_payments(
            payments=payments,
            attachment=PILOT_XOL_ATTACHMENT,
            limit=PILOT_XOL_LIMIT,
        )
    )

    reinsured_payments.to_parquet(
        PAYMENT_OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    claim_summary.to_parquet(
        CLAIM_OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    total_gross = float(
        reinsured_payments[
            "nominal_gross_payment"
        ].sum()
    )

    total_ceded = float(
        reinsured_payments[
            "nominal_ceded_payment"
        ].sum()
    )

    total_retained = float(
        reinsured_payments[
            "nominal_retained_payment"
        ].sum()
    )

    attachment_breached = int(
        claim_summary[
            "attachment_breached"
        ].sum()
    )

    limit_exhausted = int(
        claim_summary[
            "limit_exhausted"
        ].sum()
    )

    ceded_percentage = (
        100.0 * total_ceded / total_gross
        if total_gross > 0
        else 0.0
    )

    print(
        "Basic XoL reinsurance pilot completed."
    )

    print(
        "Treaty: "
        f"£{PILOT_XOL_LIMIT:,.0f} xs "
        f"£{PILOT_XOL_ATTACHMENT:,.0f}"
    )

    print(
        f"Claims analysed: "
        f"{len(claim_summary):,}"
    )

    print(
        f"Total gross payments: "
        f"£{total_gross:,.2f}"
    )

    print(
        f"Total ceded payments: "
        f"£{total_ceded:,.2f}"
    )

    print(
        f"Total retained payments: "
        f"£{total_retained:,.2f}"
    )

    print(
        f"Ceded percentage: "
        f"{ceded_percentage:.2f}%"
    )

    print(
        "Claims breaching attachment: "
        f"{attachment_breached:,}"
    )

    print(
        "Claims exhausting treaty limit: "
        f"{limit_exhausted:,}"
    )

    print(
        "Payment output saved to: "
        f"{PAYMENT_OUTPUT_PATH}"
    )

    print(
        "Claim summary saved to: "
        f"{CLAIM_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()