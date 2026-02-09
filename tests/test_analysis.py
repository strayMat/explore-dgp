from explore_dgp.analysis import OaxacaAnalysis
from explore_dgp.dgps import LinearDGP


def test_oaxaca_analysis():
    dgp = LinearDGP(n_samples=500, seed=42)
    df = dgp.generate()

    covariates = ["age", "sex", "revenue", "unemployment_rate"]
    analysis = OaxacaAnalysis(df, "sick_leave", covariates, "group")
    analysis.run()

    summary = analysis.get_summary_table()
    assert "Endowment Effect" in summary.index
    assert "Coefficient Effect" in summary.index
    assert "Interaction Effect" in summary.index
    assert "Total Difference" in summary.index

    details = analysis.get_coefficient_details()
    assert "Group 0" in details.columns
    assert "Group 1" in details.columns
    # Check that it has more rows than covariates (due to the added constant)
    assert len(details.index) > len(covariates)
