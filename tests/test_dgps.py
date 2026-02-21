from explore_dgp.dgps import NonLinearDGP, ObLinearDGP, UnobservedConfounderDGP


def test_linear_dgp():
    dgp = ObLinearDGP(n_samples=100, seed=42)
    df = dgp.generate()
    assert len(df) == 100
    assert "sick_leave" in df.columns
    assert "group" in df.columns
    assert set(df["group"].unique()) == {0, 1}


def test_nonlinear_dgp():
    dgp = NonLinearDGP(n_samples=100, seed=42)
    df = dgp.generate()
    assert len(df) == 100
    assert "sick_leave" in df.columns


def test_confounder_dgp():
    dgp = UnobservedConfounderDGP(n_samples=100, seed=42)
    df = dgp.generate()
    assert len(df) == 100
    assert "sick_leave" in df.columns
    # Z should not be in the output
    assert "z" not in df.columns
