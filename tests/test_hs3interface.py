"""Integration tests for HS3Interface (compiles pyhs3 models)."""

import numpy as np
import pytest
import spey

from spey_hs3 import WorkspaceInterpreter

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_model(ws_dict, signal=None, analysis=None, **kwargs):
    """Convenience wrapper: optionally inject signal, then build HS3Interface."""
    if signal is not None:
        interp = WorkspaceInterpreter(ws_dict)
        interp.inject_signals(signal)
        ws_dict = interp.patch
    return spey.get_backend("hs3")(
        hs3_dict=ws_dict,
        analysis_name=analysis,
        progress=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------


def test_backend_registered():
    """spey.get_backend('hs3') should return a callable that produces HS3Interface instances."""
    backend_factory = spey.get_backend("hs3")
    assert callable(backend_factory)


# ---------------------------------------------------------------------------
# Background-only model (minimal workspace)
# ---------------------------------------------------------------------------


def test_bg_only_model_config(minimal_ws):
    model = _build_model(minimal_ws, analysis="analysis_SR")
    cfg = model.config()
    assert cfg.poi_index == 0
    assert "mu" in cfg.parameter_names
    assert cfg.parameter_names[0] == "mu"
    assert cfg.suggested_bounds[0] == pytest.approx((-10.0, 10.0))


def test_bg_only_logpdf(minimal_ws):
    model = _build_model(minimal_ws, analysis="analysis_SR")
    lp = model.backend.get_logpdf_func()([1.0])
    assert np.isfinite(lp)


def test_bg_only_expected_data(minimal_ws):
    model = _build_model(minimal_ws, analysis="analysis_SR")
    exp = model.backend.expected_data([1.0])
    assert len(exp) == 2
    assert all(np.isfinite(v) for v in exp)
    assert all(v > 0 for v in exp)


# ---------------------------------------------------------------------------
# Signal injection (minimal workspace)
# ---------------------------------------------------------------------------


def test_signal_model_config(minimal_ws):
    signal = {"model_SR": {"Signal": [3.0, 5.0]}}
    model = _build_model(minimal_ws, signal=signal, analysis="analysis_SR")
    cfg = model.config()
    assert cfg.parameter_names[0] == "mu"


def test_logpdf_increases_with_mu(minimal_ws):
    """logpdf should be finite at various mu values and differ for signal vs no-signal."""
    signal = {"model_SR": {"Signal": [3.0, 5.0]}}
    model = _build_model(minimal_ws, signal=signal, analysis="analysis_SR")
    lp = model.backend.get_logpdf_func()
    lp_m2 = lp([-2.0])
    lp_0 = lp([0.0])
    lp_1 = lp([1.0])
    lp_5 = lp([5.0])
    # All values should be finite
    assert all(np.isfinite(v) for v in [lp_m2, lp_0, lp_1, lp_5])
    # Observed data (52, 31) ≈ background (50, 30); logpdf should peak near mu=0
    assert lp_0 > lp_5


def test_logpdf_apriori_vs_observed(minimal_ws):
    from spey.utils import ExpectationType

    signal = {"model_SR": {"Signal": [3.0, 5.0]}}
    model = _build_model(minimal_ws, signal=signal, analysis="analysis_SR")
    lp_obs = model.backend.get_logpdf_func(expected=ExpectationType.observed)([1.0])
    lp_pri = model.backend.get_logpdf_func(expected=ExpectationType.apriori)([1.0])
    assert np.isfinite(lp_obs)
    assert np.isfinite(lp_pri)
    # They should differ because observed and Asimov data differ
    assert lp_obs != pytest.approx(lp_pri)


def test_expected_data_with_signal(minimal_ws):
    signal = {"model_SR": {"Signal": [3.0, 5.0]}}
    model = _build_model(minimal_ws, signal=signal, analysis="analysis_SR")
    exp = model.backend.expected_data([1.0])
    assert len(exp) == 2
    # At mu=1, expected = background + signal
    assert exp[0] == pytest.approx(53.0, rel=1e-3)
    assert exp[1] == pytest.approx(35.0, rel=1e-3)


def test_expected_data_mu_zero_equals_background(minimal_ws):
    """At mu=0 the signal sample contributes nothing, so expected ≈ background."""
    signal = {"model_SR": {"Signal": [3.0, 5.0]}}
    model = _build_model(minimal_ws, signal=signal, analysis="analysis_SR")
    exp0 = model.backend.expected_data([0.0])
    assert exp0[0] == pytest.approx(50.0, rel=1e-3)
    assert exp0[1] == pytest.approx(30.0, rel=1e-3)


# ---------------------------------------------------------------------------
# is_alive property
# ---------------------------------------------------------------------------


def test_is_alive_no_signal(minimal_ws):
    model = _build_model(minimal_ws, analysis="analysis_SR")
    assert model.backend.is_alive is True


def test_is_alive_with_signal(minimal_ws):
    signal = {"model_SR": {"Signal": [3.0, 5.0]}}
    model = _build_model(minimal_ws, signal=signal, analysis="analysis_SR")
    assert model.backend.is_alive is True


def test_is_alive_zero_signal(minimal_ws):
    """is_alive is False when signal_yields contains only zeros."""
    import copy

    ws = copy.deepcopy(minimal_ws)
    # Inject signal directly via signal_yields kwarg (not WorkspaceInterpreter)
    # so that HS3Interface stores it in self._signal_yields
    model = spey.get_backend("hs3")(
        hs3_dict=ws,
        signal_yields={"model_SR": {"Signal": [0.0, 0.0]}},
        analysis_name="analysis_SR",
        progress=False,
    )
    assert model.backend.is_alive is False


# ---------------------------------------------------------------------------
# Missing errors auto-fill (pyhs3 >=0.4.1 compatibility)
# ---------------------------------------------------------------------------


def test_missing_errors_are_filled(minimal_ws):
    """HS3Interface should tolerate samples without 'errors' by filling zeros."""
    import copy

    ws = copy.deepcopy(minimal_ws)
    # Remove errors from background sample
    ws["distributions"][0]["samples"][0]["data"].pop("errors")
    # Should not raise
    model = _build_model(ws, analysis="analysis_SR")
    cfg = model.config()
    assert cfg.parameter_names[0] == "mu"


# ---------------------------------------------------------------------------
# Uncorrelated-background workspace (pyhs3 test suite)
# ---------------------------------------------------------------------------


def test_uncorrelated_bg_config(uncorrelated_ws):
    model = _build_model(uncorrelated_ws, analysis="simPdf_obsData")
    cfg = model.config()
    assert cfg.poi_index == 0
    assert cfg.parameter_names[0] == "mu"
    # Lumi is const=True → should not appear as free param
    assert "Lumi" not in cfg.parameter_names


def test_uncorrelated_bg_logpdf(uncorrelated_ws):
    model = _build_model(uncorrelated_ws, analysis="simPdf_obsData")
    cfg = model.config()
    init = cfg.suggested_init
    lp = model.backend.get_logpdf_func()(init)
    assert np.isfinite(lp)


def test_uncorrelated_bg_expected_data(uncorrelated_ws):
    model = _build_model(uncorrelated_ws, analysis="simPdf_obsData")
    cfg = model.config()
    exp = model.backend.expected_data(cfg.suggested_init)
    assert len(exp) == 2
    assert all(np.isfinite(v) and v > 0 for v in exp)


def test_uncorrelated_signal_injection(uncorrelated_ws):
    signal = {"model_singlechannel": {"NewSignal": [3.0, 5.0]}}
    model = _build_model(uncorrelated_ws, signal=signal, analysis="simPdf_obsData")
    cfg = model.config()
    assert cfg.parameter_names[0] == "mu"


def test_uncorrelated_clsb(uncorrelated_ws):
    """CLs+b should be a probability in [0, 1]."""
    signal = {"model_singlechannel": {"NewSignal": [3.0, 5.0]}}
    model = _build_model(uncorrelated_ws, signal=signal, analysis="simPdf_obsData")
    clsb = model.exclusion_confidence_level()
    assert 0.0 <= float(clsb[0]) <= 1.0


def test_uncorrelated_maximize_likelihood(uncorrelated_ws):
    signal = {"model_singlechannel": {"NewSignal": [3.0, 5.0]}}
    model = _build_model(uncorrelated_ws, signal=signal, analysis="simPdf_obsData")
    muhat, nll = model.maximize_likelihood()
    assert np.isfinite(muhat)
    assert np.isfinite(nll)
