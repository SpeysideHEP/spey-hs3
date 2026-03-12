"""Tests for WorkspaceInterpreter (no pyhs3 compilation required)."""

import copy
import pytest
from spey_hs3 import WorkspaceInterpreter


# ---------------------------------------------------------------------------
# Inspection properties
# ---------------------------------------------------------------------------


def test_distributions(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    assert interp.distributions == ["model_SR"]


def test_analyses(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    assert interp.analyses == ["analysis_SR"]


def test_poi_names(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    assert interp.poi_names == {"analysis_SR": ["mu"]}


def test_likelihoods(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    assert interp.likelihoods == {"analysis_SR": "lh_SR"}


def test_bin_map(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    assert interp.bin_map == {"model_SR": 2}


def test_samples(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    assert interp.samples == {"model_SR": ["background"]}


def test_modifier_types(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    mt = interp.modifier_types
    # background has no modifiers in the minimal workspace
    assert mt["model_SR"]["background"] == []


def test_modifier_types_with_normfactor(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    mt = interp.modifier_types
    # The uncorrelated workspace has normfactor and shapesys on the background sample
    assert "normfactor" in mt["model_singlechannel"]["background"]


def test_observed_data(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    obs = interp.observed_data
    assert "model_SR" in obs
    assert obs["model_SR"] == [52.0, 31.0]


def test_expected_background_yields(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    bg = interp.expected_background_yields
    assert bg["model_SR"] == pytest.approx([50.0, 30.0])


def test_parameters(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    params = interp.parameters
    assert "mu" in params
    assert params["mu"]["value"] == pytest.approx(1.0)
    assert params["mu"]["const"] is False


def test_get_sample_yields(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    contents, errors = interp.get_sample_yields("model_SR", "background")
    assert contents == pytest.approx([50.0, 30.0])
    assert errors == pytest.approx([5.0, 3.0])


def test_get_distributions_by_analysis(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    dists = interp.get_distributions(analysis_name="analysis_SR")
    assert dists == ["model_SR"]


def test_repr(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    assert "analyses=1" in repr(interp)


# ---------------------------------------------------------------------------
# Signal injection
# ---------------------------------------------------------------------------


def test_inject_signal_basic(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    interp.inject_signal("model_SR", "Signal", [3.0, 5.0])
    assert interp.signal_per_distribution == {"model_SR": {"Signal": [3.0, 5.0]}}


def test_inject_signal_with_errors(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    interp.inject_signal("model_SR", "Signal", [3.0, 5.0], errors=[0.3, 0.5])
    patched = interp.patch
    injected = next(
        s
        for d in patched["distributions"]
        if d["name"] == "model_SR"
        for s in d["samples"]
        if s["name"] == "Signal"
    )
    assert injected["data"]["errors"] == pytest.approx([0.3, 0.5])


def test_inject_signals_bulk(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    interp.inject_signals(
        {
            "model_SR": {
                "Sig1": [1.0, 2.0],
                "Sig2": {"contents": [3.0, 4.0], "errors": [0.1, 0.2]},
            }
        }
    )
    assert "Sig1" in interp.signal_per_distribution["model_SR"]
    assert "Sig2" in interp.signal_per_distribution["model_SR"]


def test_inject_signal_bin_mismatch_raises(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    with pytest.raises(ValueError, match="Bin count mismatch"):
        interp.inject_signal("model_SR", "Signal", [1.0])


def test_inject_signal_duplicate_raises(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    interp.inject_signal("model_SR", "Signal", [3.0, 5.0])
    with pytest.raises(ValueError, match="already exists"):
        interp.inject_signal("model_SR", "Signal", [1.0, 2.0])


def test_inject_signal_unknown_dist_raises(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    with pytest.raises(ValueError, match="not a histfactory_dist"):
        interp.inject_signal("nonexistent", "Signal", [1.0, 2.0])


def test_reset_signal(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    interp.inject_signal("model_SR", "Signal", [3.0, 5.0])
    interp.reset_signal()
    assert interp.signal_per_distribution == {}


# ---------------------------------------------------------------------------
# Patch output
# ---------------------------------------------------------------------------


def test_patch_adds_normfactor(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    interp.inject_signal("model_SR", "Signal", [3.0, 5.0], poi_name="mu")
    patched = interp.patch
    samples = next(
        d["samples"]
        for d in patched["distributions"]
        if d["name"] == "model_SR"
    )
    sig = next(s for s in samples if s["name"] == "Signal")
    assert any(m["type"] == "normfactor" for m in sig["modifiers"])
    assert any(m["name"] == "mu" for m in sig["modifiers"])


def test_patch_registers_poi_in_analysis(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    interp.inject_signal("model_SR", "Signal", [3.0, 5.0], poi_name="mu")
    patched = interp.patch
    analysis = next(a for a in patched["analyses"] if a["name"] == "analysis_SR")
    assert "mu" in analysis["parameters_of_interest"]


def test_patch_is_deep_copy(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    interp.inject_signal("model_SR", "Signal", [3.0, 5.0])
    p1 = interp.patch
    p2 = interp.patch
    assert p1 is not p2


def test_patch_without_injection_raises(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    with pytest.raises(ValueError, match="Nothing to patch"):
        _ = interp.patch


# ---------------------------------------------------------------------------
# Remove distribution
# ---------------------------------------------------------------------------


def test_remove_distribution(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    interp.remove_distribution("model_SR")
    interp.inject_signal("model_SR", "Signal", [3.0, 5.0])  # also inject to allow patch
    patched = interp.patch
    dist_names = [d["name"] for d in patched["distributions"]]
    assert "model_SR" not in dist_names


def test_remove_unknown_distribution_warns(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    # Should not raise, just log a warning
    interp.remove_distribution("does_not_exist")
