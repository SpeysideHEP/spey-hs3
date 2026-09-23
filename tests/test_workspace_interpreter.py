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


# ---------------------------------------------------------------------------
# observed_data vs expected_background_yields (the original bug)
# ---------------------------------------------------------------------------


def test_observed_differs_from_background(uncorrelated_ws):
    """The reported bug: observed data and background MC must NOT coincide.

    The workspace pairs ``model_singlechannel`` with a real ``obsData`` ([51, 48])
    and an ``asimovData`` ([62, 63]); the background MC alone is [50, 52].
    """
    interp = WorkspaceInterpreter(uncorrelated_ws)
    obs = interp.observed_data
    bkg = interp.expected_background_yields
    assert obs == {"model_singlechannel": [51.0, 48.0]}
    assert bkg == {"model_singlechannel": pytest.approx([50.0, 52.0])}
    assert obs != bkg


def test_expected_signal_yields(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    assert interp.expected_signal_yields == {
        "model_singlechannel": pytest.approx([12.0, 11.0])
    }


def test_observed_data_per_likelihood(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    per_lh = interp.observed_data_per_likelihood
    assert per_lh["simPdf_asimovData"] == {"model_singlechannel": [62.0, 63.0]}
    assert per_lh["simPdf_obsData"] == {"model_singlechannel": [51.0, 48.0]}


def test_get_observed_data_by_analysis(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    asimov = interp.get_observed_data(analysis_name="simPdf_asimovData")
    obs = interp.get_observed_data(analysis_name="simPdf_obsData")
    assert asimov == {"model_singlechannel": [62.0, 63.0]}
    assert obs == {"model_singlechannel": [51.0, 48.0]}


def test_get_observed_data_unknown_likelihood_raises(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    with pytest.raises(ValueError, match="not found"):
        interp.get_observed_data(likelihood_name="does_not_exist")


def test_observed_data_inline_values():
    """A likelihood may carry inline numeric data instead of a data reference."""
    ws = {
        "metadata": {"hs3_version": "0.2"},
        "distributions": [
            {
                "name": "m",
                "type": "histfactory_dist",
                "axes": [{"name": "x", "min": 0, "max": 2, "nbins": 2}],
                "samples": [
                    {"name": "bkg", "data": {"contents": [1.0, 2.0], "errors": [0, 0]}}
                ],
            }
        ],
        "data": [],
        "likelihoods": [{"name": "lh", "distributions": ["m"], "data": [[7.0, 8.0]]}],
        "analyses": [{"name": "a", "likelihood": "lh", "parameters_of_interest": []}],
    }
    interp = WorkspaceInterpreter(ws)
    assert interp.observed_data == {"m": [7.0, 8.0]}


# ---------------------------------------------------------------------------
# Signal identification
# ---------------------------------------------------------------------------


def test_poi_parameters(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    assert interp.poi_parameters == ["mu"]


def test_signal_samples(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    assert interp.signal_samples == {"model_singlechannel": ["signal"]}


def test_signal_samples_none_when_no_signal(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    # model_SR has only a background sample (no POI modifier)
    assert interp.signal_samples == {}


def test_is_signal_sample(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    assert interp.is_signal_sample("model_singlechannel", "signal") is True
    assert interp.is_signal_sample("model_singlechannel", "background") is False
    assert interp.is_signal_sample("nope", "signal") is False


def test_signal_modifiers_roles(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    mods = interp.signal_modifiers["model_singlechannel"]["signal"]
    roles = {m["name"]: m["role"] for m in mods}
    assert roles == {"Lumi": "lumi", "mu": "poi"}


def test_signal_modifiers_classifies_nuisances(signal_nuisance_ws):
    interp = WorkspaceInterpreter(signal_nuisance_ws)
    mods = interp.signal_modifiers["SR"]["sig"]
    roles = {m["name"]: m["role"] for m in mods}
    assert roles == {
        "Lumi": "lumi",
        "mu": "poi",
        "alpha_sig": "nuisance",
        "alpha_shared": "nuisance",
    }


# ---------------------------------------------------------------------------
# strip_signal
# ---------------------------------------------------------------------------


def test_strip_signal_removes_sample(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    queued = interp.strip_signal()
    assert queued == {"model_singlechannel": ["signal"]}
    patched = interp.patch
    samples = [
        s["name"]
        for d in patched["distributions"]
        if d["name"] == "model_singlechannel"
        for s in d["samples"]
    ]
    assert samples == ["background"]


def test_strip_signal_prunes_orphan_poi(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    interp.strip_signal()
    patched = interp.patch
    # mu is orphaned once the only signal sample is gone -> pruned everywhere
    for dom in patched["domains"]:
        assert all(ax["name"] != "mu" for ax in dom["axes"])
    for pp in patched["parameter_points"]:
        assert all(p["name"] != "mu" for p in pp["parameters"])
    for a in patched["analyses"]:
        assert "mu" not in a["parameters_of_interest"]
    # background parameters are retained
    default_axes = next(
        d["axes"] for d in patched["domains"] if d["name"] == "default_domain"
    )
    names = {ax["name"] for ax in default_axes}
    assert {"Lumi", "uncorr_bkguncrt_0", "uncorr_bkguncrt_1"} <= names


def test_strip_signal_keep_parameters(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    interp.strip_signal(cleanup_parameters=False)
    patched = interp.patch
    poi_axis = next(
        d["axes"]
        for d in patched["domains"]
        if d["name"] == "simPdf_obsData_parameters_of_interest"
    )
    assert any(ax["name"] == "mu" for ax in poi_axis)


def test_strip_signal_no_signal_is_noop(minimal_ws):
    interp = WorkspaceInterpreter(minimal_ws)
    assert interp.strip_signal() == {}
    with pytest.raises(ValueError, match="Nothing to patch"):
        _ = interp.patch


# ---------------------------------------------------------------------------
# strip_signal_nuisances
# ---------------------------------------------------------------------------


def test_strip_signal_nuisances(signal_nuisance_ws):
    interp = WorkspaceInterpreter(signal_nuisance_ws)
    queued = interp.strip_signal_nuisances()
    assert queued == {"SR": {"sig": ["alpha_shared", "alpha_sig"]}}
    patched = interp.patch
    sig = next(
        s
        for d in patched["distributions"]
        for s in d["samples"]
        if s["name"] == "sig"
    )
    # only signal strength (mu) and luminosity remain
    assert [m["name"] for m in sig["modifiers"]] == ["Lumi", "mu"]


def test_strip_signal_nuisances_prunes_orphans_only(signal_nuisance_ws):
    interp = WorkspaceInterpreter(signal_nuisance_ws)
    interp.strip_signal_nuisances()
    patched = interp.patch
    axes = {ax["name"] for ax in patched["domains"][0]["axes"]}
    # alpha_sig was signal-only -> pruned; alpha_shared still on bkg -> kept
    assert "alpha_sig" not in axes
    assert {"mu", "Lumi", "alpha_shared"} <= axes
    # background sample is untouched
    bkg = next(
        s
        for d in patched["distributions"]
        for s in d["samples"]
        if s["name"] == "bkg"
    )
    assert sorted(m["name"] for m in bkg["modifiers"]) == ["Lumi", "alpha_shared"]


def test_strip_signal_nuisances_keep_lumi_false(signal_nuisance_ws):
    interp = WorkspaceInterpreter(signal_nuisance_ws)
    interp.strip_signal_nuisances(keep_lumi=False)
    patched = interp.patch
    sig = next(
        s
        for d in patched["distributions"]
        for s in d["samples"]
        if s["name"] == "sig"
    )
    assert [m["name"] for m in sig["modifiers"]] == ["mu"]
    # Lumi is still used by the background sample, so it must not be pruned
    assert any(ax["name"] == "Lumi" for ax in patched["domains"][0]["axes"])


def test_strip_signal_nuisances_noop_when_clean(uncorrelated_ws):
    # The uncorrelated workspace's signal only has Lumi + mu (no nuisances)
    interp = WorkspaceInterpreter(uncorrelated_ws)
    assert interp.strip_signal_nuisances() == {}


def test_reset_signal_clears_strips(uncorrelated_ws):
    interp = WorkspaceInterpreter(uncorrelated_ws)
    interp.strip_signal()
    interp.reset_signal()
    assert interp.remove_list == []
    with pytest.raises(ValueError, match="Nothing to patch"):
        _ = interp.patch
