"""Shared fixtures for spey-hs3 tests."""

import pytest


# ---------------------------------------------------------------------------
# Minimal 2-bin workspace (built from scratch, no external files)
# ---------------------------------------------------------------------------

MINIMAL_WS = {
    "metadata": {"hs3_version": "0.2", "description": "minimal 2-bin test"},
    "distributions": [
        {
            "name": "model_SR",
            "type": "histfactory_dist",
            "axes": [{"name": "obs_x", "min": 0.0, "max": 2.0, "nbins": 2}],
            "samples": [
                {
                    "name": "background",
                    # No normfactor on the background — mu only enters via
                    # injected signal so that expected_data at mu=0 is non-zero.
                    "data": {"contents": [50.0, 30.0], "errors": [5.0, 3.0]},
                    "modifiers": [],
                }
            ],
        }
    ],
    "domains": [
        {
            "name": "default_domain",
            "type": "product_domain",
            "axes": [{"name": "mu", "min": -10.0, "max": 10.0}],
        }
    ],
    "parameter_points": [
        {
            "name": "default_values",
            "parameters": [{"name": "mu", "value": 1.0, "const": False}],
        }
    ],
    "data": [
        {
            "name": "obsdata_SR",
            "type": "binned",
            "contents": [52.0, 31.0],
            "axes": [{"name": "obs_x", "min": 0.0, "max": 2.0, "nbins": 2}],
        }
    ],
    "likelihoods": [
        {"name": "lh_SR", "distributions": ["model_SR"], "data": ["obsdata_SR"]}
    ],
    "analyses": [
        {
            "name": "analysis_SR",
            "likelihood": "lh_SR",
            "parameters_of_interest": ["mu"],
            "domains": ["default_domain"],
            "init": "default_values",
        }
    ],
}


# Uncorrelated-background workspace from pyhs3 test suite
UNCORRELATED_WS_PATH = "tests/simplemodel_uncorrelated-background_hs3.json"


@pytest.fixture
def minimal_ws():
    """Return a deep copy of the minimal 2-bin workspace dict."""
    import copy

    return copy.deepcopy(MINIMAL_WS)


@pytest.fixture
def uncorrelated_ws():
    """Return the pyhs3 uncorrelated-background test workspace dict."""
    import json

    with open(UNCORRELATED_WS_PATH) as f:
        return json.load(f)
