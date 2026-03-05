# spey-hs3: HS3 Statistical Models in spey

* [pyhs3](https://pyhs3.readthedocs.io/en/latest/broadcasting.html)
* [Giordon's talk on pyhs3](https://indico.cern.ch/event/1566952/)
* Example published hist factory for hs3 [in hepdata](https://www.hepdata.net/record/resource/3888824?landing_page=true)
* convert [hs3 to pyhf](https://gitlab.cern.ch/cburgard/RooFitUtils/-/blob/master/scripts/json-roofit2pyhf.py?ref_type=heads)

## How to use Spey-HS3 plug-in

```python 
simple_hs3 = {
    "metadata": {"hs3_version": "0.2"},
    "distributions": [
        {
            "name": "channel",
            "type": "histfactory_dist",
            "axes": [{"name": "obs", "edges": [0.0, 1.0, 2.0]}],
            "samples": [
                {
                    "name": "bkg",
                    "data": {"contents": [40.0, 60.0], "errors": [0.0, 0.0]},
                    "modifiers": [
                        {
                            "name": "mu_bkg",
                            "type": "normfactor",
                            "parameter": "mu_bkg",
                        },
                        {
                            "name": "bkg_sys",
                            "type": "normsys",
                            "parameter": "alpha_bkg",
                            "data": {"lo": 0.90, "hi": 1.10},
                        },
                    ],
                }
            ],
        }
    ],
    "parameter_points": [
        {
            "name": "nominal",
            "parameters": [
                {"name": "mu", "value": 1.0},
                {"name": "mu_bkg", "value": 1.0, "const": True},
                {"name": "alpha_bkg", "value": 0.0},
            ],
        }
    ],
    "domains": [
        {
            "name": "model_domain",
            "type": "product_domain",
            "axes": [
                {"name": "mu", "min": -5.0, "max": 10.0},
                {"name": "mu_bkg", "min": 0.0, "max": 3.0},
                {"name": "alpha_bkg", "min": -5.0, "max": 5.0},
            ],
        }
    ],
    "data": [
        {
            "name": "observed",
            "type": "binned",
            "contents": [53, 57],
            "axes": [{"name": "obs", "edges": [0.0, 1.0, 2.0]}],
        }
    ],
    "likelihoods": [{"name": "L", "distributions": ["channel"], "data": ["observed"]}],
    "analyses": [
        {
            "name": "demo",
            "likelihood": "L",
            "domains": ["model_domain"],
            "parameters_of_interest": ["mu"],
            "init": "nominal",
        }
    ],
}

# Signal: 5 events in each bin (injected as a new sample)
simple_signal = {"channel": {"signal": [5.0, 5.0]}}

HS3 = spey.get_backend("hs3")
simple_model = HS3(
    hs3_dict=simple_hs3,
    signal_yields=simple_signal,
    mode="FAST_COMPILE",
)

cls_s_obs = simple_model.exclusion_confidence_level(
    expected=spey.ExpectationType.observed
)
cls_s_exp = simple_model.exclusion_confidence_level(expected=spey.ExpectationType.apriori)

print(f"Observed  CLs (mu=1) = {cls_s_obs[0]:.4f}")
print(f"Expected  CLs (mu=1) = {cls_s_exp[2]:.4f}")

# Observed CLs (mu=1) = 0.2361
# Expected CLs (mu=1) = 0.5299
```