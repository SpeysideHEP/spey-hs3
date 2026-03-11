"""spey plug-in for pyhs3 (HS3-format statistical models)."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pyhs3
from pyhs3.domains import ProductDomain
from spey.base import ModelConfig
from spey.base.backend_base import BackendBase
from spey.utils import ExpectationType

from ._version import __version__
from .helper_functions import WorkspaceInterpreter

__all__ = ["HS3Interface", "WorkspaceInterpreter"]

log = logging.getLogger("Spey")


def __dir__():
    return __all__


def _merge_domains(workspace: pyhs3.Workspace, domain_names: List[str]) -> ProductDomain:
    """
    Merge multiple ``ProductDomain`` objects (referenced by name in the
    workspace) into a single ``ProductDomain`` whose axes are the union of all
    referenced axes.

    Parameters
    ----------
    workspace:
        Loaded pyhs3 Workspace.
    domain_names:
        List of domain names to merge (order matters only for deduplication).

    Returns
    -------
    ProductDomain
        A new synthetic domain containing all axes from every referenced domain.
    """
    merged_axes: Dict[str, object] = {}
    for dname in domain_names:
        domain = workspace.domains[dname]
        if not isinstance(domain, ProductDomain):
            log.warning("Domain '%s' is not a ProductDomain; skipping.", dname)
            continue
        for axis in domain.axes:
            # Later domains override earlier ones for the same axis name
            merged_axes[axis.name] = axis  # reuse existing axis objects directly

    return ProductDomain(
        name="_spey_merged_domain",
        type="product_domain",
        axes=list(merged_axes.values()),
    )


# ---------------------------------------------------------------------------
# HS3Interface — the spey BackendBase implementation
# ---------------------------------------------------------------------------


class HS3Interface(BackendBase):
    """
    Spey plug-in for statistical models described in HS3 (HEP Statistics
    Serialisation Standard) format via the ``pyhs3`` package.

    Parameters
    ----------
    hs3_dict:
        HS3 JSON workspace as a Python dictionary.  This is the background-only
        model; signal yields are provided separately through *signal_yields*.
    signal_yields:
        Per-distribution signal sample specifications.  The expected structure
        is a nested dictionary::

            {
                "dist_name": {
                    "sample_name": [y0, y1, ...],        # bin yields
                    # OR
                    "sample_name": {
                        "contents": [y0, y1, ...],       # bin yields
                        "errors": [e0, e1, ...],         # optional MC errors
                    },
                },
            }

        Each injected sample receives a ``normfactor`` modifier tied to the
        parameter of interest so that the signal strength scales the whole
        sample.  Pass ``None`` (or omit) for a background-only model.
    analysis_name:
        Name of the HS3 analysis to use.  Defaults to the first analysis
        in the workspace.
    poi_name:
        Name of the parameter of interest.  If ``None``, the first entry in
        the chosen analysis's ``parameters_of_interest`` list is used.
    progress:
        Show the pyhs3 compilation progress bar (default: ``False``).
    mode:
        PyTensor compilation mode forwarded to ``pyhs3.Workspace.model()``.
        Choose ``"FAST_COMPILE"`` for faster start-up (less optimisation) or
        ``"FAST_RUN"`` for faster repeated evaluations.

    Examples
    --------
    >>> import json, spey
    >>> with open("workspace.json") as f:
    ...     hs3 = json.load(f)
    >>> signal = {"model_SR": {"Signal": [3.0, 5.0, 2.0]}}
    >>> stat_model = spey.get_backend("hs3")(
    ...     hs3_dict=hs3,
    ...     signal_yields=signal,
    ...     analysis_name="myAnalysis",
    ... )
    >>> stat_model.exclusion_confidence_level()
    """

    name: str = "hs3"
    version: str = __version__
    author: str = "SpeysideHEP"
    spey_requires: str = ">=0.2.1"

    __slots__ = [
        "_workspace",
        "_analysis",
        "_likelihood",
        "_model",
        "_poi_name",
        "_signal_yields",
        # parameter book-keeping
        "_free_params",  # ordered list: [poi_name, nuis1, nuis2, ...]
        "_const_params",  # dict: name -> value
        "_observed_data",  # dict: dist_name -> np.ndarray (observed counts)
        "_dist_par_map",  # dict: dist_name -> list[str]  (pars in compiled order)
        "_domain",  # merged ProductDomain
        "_init_values",  # dict: param_name -> float
        "_param_bounds",  # dict: param_name -> (min, max)
    ]

    def __init__(
        self,
        hs3_dict: Dict,
        signal_yields: Optional[Dict[str, Dict[str, Union[List[float], Dict]]]] = None,
        analysis_name: Optional[str] = None,
        poi_name: Optional[str] = None,
        progress: bool = True,
        mode: str = "FAST_COMPILE",
    ):
        # ------------------------------------------------------------------ #
        # 1. Determine POI name (before injection so we know what to inject)  #
        # ------------------------------------------------------------------ #
        interpreter = WorkspaceInterpreter(hs3_dict)

        analyses = interpreter.analyses
        if analysis_name is None:
            analysis_name = analyses[0]
            log.debug("Setting analysis name as '%s'", analysis_name)
        else:
            assert analysis_name in analyses, (
                f"Analysis {analysis_name} does not exist. Available analyses are"
                + ", ".join(analyses)
            )

        if poi_name is None:
            poi_names = interpreter.poi_names.get(analysis_name, [])
            if len(poi_names) == 0:
                raise ValueError(
                    "No parameters_of_interest found in the selected analysis. "
                    "Please specify `poi_name` explicitly."
                )
            poi_name = poi_names[0]
            log.warning(
                "Parameter of interest is not defined, setting it as '%s'.", poi_name
            )
        self._poi_name: str = poi_name

        # ------------------------------------------------------------------ #
        # 2. Inject signal into the workspace dict                            #
        # ------------------------------------------------------------------ #
        self._signal_yields = signal_yields or {}
        if signal_yields:
            interpreter.inject_signals(signal_yields)
            working_dict = interpreter.patch
        else:
            working_dict = interpreter._workspace

        # ------------------------------------------------------------------ #
        # 3. Build pyhs3 Workspace                                            #
        # ------------------------------------------------------------------ #
        self._workspace: pyhs3.Workspace = pyhs3.Workspace(**working_dict)

        # ------------------------------------------------------------------ #
        # 4. Resolve analysis / likelihood                                    #
        # ------------------------------------------------------------------ #
        self._analysis = self._workspace.analyses[analysis_name]
        self._likelihood = self._workspace.likelihoods[
            interpreter.likelihoods[analysis_name]
        ]

        # ------------------------------------------------------------------ #
        # 5. Build merged domain and select parameter set                     #
        # ------------------------------------------------------------------ #
        domain_names = [d.name for d in self._analysis.domains or []]
        if domain_names:
            self._domain = _merge_domains(self._workspace, domain_names)
        else:
            self._domain = ProductDomain(
                name="_spey_empty_domain",
                type="product_domain",
                axes=[],
            )

        init_ps_name = self._analysis.init
        if init_ps_name and init_ps_name in self._workspace.parameter_points:
            init_ps = self._workspace.parameter_points[init_ps_name]
        elif len(self._workspace.parameter_points) > 0:
            init_ps = self._workspace.parameter_points[0]
        else:
            init_ps = None

        # ------------------------------------------------------------------ #
        # 6. Compile pyhs3 Model                                              #
        # ------------------------------------------------------------------ #
        self._model = self._workspace.model(
            domain=self._domain,
            parameter_set=init_ps if init_ps is not None else 0,
            progress=progress,
            mode=mode,
        )

        # ------------------------------------------------------------------ #
        # 7. Book-keep parameters                                             #
        # ------------------------------------------------------------------ #
        self._setup_parameters(init_ps)

    # ------------------------------------------------------------------ #
    # Parameter setup                                                      #
    # ------------------------------------------------------------------ #

    def _setup_parameters(self, init_ps) -> None:
        """
        Inspect the compiled model to separate POI, nuisance, const, and
        observed-data parameters for every distribution in the likelihood.
        """
        lh = self._likelihood

        # Collect const param names / values from the init parameter set
        const_params: Dict[str, float] = {}
        init_values: Dict[str, float] = {}
        if init_ps is not None:
            for p in init_ps.parameters:
                init_values[p.name] = p.value
                if p.const:
                    const_params[p.name] = p.value

        # Collect domain bounds
        # DomainCoordinateAxis uses v_min/v_max as Python field names (alias='min'/'max')
        param_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for axis in self._domain.axes:
            lo = getattr(axis, "v_min", None) if hasattr(axis, "v_min") else getattr(axis, "min", None)
            hi = getattr(axis, "v_max", None) if hasattr(axis, "v_max") else getattr(axis, "max", None)
            param_bounds[axis.name] = (lo, hi)

        # Walk every distribution in the likelihood and collect all parameter names
        # Separate observed-data params from free/const params
        observed_data: Dict[str, np.ndarray] = {}
        dist_par_map: Dict[str, List[str]] = {}
        free_params_set: List[str] = []  # ordered, deduped

        for dist_obj, data_obj in zip(lh.distributions, lh.data):
            dist_name = dist_obj.name
            pars = self._model.pars(dist_name)
            dist_par_map[dist_name] = pars

            # Get observed data directly from the paired BinnedData object
            try:
                obs = np.array(data_obj.contents, dtype=np.float64)
                observed_data[dist_name] = obs
            except AttributeError:
                pass

            # Collect free parameters (not const, not observed)
            for par_name in pars:
                if par_name.endswith("_observed"):
                    continue  # data parameter, handled separately
                if par_name in const_params:
                    continue  # fixed
                if par_name not in free_params_set:
                    free_params_set.append(par_name)

        # Ensure POI is first in free_params
        if self._poi_name in free_params_set:
            free_params_set.remove(self._poi_name)
        free_params_set.insert(0, self._poi_name)

        self._free_params = free_params_set
        self._const_params = const_params
        self._observed_data = observed_data
        self._dist_par_map = dist_par_map
        self._init_values = init_values
        self._param_bounds = param_bounds

    # ------------------------------------------------------------------ #
    # BackendBase required properties / methods                           #
    # ------------------------------------------------------------------ #

    @property
    def is_alive(self) -> bool:
        """
        Returns ``True`` if at least one injected signal yield is non-zero,
        or if no signal was injected (background-only model).
        """
        if not self._signal_yields:
            return True
        for _, samples in self._signal_yields.items():
            for _, spec in samples.items():
                counts = spec if isinstance(spec, list) else spec.get("contents", [])
                if any(c != 0.0 for c in counts):
                    return True
        return False

    def config(
        self,
        allow_negative_signal: bool = True,
        poi_upper_bound: float = 10.0,
    ) -> ModelConfig:
        r"""
        Model configuration container.

        Parameters
        ----------
        allow_negative_signal:
            If ``True``, :math:`\hat\mu` is allowed to be negative.
        poi_upper_bound:
            Upper bound for the parameter of interest, :math:`\mu`.

        Returns
        -------
        ModelConfig
            Spey model configuration.
        """
        poi_min = -poi_upper_bound if allow_negative_signal else 0.0

        # Suggested init values: POI first, then nuisance params
        suggested_init: List[float] = []
        suggested_bounds: List[Tuple[float, float]] = []
        parameter_names: List[str] = []

        for par_name in self._free_params:
            init_val = self._init_values.get(par_name, 1.0)
            lo, hi = self._param_bounds.get(par_name, (None, None))

            if par_name == self._poi_name:
                lo_eff = poi_min
                hi_eff = poi_upper_bound
                init_val = max(poi_min, min(init_val, poi_upper_bound))
            else:
                lo_eff = lo if lo is not None else -np.inf
                hi_eff = hi if hi is not None else np.inf

            suggested_init.append(init_val)
            suggested_bounds.append((lo_eff, hi_eff))
            parameter_names.append(par_name)

        return ModelConfig(
            poi_index=0,
            minimum_poi=poi_min,
            suggested_init=suggested_init,
            suggested_bounds=suggested_bounds,
            parameter_names=parameter_names,
        )

    def get_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Callable[[np.ndarray], float]:
        r"""
        Return a function that computes :math:`\log\mathcal{L}(\mu, \theta)`.

        Parameters
        ----------
        expected:
            * ``observed`` / ``aposteriori``: use the observed data from the
              workspace.
            * ``apriori``: use the background-only Asimov data (expected
              counts at :math:`\mu = 0` and nominal nuisance parameters).
        data:
            External data override.  When provided, this flat array replaces
            the workspace's observed data for *all* distributions (in
            likelihood order).

        Returns
        -------
        Callable
            Function ``f(pars: np.ndarray) -> float`` where ``pars`` is the
            concatenated parameter array ``[mu, theta1, theta2, ...]``.
        """
        lh = self._likelihood
        model = self._model
        free_params = self._free_params
        const_params = self._const_params
        dist_par_map = self._dist_par_map

        # Resolve observed data for each distribution
        if data is not None:
            obs_data = _split_data_by_dist(
                np.asarray(data, dtype=np.float64),
                [d.name for d in lh.distributions],
                dist_par_map,
                self._observed_data,
            )
        elif expected == ExpectationType.apriori:
            obs_data = self._compute_asimov_data()
        else:
            obs_data = self._observed_data

        def logpdf(pars: np.ndarray) -> float:
            named = {n: pars[i] for i, n in enumerate(free_params)}
            named.update(const_params)

            total = 0.0
            for dist_obj in lh.distributions:
                dist_name = dist_obj.name
                ordered_pars = dist_par_map[dist_name]
                call_kwargs = {}
                for par_name in ordered_pars:
                    if par_name.endswith("_observed"):
                        call_kwargs[par_name] = np.array(
                            obs_data.get(dist_name, np.zeros(1, dtype=np.float64))
                        )
                    else:
                        call_kwargs[par_name] = np.array(named.get(par_name, 1.0))
                result = model.logpdf(dist_name, **call_kwargs)
                total += float(np.sum(result))
            return total

        return logpdf

    def expected_data(self, pars: List[float]) -> List[float]:
        r"""
        Compute the expected bin counts for all distributions in the likelihood
        given model parameters *pars*.

        This is used by spey to construct Asimov datasets.  The expected
        counts are derived numerically from the likelihood via::

            lambda_i = exp(logL(n_i=1, rest=0) - logL(n_i=0, rest=0))

        Parameters
        ----------
        pars:
            Flat parameter array ``[mu, theta1, theta2, ...]``.

        Returns
        -------
        List[float]
            Concatenated expected bin counts across all distributions.
        """
        pars_arr = np.asarray(pars, dtype=np.float64)
        lh = self._likelihood
        model = self._model
        free_params = self._free_params
        const_params = self._const_params
        dist_par_map = self._dist_par_map

        named = {n: pars_arr[i] for i, n in enumerate(free_params)}
        named.update(const_params)

        all_expected: List[float] = []

        for dist_obj in lh.distributions:
            dist_name = dist_obj.name
            ordered_pars = dist_par_map[dist_name]
            obs_key = f"{dist_name}_observed"

            if obs_key not in ordered_pars:
                # Non-histfactory distribution: append a single placeholder
                all_expected.append(0.0)
                continue

            # Determine the number of bins from the known observed data
            known_obs = self._observed_data.get(dist_name)
            n_bins = len(known_obs) if known_obs is not None else 1

            # Build base call kwargs (observed = all zeros)
            base_kwargs = {}
            for par_name in ordered_pars:
                if par_name == obs_key:
                    base_kwargs[par_name] = np.zeros(n_bins, dtype=np.float64)
                else:
                    base_kwargs[par_name] = named.get(par_name, 1.0)

            logpdf_zero = float(np.sum(model.logpdf_unsafe(dist_name, **base_kwargs)))

            # For each bin, compute lambda_i = exp(logpdf(e_i) - logpdf(0))
            for i in range(n_bins):
                e_i = np.zeros(n_bins, dtype=np.float64)
                e_i[i] = 1.0
                ei_kwargs = dict(base_kwargs)
                ei_kwargs[obs_key] = e_i
                logpdf_ei = float(np.sum(model.logpdf_unsafe(dist_name, **ei_kwargs)))
                lambda_i = np.exp(logpdf_ei - logpdf_zero)
                all_expected.append(float(lambda_i))

        return all_expected

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_asimov_data(self) -> Dict[str, np.ndarray]:
        """
        Compute background-only Asimov dataset (mu=0, nuisances at nominal).

        Returns a dict mapping distribution name to expected bin counts.
        """
        # Build parameter vector at mu=0 and nominal nuisance values
        asimov_pars = np.array(
            [
                0.0 if n == self._poi_name else self._init_values.get(n, 1.0)
                for n in self._free_params
            ],
            dtype=np.float64,
        )
        expected = self.expected_data(asimov_pars.tolist())

        # Re-package into per-distribution arrays
        asimov: Dict[str, np.ndarray] = {}
        idx = 0
        for dist_obj in self._likelihood.distributions:
            dist_name = dist_obj.name
            n_bins_known = self._observed_data.get(dist_name)
            if n_bins_known is not None:
                n = len(n_bins_known)
            else:
                n = 1
            asimov[dist_name] = np.array(expected[idx : idx + n], dtype=np.float64)
            idx += n
        return asimov


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _split_data_by_dist(
    flat_data: np.ndarray,
    dist_names: List[str],
    dist_par_map: Dict[str, List[str]],
    obs_data_ref: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Split a flat external data array across distributions based on the number
    of bins in each distribution's observed data.
    """
    result: Dict[str, np.ndarray] = {}
    idx = 0
    for dist_name in dist_names:
        par_list = dist_par_map.get(dist_name, [])
        has_obs = any(p.endswith("_observed") for p in par_list)
        if not has_obs:
            continue
        ref = obs_data_ref.get(dist_name)
        n = len(ref) if ref is not None else 1
        result[dist_name] = flat_data[idx : idx + n]
        idx += n
    return result
