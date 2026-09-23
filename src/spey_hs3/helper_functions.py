"""Helper class for inspecting and manipulating HS3 workspace inputs."""

import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

__all__ = ["WorkspaceInterpreter"]

log = logging.getLogger("Spey")


def __dir__():
    return __all__


class WorkspaceInterpreter:
    """
    An HS3 workspace interpreter for bookkeeping, inspection and signal editing.

    Provides a user-friendly interface to inspect HS3 JSON workspaces and to
    edit ``histfactory_dist`` distributions — injecting signal samples,
    identifying an existing signal and its modifiers, or stripping the signal
    away — analogous to the pyhf ``WorkspaceInterpreter`` but for the HS3
    format.

    The class works purely from the raw JSON dictionary — no ``pyhs3``
    compilation is required for inspection.  The :attr:`patch` property
    returns a modified copy of the workspace ready to be passed to
    :class:`~spey_hs3.HS3Interface`.

    **Key inspection properties**

    * :attr:`observed_data` / :meth:`get_observed_data` — real observations,
      resolved through the likelihoods (preferring data over Asimov).
    * :attr:`expected_background_yields` — total background MC (signal excluded).
    * :attr:`expected_signal_yields` / :attr:`signal_samples` /
      :attr:`signal_modifiers` — the signal already present in the model.

    **Signal editing**

    * :meth:`inject_signal` — add a signal sample scaled by the POI.
    * :meth:`strip_signal` — delete all signal samples (and the orphaned POI).
    * :meth:`strip_signal_nuisances` — drop the signal's nuisance modifiers,
      keeping only the signal-strength and luminosity scaling.

    Args:
        workspace (``Dict``): HS3 JSON workspace as a Python dictionary.

    Examples:
        .. code-block:: python

            import json, spey
            with open("workspace.json") as f:
                ws_dict = json.load(f)
            interp = WorkspaceInterpreter(ws_dict)
            interp.summary()
            print(interp.signal_samples)              # {'model_SR': ['signal']}
            print(interp.expected_background_yields)  # background-only MC, per bin

            # Build a background-only model by stripping the signal:
            interp.strip_signal()
            bkg_only = interp.patch

            # ... or inject a fresh signal of your own:
            interp.reset_signal()
            interp.inject_signal("model_SR", "Signal", [3.0, 5.0, 2.0])
            stat_model = spey.get_backend("hs3")(
                hs3_dict=interp.patch,
                analysis_name="myAnalysis",
            )
    """

    #: Parameter names (lower-cased) recognised as luminosity scaling factors.
    #: HistFactory/ROOT uses ``"Lumi"``; pyhf uses ``"lumi"``.  These modifiers
    #: are normalisation (not nuisance) factors and are preserved by
    #: :meth:`strip_signal_nuisances`.
    LUMI_NAMES = frozenset({"lumi", "luminosity"})

    __slots__ = [
        "_workspace",
        "_signal_dict",  # {dist_name: {sample_name: {"contents", "errors", "poi_name"}}}
        "_to_remove",  # [dist_name, ...]  – whole distributions
        "_samples_to_remove",  # {dist_name: {sample_name, ...}}  – drop samples
        "_modifier_strip",  # {dist_name: {sample_name: {modifier_name, ...}}}
        "_cleanup_orphans",  # bool – prune orphaned params during patch
        "_lumi_names",  # set[str] – lower-cased luminosity parameter names
        "_dist_map",  # {dist_name: dist_dict}  – histfactory_dist only
        "_data_map",  # {data_name: data_dict}
        "_lh_map",  # {lh_name: lh_dict}
        "_analyses_list",
    ]

    def __init__(self, workspace: Dict):
        self._workspace = workspace
        self._signal_dict: Dict[str, Dict[str, Dict]] = {}
        self._to_remove: List[str] = []
        self._samples_to_remove: Dict[str, set] = {}
        self._modifier_strip: Dict[str, Dict[str, set]] = {}
        self._cleanup_orphans: bool = False
        self._lumi_names: set = set(self.LUMI_NAMES)

        # Fast-lookup maps built once at init
        self._dist_map: Dict[str, Dict] = {
            d["name"]: d
            for d in workspace.get("distributions", [])
            if d.get("type") == "histfactory_dist"
        }
        self._data_map: Dict[str, Dict] = {
            d["name"]: d for d in workspace.get("data", [])
        }
        self._lh_map: Dict[str, Dict] = {
            lh["name"]: lh for lh in workspace.get("likelihoods", [])
        }
        self._analyses_list: List[Dict] = workspace.get("analyses", [])

    def __getitem__(self, item):
        return self._workspace[item]

    def __repr__(self) -> str:
        n_signal = sum(len(v) for v in self.signal_samples.values())
        return (
            f"WorkspaceInterpreter("
            f"analyses={len(self._analyses_list)}, "
            f"histfactory_distributions={len(self._dist_map)}, "
            f"signal_samples={n_signal}, "
            f"injected_signals={sum(len(v) for v in self._signal_dict.values())})"
        )

    # ------------------------------------------------------------------
    # Read-only inspection properties
    # ------------------------------------------------------------------

    @property
    def distributions(self) -> List[str]:
        """
        Names of all ``histfactory_dist`` distributions in the workspace.

        Returns:
            ``List[str]``:
            Ordered list of distribution names.
        """
        return list(self._dist_map.keys())

    @property
    def all_distribution_types(self) -> Dict[str, str]:
        """
        All distributions (including constraint PDFs) with their types.

        Unlike :attr:`distributions`, this includes non-``histfactory_dist``
        entries such as constraint PDFs registered by pyhs3.

        Returns:
            ``Dict[str, str]``:
            Mapping of distribution name to its ``type`` field (``"unknown"``
            when the field is absent).
        """
        return {
            d["name"]: d.get("type", "unknown")
            for d in self._workspace.get("distributions", [])
        }

    @property
    def analyses(self) -> List[str]:
        """
        Names of all analyses in the workspace.

        Returns:
            ``List[str]``:
            Ordered list of analysis names.
        """
        return [a["name"] for a in self._analyses_list]

    @property
    def poi_names(self) -> Dict[str, List[str]]:
        """
        Parameters of interest per analysis.

        Returns:
            ``Dict[str, List[str]]``:
            Mapping ``{analysis_name: [poi, ...]}``.
        """
        return {
            a["name"]: a.get("parameters_of_interest", []) for a in self._analyses_list
        }

    @property
    def likelihoods(self) -> Dict[str, str]:
        """
        Likelihood name per analysis.

        Returns:
            ``Dict[str, str]``:
            Mapping ``{analysis_name: likelihood_name}``.  Analyses without a
            ``likelihood`` key are omitted.
        """
        return {
            a["name"]: a["likelihood"]
            for a in self._analyses_list
            if a.get("likelihood", False)
        }

    @property
    def bin_map(self) -> Dict[str, int]:
        """
        Number of (flattened) bins per ``histfactory_dist`` distribution.

        The flattened sample ``contents`` length is authoritative; when a
        distribution has no samples the binning is derived from its ``axes``
        (product of per-axis ``nbins``, or ``len(edges) - 1``).

        Returns:
            ``Dict[str, int]``:
            Mapping ``{dist_name: n_bins}``.
        """
        result: Dict[str, int] = {}
        for dist_name, dist in self._dist_map.items():
            samples = dist.get("samples", [])
            if samples:
                contents, _ = self._sample_contents_errors(samples[0])
                result[dist_name] = len(contents)
            else:
                result[dist_name] = self._axes_bin_count(dist.get("axes", []))
        return result

    @property
    def samples(self) -> Dict[str, List[str]]:
        """
        Sample names per distribution.

        Returns:
            ``Dict[str, List[str]]``:
            Mapping ``{dist_name: [sample_name, ...]}``.
        """
        return {
            dist_name: [s["name"] for s in dist.get("samples", [])]
            for dist_name, dist in self._dist_map.items()
        }

    @property
    def modifier_types(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Modifier types per sample per distribution.

        Returns:
            ``Dict[str, Dict[str, List[str]]]``:
            Mapping ``{dist_name: {sample_name: [modifier_type, ...]}}``.
        """
        result: Dict[str, Dict[str, List[str]]] = {}
        for dist_name, dist in self._dist_map.items():
            result[dist_name] = {}
            for s in dist.get("samples", []):
                result[dist_name][s["name"]] = [
                    m.get("type", "unknown") for m in s.get("modifiers", [])
                ]
        return result

    @property
    def expected_background_yields(self) -> Dict[str, List[float]]:
        """
        Total expected background yields per bin for each distribution.

        Sums the ``contents`` of every *background* sample, i.e. all samples
        that are **not** identified as signal (see :attr:`signal_samples`) and
        have not been injected via :meth:`inject_signal`.  This is the total MC
        background expectation and is, in general, distinct from the observed
        data (see :attr:`observed_data`).

        Distributions marked for removal are excluded.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping ``{dist_name: [yield_per_bin, ...]}``.
        """
        injected = {
            dist_name: set(samples.keys())
            for dist_name, samples in self._signal_dict.items()
        }
        signal = self.signal_samples
        result: Dict[str, List[float]] = {}

        for dist_name, dist in self._dist_map.items():
            if dist_name in self._to_remove:
                continue
            n_bins = self.bin_map.get(dist_name, 0)
            totals = [0.0] * n_bins
            skip = injected.get(dist_name, set()) | set(signal.get(dist_name, []))

            for sample in dist.get("samples", []):
                if sample["name"] in skip:
                    continue
                contents, _ = self._sample_contents_errors(sample)
                for i, val in enumerate(contents):
                    if i < n_bins:
                        totals[i] += val
            result[dist_name] = totals
        return result

    @property
    def expected_signal_yields(self) -> Dict[str, List[float]]:
        """
        Total expected signal yields per bin for each distribution.

        Sums the ``contents`` of every sample identified as signal (see
        :attr:`signal_samples`).  Only signal already present in the workspace
        is counted; yields queued through :meth:`inject_signal` are reported
        separately by :attr:`signal_per_distribution`.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping ``{dist_name: [yield_per_bin, ...]}``.  Distributions
            without a signal sample are omitted.
        """
        signal = self.signal_samples
        result: Dict[str, List[float]] = {}
        for dist_name, sample_names in signal.items():
            n_bins = self.bin_map.get(dist_name, 0)
            totals = [0.0] * n_bins
            for sample in self._dist_map[dist_name].get("samples", []):
                if sample["name"] not in sample_names:
                    continue
                contents, _ = self._sample_contents_errors(sample)
                for i, val in enumerate(contents):
                    if i < n_bins:
                        totals[i] += val
            result[dist_name] = totals
        return result

    @property
    def observed_data(self) -> Dict[str, List[float]]:
        """
        Observed data per distribution.

        Data is resolved through the likelihoods, which pair each distribution
        positionally with a data object (``distributions[i]`` ↔ ``data[i]`` per
        the HS3 specification).  A single distribution may be referenced by
        several likelihoods with *different* data (e.g. a real ``obsData`` and a
        pseudo ``asimovData``); in that case the genuine observation is
        preferred over the Asimov dataset.  The HS3 standard carries no formal
        flag distinguishing the two, so the (documented) ``"asimov"`` naming
        convention is used as the discriminator.

        For unambiguous, per-likelihood access use
        :attr:`observed_data_per_likelihood` or :meth:`get_observed_data`.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping ``{dist_name: [counts_per_bin]}``.
        """
        return self.get_observed_data()

    @property
    def observed_data_per_likelihood(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Data resolved separately for every likelihood.

        Uses the positional ``distributions[i]`` ↔ ``data[i]`` pairing without
        applying any Asimov/observed heuristic, exposing the full mapping for
        every named likelihood in the workspace.

        Returns:
            ``Dict[str, Dict[str, List[float]]]``:
            Mapping ``{likelihood_name: {dist_name: [counts_per_bin]}}``.
        """
        result: Dict[str, Dict[str, List[float]]] = {}
        for lh_name, lh in self._lh_map.items():
            result[lh_name] = self._resolve_likelihood_data(lh)
        return result

    def get_observed_data(
        self,
        analysis_name: Optional[str] = None,
        likelihood_name: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Resolve observed data, optionally scoped to one analysis or likelihood.

        When neither *analysis_name* nor *likelihood_name* is given, data is
        collected across **all** likelihoods and, where a distribution is paired
        with several datasets, the non-Asimov observation is preferred.

        Args:
            analysis_name (``Optional[str]``, default ``None``): Resolve the
                data of the likelihood referenced by this analysis.
            likelihood_name (``Optional[str]``, default ``None``): Resolve the
                data of this likelihood directly.  Takes precedence over
                *analysis_name* when both are provided.

        Raises:
            ``ValueError``: If *likelihood_name* (or the likelihood derived
                from *analysis_name*) is not found in the workspace.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping ``{dist_name: [counts_per_bin]}``.
        """
        if likelihood_name is None and analysis_name is not None:
            likelihood_name = self.get_analysis(analysis_name).get("likelihood")

        if likelihood_name is not None:
            lh = self._lh_map.get(likelihood_name)
            if lh is None:
                raise ValueError(
                    f"Likelihood '{likelihood_name}' not found. "
                    f"Available: {list(self._lh_map.keys())}"
                )
            return self._resolve_likelihood_data(lh)

        # No scope: merge all likelihoods, preferring real data over Asimov.
        result: Dict[str, List[float]] = {}
        chosen_is_asimov: Dict[str, bool] = {}
        for lh in self._lh_map.values():
            lh_is_asimov = self._is_asimov_like(lh.get("name"))
            for dist_name, data_ref in zip(
                lh.get("distributions", []), lh.get("data", [])
            ):
                if dist_name not in self._dist_map:
                    continue
                contents = self._resolve_data_contents(data_ref)
                if contents is None:
                    continue
                is_asimov = lh_is_asimov or self._is_asimov_like(data_ref)
                if dist_name not in result or (
                    chosen_is_asimov.get(dist_name) and not is_asimov
                ):
                    result[dist_name] = contents
                    chosen_is_asimov[dist_name] = is_asimov
        return result

    @property
    def parameters(self) -> Dict[str, Dict]:
        """
        Parameter metadata from the first ``parameter_point``.

        Returns:
            ``Dict[str, Dict]``:
            Mapping ``{param_name: {"value": float, "const": bool}}``.  Returns
            an empty dict when no ``parameter_points`` are defined.
        """
        pps = self._workspace.get("parameter_points", [])
        if not pps:
            return {}
        return {
            p["name"]: {"value": p.get("value", 0.0), "const": p.get("const", False)}
            for p in pps[0].get("parameters", [])
        }

    @property
    def signal_per_distribution(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Currently injected signal yields.

        Returns:
            ``Dict[str, Dict[str, List[float]]]``:
            Mapping ``{dist_name: {sample_name: [yields_per_bin]}}``.
        """
        return {
            dist_name: {sname: spec["contents"] for sname, spec in samples.items()}
            for dist_name, samples in self._signal_dict.items()
        }

    @property
    def remove_list(self) -> List[str]:
        """
        Distributions currently marked for removal.

        Returns:
            ``List[str]``:
            Distribution names queued for removal in the next :attr:`patch`.
        """
        return list(self._to_remove)

    # ------------------------------------------------------------------
    # Signal identification
    # ------------------------------------------------------------------

    @property
    def poi_parameters(self) -> List[str]:
        """
        All parameters of interest declared across every analysis (deduplicated).

        The POI (signal strength :math:`\\mu`) is the anchor used to identify
        signal samples and their modifiers.

        Returns:
            ``List[str]``:
            Ordered, deduplicated list of POI names.
        """
        seen: List[str] = []
        for a in self._analyses_list:
            for p in a.get("parameters_of_interest", []):
                if p not in seen:
                    seen.append(p)
        return seen

    @property
    def signal_samples(self) -> Dict[str, List[str]]:
        """
        Samples identified as signal, per distribution.

        A sample is flagged as signal when at least one of its modifiers is
        tied to a parameter of interest (typically a ``normfactor`` on
        :math:`\\mu`).  Distributions with no signal sample are omitted.

        Returns:
            ``Dict[str, List[str]]``:
            Mapping ``{dist_name: [sample_name, ...]}``.
        """
        pois = set(self.poi_parameters)
        if not pois:
            return {}
        result: Dict[str, List[str]] = {}
        for dist_name, dist in self._dist_map.items():
            found = [
                s["name"]
                for s in dist.get("samples", [])
                if self._sample_is_signal(s, pois)
            ]
            if found:
                result[dist_name] = found
        return result

    def is_signal_sample(self, dist_name: str, sample_name: str) -> bool:
        """
        Check whether a sample carries a POI modifier.

        Args:
            dist_name (``str``): Name of the ``histfactory_dist`` distribution.
            sample_name (``str``): Name of the sample to test.

        Returns:
            ``bool``:
            ``True`` if *sample_name* in *dist_name* has at least one modifier
            that references a parameter of interest.
        """
        dist = self._dist_map.get(dist_name)
        if dist is None:
            return False
        pois = set(self.poi_parameters)
        for s in dist.get("samples", []):
            if s.get("name") == sample_name:
                return self._sample_is_signal(s, pois)
        return False

    @property
    def signal_modifiers(self) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Classified modifiers of every signal sample.

        Each modifier entry in the returned structure has the following keys:

        .. code-block:: python

            {
                "name":       "<modifier name>",
                "type":       "<modifier type, e.g. normfactor>",
                "parameters": ["<referenced parameter names>"],
                "role":       "poi" | "lumi" | "nuisance",
            }

        The ``"role"`` values have the following meanings:

        * ``"poi"`` — the signal-strength modifier (tied to a POI).
        * ``"lumi"`` — a luminosity normalisation factor.
        * ``"nuisance"`` — any other modifier (systematics, MC stat, free
          normalisations, …); these are what :meth:`strip_signal_nuisances`
          removes.

        Returns:
            ``Dict[str, Dict[str, List[Dict]]]``:
            Mapping ``{dist_name: {sample_name: [modifier_info, ...]}}``.
        """
        pois = set(self.poi_parameters)
        result: Dict[str, Dict[str, List[Dict]]] = {}
        for dist_name, sample_names in self.signal_samples.items():
            result[dist_name] = {}
            for sample in self._dist_map[dist_name].get("samples", []):
                if sample["name"] not in sample_names:
                    continue
                infos: List[Dict] = []
                for m in sample.get("modifiers", []):
                    infos.append(
                        {
                            "name": m.get("name"),
                            "type": m.get("type"),
                            "parameters": sorted(self._modifier_param_names(m)),
                            "role": self._modifier_role(m, pois),
                        }
                    )
                result[dist_name][sample["name"]] = infos
        return result

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_analysis(self, name: Optional[str] = None) -> Dict:
        """
        Retrieve an analysis dictionary by name.

        Args:
            name (``Optional[str]``, default ``None``): Analysis name.  When
                ``None``, the first analysis in the workspace is returned.

        Raises:
            ``ValueError``: If no analyses exist or *name* is not found.

        Returns:
            ``Dict``:
            The raw analysis dictionary from the workspace.
        """
        if not self._analyses_list:
            raise ValueError("No analyses found in workspace.")
        if name is None:
            return self._analyses_list[0]
        for a in self._analyses_list:
            if a["name"] == name:
                return a
        raise ValueError(f"Analysis '{name}' not found. Available: {self.analyses}")

    def get_distributions(
        self,
        analysis_name: Optional[str] = None,
        likelihood_name: Optional[str] = None,
    ) -> List[str]:
        """
        Return ``histfactory_dist`` names for a given analysis or likelihood.

        If neither argument is given all ``histfactory_dist`` distributions
        in the workspace are returned.

        Args:
            analysis_name (``Optional[str]``, default ``None``): Filter by
                analysis — resolves the associated likelihood automatically.
            likelihood_name (``Optional[str]``, default ``None``): Filter by
                likelihood name directly.

        Raises:
            ``ValueError``: If *likelihood_name* (resolved explicitly or via
                *analysis_name*) is not found in the workspace.

        Returns:
            ``List[str]``:
            Matching ``histfactory_dist`` distribution names.
        """
        if analysis_name is not None:
            a = self.get_analysis(analysis_name)
            likelihood_name = a.get("likelihood")
        if likelihood_name is not None:
            lh = self._lh_map.get(likelihood_name)
            if lh is None:
                raise ValueError(
                    f"Likelihood '{likelihood_name}' not found. "
                    f"Available: {list(self._lh_map.keys())}"
                )
            return [d for d in lh.get("distributions", []) if d in self._dist_map]
        return self.distributions

    def guess_region_type(self, dist_name: str) -> str:
        """
        Guess region type from the distribution name.

        Uses a simple substring match (case-insensitive) for ``"SR"``, ``"CR"``
        and ``"VR"``.

        Args:
            dist_name (``str``): Distribution name to inspect.

        Returns:
            ``str``:
            One of ``"SR"``, ``"CR"``, ``"VR"``, or ``"__unknown__"`` when no
            pattern is recognised.
        """
        upper = dist_name.upper()
        for tp in ("SR", "CR", "VR"):
            if tp in upper:
                return tp
        return "__unknown__"

    def guess_signal_regions(self) -> List[str]:
        """
        Return distribution names that appear to be signal regions.

        Uses :meth:`guess_region_type` to identify ``"SR"`` distributions.

        Returns:
            ``List[str]``:
            Distribution names whose names contain ``"SR"`` (case-insensitive).
        """
        return [n for n in self.distributions if self.guess_region_type(n) == "SR"]

    def guess_control_regions(self) -> List[str]:
        """
        Return distribution names that appear to be control or validation regions.

        Uses :meth:`guess_region_type` to identify ``"CR"`` and ``"VR"``
        distributions.

        Returns:
            ``List[str]``:
            Distribution names whose names contain ``"CR"`` or ``"VR"``
            (case-insensitive).
        """
        return [
            n for n in self.distributions if self.guess_region_type(n) in ("CR", "VR")
        ]

    def get_sample_yields(
        self, dist_name: str, sample_name: str
    ) -> Tuple[List[float], List[float]]:
        """
        Return the bin contents and errors for a specific sample.

        Args:
            dist_name (``str``): Name of the ``histfactory_dist`` distribution.
            sample_name (``str``): Name of the sample within that distribution.

        Raises:
            ``ValueError``: If *dist_name* is not a ``histfactory_dist`` or
                *sample_name* is not found within it.

        Returns:
            ``Tuple[List[float], List[float]]``:
            ``(contents, errors)`` — per-bin yields and MC statistical errors.
        """
        if dist_name not in self._dist_map:
            raise ValueError(
                f"'{dist_name}' is not a histfactory_dist. "
                f"Available: {self.distributions[:10]}"
                + (" ..." if len(self.distributions) > 10 else "")
            )
        for s in self._dist_map[dist_name].get("samples", []):
            if s["name"] == sample_name:
                data = s.get("data", {})
                if isinstance(data, dict):
                    contents = [float(v) for v in data.get("contents", [])]
                    errors = [float(v) for v in data.get("errors", [])]
                else:
                    contents = [float(v) for v in data]
                    errors = [0.0] * len(contents)
                return contents, errors
        raise ValueError(
            f"Sample '{sample_name}' not found in '{dist_name}'. "
            f"Available: {self.samples.get(dist_name, [])}"
        )

    # ------------------------------------------------------------------
    # Signal injection
    # ------------------------------------------------------------------

    def inject_signal(
        self,
        dist_name: str,
        sample_name: str,
        yields: List[float],
        errors: Optional[List[float]] = None,
        poi_name: Optional[str] = None,
    ) -> None:
        """
        Inject a signal sample into a ``histfactory_dist`` distribution.

        The injected sample receives a ``normfactor`` modifier tied to
        *poi_name* so that the signal strength :math:`\\mu` scales the sample
        when used with :class:`~spey_hs3.HS3Interface`.

        Args:
            dist_name (``str``): Target distribution name (must be a
                ``histfactory_dist``).
            sample_name (``str``): Name for the new signal sample.  Must not
                already exist in the distribution.
            yields (``List[float]``): Per-bin signal yields at :math:`\\mu = 1`.
            errors (``Optional[List[float]]``, default ``None``): Per-bin MC
                statistical errors.  Defaults to zeros when ``None``.
            poi_name (``Optional[str]``, default ``None``): Name of the
                parameter of interest to attach.  When ``None``, the first POI
                of the first analysis is used.

        Raises:
            ``ValueError``: If *dist_name* does not exist, the bin count of
                *yields* does not match the distribution, the length of *errors*
                differs from *yields*, or *sample_name* already exists.
        """
        if dist_name not in self._dist_map:
            available = self.distributions
            msg = f"'{dist_name}' is not a histfactory_dist in this workspace."
            if available:
                shown = available[:10]
                msg += f"\nAvailable distributions (first {len(shown)}): {shown}"
                if len(available) > 10:
                    msg += " ..."
            raise ValueError(msg)

        expected_bins = self.bin_map.get(dist_name, 0)
        if len(yields) != expected_bins:
            raise ValueError(
                f"Bin count mismatch for '{dist_name}': "
                f"expected {expected_bins} bins, got {len(yields)}."
            )

        existing = self.samples.get(dist_name, [])
        already_injected = list((self._signal_dict.get(dist_name) or {}).keys())
        if sample_name in existing or sample_name in already_injected:
            raise ValueError(
                f"Sample '{sample_name}' already exists in distribution '{dist_name}'."
            )

        if poi_name is None:
            poi_name = self._default_poi()

        errs = list(errors) if errors is not None else [0.0] * len(yields)
        if len(errs) != len(yields):
            raise ValueError(
                f"Length of errors ({len(errs)}) must match yields ({len(yields)})."
            )

        self._signal_dict.setdefault(dist_name, {})[sample_name] = {
            "contents": list(yields),
            "errors": errs,
            "poi_name": poi_name,
        }
        log.debug(
            "Signal injected: dist='%s', sample='%s', poi='%s', n_bins=%d",
            dist_name,
            sample_name,
            poi_name,
            len(yields),
        )

    def inject_signals(
        self,
        signal_map: Dict[str, Dict[str, Union[List[float], Dict]]],
        poi_name: Optional[str] = None,
    ) -> None:
        """
        Inject multiple signal samples at once.

        Each entry in *signal_map* is forwarded to :meth:`inject_signal`.  The
        *spec* value per sample can be either a plain list of per-bin yields or
        a dict with keys ``"contents"`` (required) and ``"errors"`` (optional):

        .. code-block:: python

            interp.inject_signals(
                {
                    "model_SR_0j": {
                        "Signal": [3.0, 5.0, 2.0],
                    },
                    "model_SR_1j": {
                        "Signal": {
                            "contents": [1.0, 2.0],
                            "errors":   [0.1, 0.2],
                        },
                    },
                }
            )

        Args:
            signal_map (``Dict[str, Dict[str, Union[List[float], Dict]]]``):
                Nested mapping ``{dist_name: {sample_name: spec}}``.
            poi_name (``Optional[str]``, default ``None``): Parameter of
                interest to attach to every injected sample.  When ``None``,
                the first POI of the first analysis is used.
        """
        for dist_name, samples_spec in signal_map.items():
            for sample_name, spec in samples_spec.items():
                if isinstance(spec, dict):
                    yields = spec["contents"]
                    errors = spec.get("errors")
                else:
                    yields = list(spec)
                    errors = None
                self.inject_signal(
                    dist_name,
                    sample_name,
                    yields,
                    errors=errors,
                    poi_name=poi_name,
                )

    def remove_distribution(self, dist_name: str) -> None:
        """
        Mark a distribution for removal from the patched workspace.

        The distribution will be removed from both the ``distributions`` list
        and all likelihood references in :attr:`patch`.  If *dist_name* is not
        found a warning is logged and the call is silently ignored.

        Args:
            dist_name (``str``): Name of the ``histfactory_dist`` to remove.
        """
        if dist_name not in self._dist_map:
            log.warning(
                "Distribution '%s' not found in workspace; ignoring remove request.",
                dist_name,
            )
            return
        if dist_name not in self._to_remove:
            self._to_remove.append(dist_name)

    def reset_signal(self) -> None:
        """
        Clear every queued mutation.

        Resets all injections, distribution removals, sample strips, and
        modifier strips queued since construction or the last call to this
        method.
        """
        self._signal_dict = {}
        self._to_remove = []
        self._samples_to_remove = {}
        self._modifier_strip = {}
        self._cleanup_orphans = False

    # ------------------------------------------------------------------
    # Signal stripping
    # ------------------------------------------------------------------

    def strip_signal(
        self,
        dist_name: Optional[str] = None,
        sample_name: Optional[str] = None,
        cleanup_parameters: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Remove all signal information from the workspace.

        Every sample identified as signal (see :attr:`signal_samples`) is queued
        for deletion from :attr:`patch`.  When *cleanup_parameters* is ``True``
        any parameter that becomes orphaned as a result — most notably the
        parameter of interest :math:`\\mu` — is also removed from the
        ``domains``, ``parameter_points`` and analysis ``parameters_of_interest``
        sections, yielding a clean background-only model.

        Args:
            dist_name (``Optional[str]``, default ``None``): Restrict stripping
                to this distribution.  ``None`` targets every distribution.
            sample_name (``Optional[str]``, default ``None``): Restrict
                stripping to this sample (requires *dist_name* unless the name
                is unique).  ``None`` targets every signal sample.
            cleanup_parameters (``bool``, default ``True``): Prune parameters
                orphaned by the removal.

        Returns:
            ``Dict[str, List[str]]``:
            Signal samples queued for removal: ``{dist_name: [sample_name, ...]}``.
        """
        targets = self._resolve_signal_targets(dist_name, sample_name)
        if not targets:
            log.warning(
                "strip_signal: no signal samples matched (dist=%s, sample=%s).",
                dist_name,
                sample_name,
            )
        for d, snames in targets.items():
            self._samples_to_remove.setdefault(d, set()).update(snames)
        if cleanup_parameters and targets:
            self._cleanup_orphans = True
        return {d: sorted(s) for d, s in targets.items()}

    def strip_signal_nuisances(
        self,
        dist_name: Optional[str] = None,
        sample_name: Optional[str] = None,
        keep_lumi: bool = True,
        cleanup_parameters: bool = True,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Remove nuisance modifiers from signal samples, keeping normalisation.

        For every signal sample, all modifiers are stripped **except** the
        signal-strength modifier (the one tied to a POI) and — when *keep_lumi*
        is ``True`` — the luminosity normalisation factor.  Systematic modifiers
        (``normsys``, ``histosys``, ``shapesys``, ``staterror``, …) and any
        other non-POI/non-luminosity modifier are dropped, leaving the signal
        scaled only by :math:`\\mu` (and luminosity).

        When *cleanup_parameters* is ``True`` nuisance parameters that become
        orphaned by the strip are also pruned from ``domains`` and
        ``parameter_points``.

        Args:
            dist_name (``Optional[str]``, default ``None``): Restrict stripping
                to this distribution.  ``None`` targets every distribution.
            sample_name (``Optional[str]``, default ``None``): Restrict
                stripping to this sample.  ``None`` targets every signal sample.
            keep_lumi (``bool``, default ``True``): Preserve the luminosity
                normalisation modifier.
            cleanup_parameters (``bool``, default ``True``): Prune parameters
                orphaned by the strip.

        Returns:
            ``Dict[str, Dict[str, List[str]]]``:
            Modifier names queued for removal:
            ``{dist_name: {sample_name: [modifier_name, ...]}}``.
        """
        pois = set(self.poi_parameters)
        targets = self._resolve_signal_targets(dist_name, sample_name)
        queued: Dict[str, Dict[str, List[str]]] = {}

        for d, snames in targets.items():
            for sname in snames:
                sample = self._get_sample(d, sname)
                if sample is None:
                    continue
                remove_names = set()
                for m in sample.get("modifiers", []):
                    role = self._modifier_role(m, pois)
                    if role == "poi":
                        continue
                    if role == "lumi" and keep_lumi:
                        continue
                    name = m.get("name")
                    if name is not None:
                        remove_names.add(name)
                if remove_names:
                    self._modifier_strip.setdefault(d, {}).setdefault(
                        sname, set()
                    ).update(remove_names)
                    queued.setdefault(d, {})[sname] = sorted(remove_names)

        if cleanup_parameters and queued:
            self._cleanup_orphans = True
        if not queued:
            log.warning(
                "strip_signal_nuisances: no signal nuisance modifiers matched "
                "(dist=%s, sample=%s).",
                dist_name,
                sample_name,
            )
        return queued

    # ------------------------------------------------------------------
    # Workspace patching
    # ------------------------------------------------------------------

    @property
    def patch(self) -> Dict:
        """
        Deep copy of the workspace with every queued mutation applied.

        Mutations are applied in order: signal injection, signal-sample removal
        (:meth:`strip_signal`), modifier stripping
        (:meth:`strip_signal_nuisances`), distribution removal
        (:meth:`remove_distribution`), and — when requested — orphaned-parameter
        cleanup.  The returned dict can be passed directly to
        :class:`~spey_hs3.HS3Interface` as ``hs3_dict``, or saved to JSON with
        :meth:`save_patch`.

        Raises:
            ``ValueError``: If no mutations have been queued.

        Returns:
            ``Dict``:
            Patched HS3 workspace dictionary, ready for use with
            :class:`~spey_hs3.HS3Interface`.
        """
        if not (
            self._signal_dict
            or self._to_remove
            or self._samples_to_remove
            or self._modifier_strip
        ):
            raise ValueError(
                "Nothing to patch: use inject_signal(), remove_distribution(), "
                "strip_signal() or strip_signal_nuisances() first."
            )

        data = copy.deepcopy(self._workspace)
        dists = data.get("distributions", [])

        # --- Inject signal samples ---
        for dist_name, samples_spec in self._signal_dict.items():
            for dist in dists:
                if (
                    dist.get("name") == dist_name
                    and dist.get("type") == "histfactory_dist"
                ):
                    for sample_name, spec in samples_spec.items():
                        poi = spec["poi_name"]
                        new_sample = {
                            "name": sample_name,
                            "data": {
                                "contents": spec["contents"],
                                "errors": spec["errors"],
                            },
                            "modifiers": [
                                {
                                    "name": poi,
                                    "type": "normfactor",
                                    "parameter": poi,
                                }
                            ],
                        }
                        dist.setdefault("samples", []).append(new_sample)
                    break

        # --- Strip signal samples / modifiers ---
        if self._samples_to_remove or self._modifier_strip:
            for dist in dists:
                if dist.get("type") != "histfactory_dist":
                    continue
                name = dist.get("name")
                drop = self._samples_to_remove.get(name, set())
                strip = self._modifier_strip.get(name, {})
                new_samples = []
                for s in dist.get("samples", []):
                    if s.get("name") in drop:
                        continue
                    rm = strip.get(s.get("name"))
                    if rm:
                        s["modifiers"] = [
                            m for m in s.get("modifiers", []) if m.get("name") not in rm
                        ]
                    new_samples.append(s)
                dist["samples"] = new_samples
                if not new_samples:
                    log.warning(
                        "Distribution '%s' has no samples left after stripping; "
                        "consider remove_distribution('%s').",
                        name,
                        name,
                    )

        # --- Remove distributions ---
        if self._to_remove:
            data["distributions"] = [
                d
                for d in dists
                if not (
                    d.get("type") == "histfactory_dist"
                    and d.get("name") in self._to_remove
                )
            ]
            for lh in data.get("likelihoods", []):
                new_dists, new_data_refs = [], []
                for d, dat in zip(lh.get("distributions", []), lh.get("data", [])):
                    if d not in self._to_remove:
                        new_dists.append(d)
                        new_data_refs.append(dat)
                lh["distributions"] = new_dists
                lh["data"] = new_data_refs

        # --- Register injected POIs in all analyses ---
        injected_pois = {
            spec["poi_name"]
            for samples in self._signal_dict.values()
            for spec in samples.values()
        }
        for poi in injected_pois:
            for a in data.get("analyses", []):
                pois_list = a.setdefault("parameters_of_interest", [])
                if poi not in pois_list:
                    pois_list.append(poi)

        # --- Prune parameters orphaned by stripping ---
        if self._cleanup_orphans and (self._samples_to_remove or self._modifier_strip):
            self._cleanup_orphan_parameters(data)

        return data

    def save_patch(self, path: str) -> None:
        """
        Save the patched workspace to a JSON file.

        Args:
            path (``str``): Destination file path (e.g. ``"patched_workspace.json"``).
        """
        import json

        patched = self.patch
        with open(path, "w") as fh:
            json.dump(patched, fh, indent=2)
        log.info("Patched workspace saved to '%s'.", path)

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------

    def summary(
        self,
        analysis_name: Optional[str] = None,
        show_samples: bool = False,
        show_parameters: bool = False,
        max_dists: int = 50,
    ) -> None:
        """
        Print a human-readable summary of the workspace.

        Args:
            analysis_name (``Optional[str]``, default ``None``): Restrict the
                distribution listing to one analysis.  ``None`` shows all
                analyses.
            show_samples (``bool``, default ``False``): Also list each sample
                name per distribution.
            show_parameters (``bool``, default ``False``): Print free and const
                parameter counts.
            max_dists (``int``, default ``50``): Maximum number of distributions
                to print per analysis.
        """
        ws = self._workspace

        # Header
        meta = ws.get("metadata", {})
        sep = "=" * 60
        print(sep)
        print("HS3 Workspace Summary")
        if meta:
            if "hs3_version" in meta:
                print(f"  hs3_version : {meta['hs3_version']}")
            if "description" in meta:
                print(f"  description : {meta['description']}")
        print(f"  analyses              : {len(self._analyses_list)}")
        print(f"  likelihoods           : {len(self._lh_map)}")
        print(f"  histfactory dists     : {len(self._dist_map)}")
        print(f"  data entries          : {len(self._data_map)}")
        print()

        # Per-analysis detail
        bm = self.bin_map
        obs = self.observed_data
        signal = self.signal_samples
        for a in self._analyses_list:
            if analysis_name and a["name"] != analysis_name:
                continue
            print(f"Analysis : {a['name']}")
            pois = a.get("parameters_of_interest", [])
            print(f"  POIs      : {pois}")
            print(f"  Likelihood: {a.get('likelihood', '?')}")
            dists_in_lh = self.get_distributions(analysis_name=a["name"])
            print(f"  Distributions ({len(dists_in_lh)}):")

            for i, dname in enumerate(dists_in_lh[:max_dists]):
                region = self.guess_region_type(dname)
                tag = f"[{region}]" if region != "__unknown__" else "     "
                n = bm.get(dname, "?")
                n_obs = len(obs.get(dname, []))
                injected = list((self._signal_dict.get(dname) or {}).keys())
                inj_str = f"  <- signal: {injected}" if injected else ""
                sig_here = signal.get(dname, [])
                sig_str = f"  signal: {sig_here}" if sig_here else ""
                removed_str = "  [REMOVED]" if dname in self._to_remove else ""
                obs_str = f"  obs={n_obs}" if n_obs else ""
                bins_str = f"({n} bin{'s' if n != 1 else ''})"
                print(
                    f"    {i+1:4d}. {tag} {dname}  {bins_str}"
                    f"{obs_str}{sig_str}{inj_str}{removed_str}"
                )
                if show_samples:
                    for sname in self.samples.get(dname, []):
                        if sname in (self._signal_dict.get(dname) or {}):
                            flag = " [injected signal]"
                        elif sname in sig_here:
                            flag = " [signal]"
                        else:
                            flag = ""
                        print(f"              sample: {sname}{flag}")

            if len(dists_in_lh) > max_dists:
                print(
                    f"    ... ({len(dists_in_lh) - max_dists} more not shown; "
                    f"increase max_dists to see all)"
                )
            print()

        # Parameter summary
        if show_parameters:
            params = self.parameters
            if params:
                free = [n for n, p in params.items() if not p["const"]]
                const = [n for n, p in params.items() if p["const"]]
                print(f"Parameters ({len(params)} total):")
                print(f"  Free  : {len(free)}")
                print(f"  Const : {len(const)}")
                print()

        # Signal-in-workspace summary
        sig_mods = self.signal_modifiers
        if sig_mods:
            n_sig = sum(len(v) for v in sig_mods.values())
            print(
                f"Signal in workspace: {n_sig} sample(s)  "
                f"(POIs: {self.poi_parameters})"
            )
            for dname, samples in sig_mods.items():
                for sname, mods in samples.items():
                    roles = ", ".join(f"{m['name']}[{m['role']}]" for m in mods) or "none"
                    print(f"  {dname} / {sname}  modifiers: {roles}")
            print()

        # Injection summary
        if self._signal_dict:
            total_samples = sum(len(v) for v in self._signal_dict.values())
            print(
                f"Injected signal: {total_samples} sample(s) across "
                f"{len(self._signal_dict)} distribution(s)"
            )
            for dname, samples in self._signal_dict.items():
                for sname, spec in samples.items():
                    poi = spec["poi_name"]
                    n = len(spec["contents"])
                    print(
                        f"  {dname} / {sname}  (poi={poi}, {n} bin{'s' if n != 1 else ''})"
                    )
            print()

        # Pending strips
        if self._samples_to_remove:
            print("Signal samples to strip:")
            for dname, snames in self._samples_to_remove.items():
                print(f"  {dname}: {sorted(snames)}")
            print()
        if self._modifier_strip:
            print("Signal modifiers to strip:")
            for dname, smap in self._modifier_strip.items():
                for sname, mods in smap.items():
                    print(f"  {dname} / {sname}: {sorted(mods)}")
            print()

        if self._to_remove:
            print(f"Distributions to remove ({len(self._to_remove)}):")
            for dname in self._to_remove:
                print(f"  {dname}")
            print()

        print(sep)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _axes_bin_count(axes: List[Dict]) -> int:
        """
        Compute flattened bin count from HS3 ``axes``.

        Supports both regular-grid axes (``nbins``) and variable-width axes
        (``edges``).

        Args:
            axes (``List[Dict]``): List of HS3 axis dictionaries.

        Returns:
            ``int``:
            Total number of bins (product over axes), or ``0`` when *axes* is
            empty or contains no recognised binning keys.
        """
        total = 1
        found = False
        for ax in axes:
            if "nbins" in ax:
                total *= int(ax["nbins"])
                found = True
            elif "edges" in ax and isinstance(ax["edges"], list):
                total *= max(len(ax["edges"]) - 1, 0)
                found = True
        return total if found else 0

    @staticmethod
    def _sample_contents_errors(sample: Dict) -> Tuple[List[float], List[float]]:
        """
        Extract bin contents and errors from a sample dict.

        Tolerates both the dict form ``{"contents": [...], "errors": [...]}``
        and the legacy list form where ``data`` is a bare list of counts.

        Args:
            sample (``Dict``): Raw sample dictionary from the workspace.

        Returns:
            ``Tuple[List[float], List[float]]``:
            ``(contents, errors)`` with *errors* padded to zeros when absent or
            length-mismatched.
        """
        data = sample.get("data", {})
        if isinstance(data, dict):
            contents = [float(v) for v in data.get("contents", [])]
            errors = [float(v) for v in data.get("errors", [])]
        else:
            contents = [float(v) for v in data]
            errors = []
        if len(errors) != len(contents):
            errors = [0.0] * len(contents)
        return contents, errors

    @staticmethod
    def _is_asimov_like(name) -> bool:
        """
        Heuristic test for Asimov (pseudo) dataset names.

        Args:
            name: Value to test (typically a string; non-strings always return
                ``False``).

        Returns:
            ``bool``:
            ``True`` when *name* is a string containing ``"asimov"``
            (case-insensitive).
        """
        return isinstance(name, str) and "asimov" in name.lower()

    def _resolve_data_contents(self, data_ref) -> Optional[List[float]]:
        """
        Resolve a likelihood ``data`` entry to a list of bin contents.

        Per the HS3 spec, a data entry is either a string naming a top-level
        data object or, for single-dimensional distributions, inline numeric
        values (a list, tuple, or scalar).

        Args:
            data_ref: Raw entry from a likelihood ``data`` list.

        Returns:
            ``Optional[List[float]]``:
            Resolved bin contents, or ``None`` when the reference cannot be
            resolved.
        """
        if isinstance(data_ref, str):
            entry = self._data_map.get(data_ref)
            if entry is None:
                return None
            return [float(v) for v in entry.get("contents", [])]
        if isinstance(data_ref, (list, tuple)):
            return [float(v) for v in data_ref]
        if isinstance(data_ref, (int, float)):
            return [float(data_ref)]
        return None

    def _resolve_likelihood_data(self, lh: Dict) -> Dict[str, List[float]]:
        """
        Resolve ``{dist_name: contents}`` for one likelihood.

        Uses positional pairing (``distributions[i]`` ↔ ``data[i]``) as
        specified by the HS3 standard.

        Args:
            lh (``Dict``): Raw likelihood dictionary from the workspace.

        Returns:
            ``Dict[str, List[float]]``:
            Mapping ``{dist_name: [counts_per_bin]}`` for all resolvable
            distributions in the likelihood.
        """
        result: Dict[str, List[float]] = {}
        for dist_name, data_ref in zip(
            lh.get("distributions", []), lh.get("data", [])
        ):
            if dist_name not in self._dist_map or dist_name in result:
                continue
            contents = self._resolve_data_contents(data_ref)
            if contents is not None:
                result[dist_name] = contents
        return result

    @staticmethod
    def _modifier_param_names(modifier: Dict) -> set:
        """
        Return the set of model-parameter names a modifier references.

        Covers both the single-``parameter`` modifiers (``normfactor``,
        ``normsys``, ``histosys``) and the per-bin ``parameters`` modifiers
        (``shapesys``, ``staterror``, ``shapefactor``).  ``custom`` modifiers
        reference a workspace *function* by their ``name`` (not a domain
        parameter) and therefore contribute nothing here.

        Args:
            modifier (``Dict``): Raw modifier dictionary.

        Returns:
            ``set``:
            Set of parameter name strings referenced by the modifier.
        """
        names = set()
        param = modifier.get("parameter")
        if isinstance(param, str):
            names.add(param)
        params = modifier.get("parameters")
        if isinstance(params, (list, tuple)):
            names.update(p for p in params if isinstance(p, str))
        return names

    def _sample_is_signal(self, sample: Dict, pois: set) -> bool:
        """
        Test whether a sample should be classified as signal.

        A sample is signal if any of its modifiers references a parameter of
        interest.

        Args:
            sample (``Dict``): Raw sample dictionary.
            pois (``set``): Set of POI parameter names to match against.

        Returns:
            ``bool``:
            ``True`` when the sample has at least one modifier tied to a POI.
        """
        if not pois:
            return False
        for m in sample.get("modifiers", []):
            if self._modifier_param_names(m) & pois:
                return True
        return False

    def _is_lumi_modifier(self, modifier: Dict) -> bool:
        """
        Test whether a modifier is a luminosity normalisation factor.

        A modifier is considered a luminosity factor when its type is
        ``"normfactor"`` and its ``parameter`` or ``name`` field matches a
        known luminosity name (see :attr:`LUMI_NAMES`).

        Args:
            modifier (``Dict``): Raw modifier dictionary.

        Returns:
            ``bool``:
            ``True`` when the modifier is a luminosity normalisation factor.
        """
        if modifier.get("type") != "normfactor":
            return False
        for key in ("parameter", "name"):
            val = modifier.get(key)
            if isinstance(val, str) and val.lower() in self._lumi_names:
                return True
        return False

    def _modifier_role(self, modifier: Dict, pois: set) -> str:
        """
        Classify a modifier as ``"poi"``, ``"lumi"`` or ``"nuisance"``.

        Args:
            modifier (``Dict``): Raw modifier dictionary.
            pois (``set``): Set of POI parameter names.

        Returns:
            ``str``:
            ``"poi"`` if the modifier references a POI, ``"lumi"`` if it is a
            luminosity normalisation factor, or ``"nuisance"`` otherwise.
        """
        if self._modifier_param_names(modifier) & pois:
            return "poi"
        if self._is_lumi_modifier(modifier):
            return "lumi"
        return "nuisance"

    def _get_sample(self, dist_name: str, sample_name: str) -> Optional[Dict]:
        """
        Return the raw sample dict for a given distribution and sample name.

        Args:
            dist_name (``str``): Distribution name.
            sample_name (``str``): Sample name.

        Returns:
            ``Optional[Dict]``:
            The raw sample dictionary, or ``None`` when either the distribution
            or the sample is not found.
        """
        dist = self._dist_map.get(dist_name)
        if dist is None:
            return None
        for s in dist.get("samples", []):
            if s.get("name") == sample_name:
                return s
        return None

    def _resolve_signal_targets(
        self, dist_name: Optional[str], sample_name: Optional[str]
    ) -> Dict[str, set]:
        """
        Resolve the ``{dist: {sample, ...}}`` set targeted by a strip call.

        With no filters, every detected signal sample is targeted.  An explicit
        *sample_name* is honoured even if detection did not flag it as signal
        (the caller is being explicit), but the distribution must exist.

        Args:
            dist_name (``Optional[str]``): Distribution to restrict to, or
                ``None`` for all distributions.
            sample_name (``Optional[str]``): Sample to restrict to, or ``None``
                for all signal samples within each targeted distribution.

        Raises:
            ``ValueError``: If *dist_name* is not a ``histfactory_dist``, or
                *sample_name* is given together with an explicit *dist_name* but
                does not exist in that distribution.

        Returns:
            ``Dict[str, set]``:
            Mapping ``{dist_name: {sample_name, ...}}`` of targeted samples.
        """
        signal = self.signal_samples
        targets: Dict[str, set] = {}

        if dist_name is not None and dist_name not in self._dist_map:
            raise ValueError(
                f"'{dist_name}' is not a histfactory_dist. "
                f"Available: {self.distributions}"
            )

        dist_names = [dist_name] if dist_name is not None else list(self._dist_map)
        for d in dist_names:
            if sample_name is not None:
                if self._get_sample(d, sample_name) is None:
                    if dist_name is not None:
                        raise ValueError(
                            f"Sample '{sample_name}' not found in '{d}'. "
                            f"Available: {self.samples.get(d, [])}"
                        )
                    continue
                targets.setdefault(d, set()).add(sample_name)
            else:
                for sname in signal.get(d, []):
                    targets.setdefault(d, set()).add(sname)
        return targets

    def _cleanup_orphan_parameters(self, data: Dict) -> None:
        """
        Remove parameters orphaned by stripping from the patched workspace.

        A candidate is a parameter that was referenced by a removed signal
        sample or stripped modifier and is no longer referenced anywhere in the
        post-strip workspace.  Such candidates are dropped from every
        ``domains`` axis list, every ``parameter_points`` parameter list, and
        every analysis ``parameters_of_interest`` list.

        Args:
            data (``Dict``): Deep-copied workspace dictionary being patched
                in-place by :attr:`patch`.
        """
        # Parameters touched (referenced) by the removed samples / modifiers,
        # computed from the original workspace.
        touched: set = set()
        for dist_name, snames in self._samples_to_remove.items():
            dist = self._dist_map.get(dist_name, {})
            for s in dist.get("samples", []):
                if s.get("name") in snames:
                    for m in s.get("modifiers", []):
                        touched |= self._modifier_param_names(m)
        for dist_name, smap in self._modifier_strip.items():
            dist = self._dist_map.get(dist_name, {})
            for s in dist.get("samples", []):
                rm = smap.get(s.get("name"))
                if not rm:
                    continue
                for m in s.get("modifiers", []):
                    if m.get("name") in rm:
                        touched |= self._modifier_param_names(m)
        if not touched:
            return

        # Parameters still referenced after stripping: histfactory modifier
        # parameters plus any string reference inside non-histfactory
        # distributions, functions and auxiliary likelihood terms.
        referenced: set = set()
        for dist in data.get("distributions", []):
            if dist.get("type") == "histfactory_dist":
                for s in dist.get("samples", []):
                    for m in s.get("modifiers", []):
                        referenced |= self._modifier_param_names(m)
            else:
                referenced |= self._leaf_strings(dist)
        for fn in data.get("functions", []):
            referenced |= self._leaf_strings(fn)
        for lh in data.get("likelihoods", []):
            referenced.update(
                a for a in lh.get("aux_distributions", []) if isinstance(a, str)
            )

        orphans = touched - referenced
        if not orphans:
            return

        for domain in data.get("domains", []):
            axes = domain.get("axes")
            if isinstance(axes, list):
                domain["axes"] = [
                    ax for ax in axes if ax.get("name") not in orphans
                ]
        for pp in data.get("parameter_points", []):
            params = pp.get("parameters")
            if isinstance(params, list):
                pp["parameters"] = [
                    p for p in params if p.get("name") not in orphans
                ]
        for a in data.get("analyses", []):
            pois = a.get("parameters_of_interest")
            if isinstance(pois, list):
                a["parameters_of_interest"] = [p for p in pois if p not in orphans]

        log.debug("Pruned orphaned parameters: %s", sorted(orphans))

    @classmethod
    def _leaf_strings(cls, obj) -> set:
        """
        Recursively collect every string *value* contained in *obj*.

        Used to identify all parameter names potentially referenced by
        non-histfactory distributions and workspace functions.

        Args:
            obj: Arbitrary Python object (dict, list, str, or scalar).

        Returns:
            ``set``:
            Set of all string values found anywhere inside *obj*.
        """
        found: set = set()
        if isinstance(obj, str):
            found.add(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                found |= cls._leaf_strings(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                found |= cls._leaf_strings(v)
        return found

    def _default_poi(self) -> str:
        """
        Return the first POI of the first analysis, or raise.

        Raises:
            ``ValueError``: If no analyses exist in the workspace, or the first
                analysis has no ``parameters_of_interest``.

        Returns:
            ``str``:
            Name of the first parameter of interest.
        """
        if not self._analyses_list:
            raise ValueError(
                "No analyses in workspace; please specify poi_name explicitly."
            )
        pois = self._analyses_list[0].get("parameters_of_interest", [])
        if not pois:
            raise ValueError(
                "No parameters_of_interest in the first analysis; "
                "please specify poi_name explicitly."
            )
        return pois[0]
