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
    An HS3 workspace interpreter for bookkeeping and signal injection.

    Provides a user-friendly interface to inspect HS3 JSON workspaces and
    inject signal samples into ``histfactory_dist`` distributions, analogous
    to the pyhf :class:`WorkspaceInterpreter` but for the HS3 format.

    The class works purely from the raw JSON dictionary — no ``pyhs3``
    compilation is required for inspection.  The :attr:`patch` property
    returns a modified copy of the workspace ready to be passed to
    :class:`~spey_hs3.HS3Interface`.

    Parameters
    ----------
    workspace:
        HS3 JSON workspace as a Python dictionary (background-only model).

    Examples
    --------
    >>> import json, spey
    >>> with open("workspace.json") as f:
    ...     ws_dict = json.load(f)
    >>> interp = WorkspaceInterpreter(ws_dict)
    >>> interp.summary()
    >>> interp.inject_signal("model_SR", "Signal", [3.0, 5.0, 2.0])
    >>> stat_model = spey.get_backend("hs3")(
    ...     hs3_dict=interp.patch,
    ...     analysis_name="myAnalysis",
    ... )
    """

    __slots__ = [
        "_workspace",
        "_signal_dict",  # {dist_name: {sample_name: {"contents", "errors", "poi_name"}}}
        "_to_remove",  # [dist_name, ...]
        "_dist_map",  # {dist_name: dist_dict}  – histfactory_dist only
        "_data_map",  # {data_name: data_dict}
        "_lh_map",  # {lh_name: lh_dict}
        "_analyses_list",
    ]

    def __init__(self, workspace: Dict):
        self._workspace = workspace
        self._signal_dict: Dict[str, Dict[str, Dict]] = {}
        self._to_remove: List[str] = []

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
        return (
            f"WorkspaceInterpreter("
            f"analyses={len(self._analyses_list)}, "
            f"histfactory_distributions={len(self._dist_map)}, "
            f"injected_signals={sum(len(v) for v in self._signal_dict.values())})"
        )

    # ------------------------------------------------------------------
    # Read-only inspection properties
    # ------------------------------------------------------------------

    @property
    def distributions(self) -> List[str]:
        """Names of all ``histfactory_dist`` distributions in the workspace."""
        return list(self._dist_map.keys())

    @property
    def all_distribution_types(self) -> Dict[str, str]:
        """All distributions (including constraint PDFs) with their types."""
        return {
            d["name"]: d.get("type", "unknown")
            for d in self._workspace.get("distributions", [])
        }

    @property
    def analyses(self) -> List[str]:
        """Names of all analyses in the workspace."""
        return [a["name"] for a in self._analyses_list]

    @property
    def poi_names(self) -> Dict[str, List[str]]:
        """Parameters of interest per analysis: ``{analysis_name: [poi, ...]}}``."""
        return {
            a["name"]: a.get("parameters_of_interest", []) for a in self._analyses_list
        }

    @property
    def bin_map(self) -> Dict[str, int]:
        """Number of bins per ``histfactory_dist`` distribution."""
        result: Dict[str, int] = {}
        for dist_name, dist in self._dist_map.items():
            # Prefer the axes field on the distribution itself
            axes = dist.get("axes", [])
            if axes:
                result[dist_name] = axes[0].get("nbins", len(axes))
                continue
            # Fall back: infer from first sample contents
            samples = dist.get("samples", [])
            if samples:
                data = samples[0].get("data", {})
                if isinstance(data, dict):
                    result[dist_name] = len(data.get("contents", []))
                elif isinstance(data, list):
                    result[dist_name] = len(data)
                else:
                    result[dist_name] = 0
            else:
                result[dist_name] = 0
        return result

    @property
    def samples(self) -> Dict[str, List[str]]:
        """Sample names per distribution: ``{dist_name: [sample_name, ...]}}``."""
        return {
            dist_name: [s["name"] for s in dist.get("samples", [])]
            for dist_name, dist in self._dist_map.items()
        }

    @property
    def modifier_types(self) -> Dict[str, Dict[str, List[str]]]:
        """Modifier types per sample per distribution."""
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

        Sums the ``contents`` of all existing (non-injected) samples.
        Distributions marked for removal are excluded.
        """
        result: Dict[str, List[float]] = {}
        injected_samples = {
            dist_name: set(samples.keys())
            for dist_name, samples in self._signal_dict.items()
        }

        for dist_name, dist in self._dist_map.items():
            if dist_name in self._to_remove:
                continue
            n_bins = self.bin_map.get(dist_name, 0)
            totals = [0.0] * n_bins
            skip = injected_samples.get(dist_name, set())

            for sample in dist.get("samples", []):
                if sample["name"] in skip:
                    continue
                data = sample.get("data", {})
                contents = data.get("contents", []) if isinstance(data, dict) else data
                for i, val in enumerate(contents):
                    if i < n_bins:
                        totals[i] += float(val)
            result[dist_name] = totals
        return result

    @property
    def observed_data(self) -> Dict[str, List[float]]:
        """
        Observed data per distribution, resolved from the workspace data section.

        The mapping is established via the likelihoods (which pair each
        distribution with its data object).  Only the first matching entry per
        distribution is used.
        """
        result: Dict[str, List[float]] = {}
        for lh in self._lh_map.values():
            for dist_name, data_name in zip(
                lh.get("distributions", []), lh.get("data", [])
            ):
                if dist_name in self._dist_map and dist_name not in result:
                    data_entry = self._data_map.get(data_name)
                    if data_entry is not None:
                        result[dist_name] = [
                            float(v) for v in data_entry.get("contents", [])
                        ]
        return result

    @property
    def parameters(self) -> Dict[str, Dict]:
        """
        Parameter metadata from the first parameter_point.

        Returns ``{param_name: {"value": float, "const": bool}}``.
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
        """Currently injected signal yields: ``{dist_name: {sample_name: [yields]}}``."""
        return {
            dist_name: {sname: spec["contents"] for sname, spec in samples.items()}
            for dist_name, samples in self._signal_dict.items()
        }

    @property
    def remove_list(self) -> List[str]:
        """Distributions currently marked for removal."""
        return list(self._to_remove)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_analysis(self, name: Optional[str] = None) -> Dict:
        """
        Retrieve an analysis dictionary by name.

        Parameters
        ----------
        name:
            Analysis name.  If ``None``, the first analysis is returned.

        Raises
        ------
        ValueError
            If no analyses exist or *name* is not found.
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

        Parameters
        ----------
        analysis_name:
            Filter by analysis (resolves the associated likelihood automatically).
        likelihood_name:
            Filter by likelihood name directly.

        Returns
        -------
        List[str]
            Matching distribution names.
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
        Guess region type (``"SR"``, ``"CR"``, ``"VR"``) from the distribution name.

        Returns ``"__unknown__"`` when no pattern is recognised.
        """
        upper = dist_name.upper()
        for tp in ("SR", "CR", "VR"):
            if tp in upper:
                return tp
        return "__unknown__"

    def guess_signal_regions(self) -> List[str]:
        """Return distribution names that appear to be signal regions."""
        return [n for n in self.distributions if self.guess_region_type(n) == "SR"]

    def guess_control_regions(self) -> List[str]:
        """Return distribution names that appear to be control or validation regions."""
        return [
            n for n in self.distributions if self.guess_region_type(n) in ("CR", "VR")
        ]

    def get_sample_yields(
        self, dist_name: str, sample_name: str
    ) -> Tuple[List[float], List[float]]:
        """
        Return the ``(contents, errors)`` for a specific sample in a distribution.

        Parameters
        ----------
        dist_name:
            Name of the ``histfactory_dist`` distribution.
        sample_name:
            Name of the sample within that distribution.

        Returns
        -------
        Tuple[List[float], List[float]]
            ``(contents, errors)`` per bin.

        Raises
        ------
        ValueError
            If the distribution or sample is not found.
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

        The injected sample will receive a ``normfactor`` modifier tied to
        *poi_name* so that the signal strength :math:`\\mu` scales the sample
        when used with :class:`~spey_hs3.HS3Interface`.

        Parameters
        ----------
        dist_name:
            Target distribution name (must be a ``histfactory_dist``).
        sample_name:
            Name for the new signal sample.  Must not already exist.
        yields:
            Per-bin signal yields (at :math:`\\mu = 1`).
        errors:
            Per-bin MC statistical errors.  Defaults to zeros.
        poi_name:
            Name of the parameter of interest to attach.  If ``None``, the
            first POI of the first analysis is used.

        Raises
        ------
        ValueError
            If *dist_name* does not exist, the bin count does not match,
            or *sample_name* is already present.
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

        Parameters
        ----------
        signal_map:
            Nested mapping ``{dist_name: {sample_name: spec}}``.

            *spec* can be either a plain list of per-bin yields or a dict
            with keys ``"contents"`` (required) and ``"errors"`` (optional)::

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

        poi_name:
            Parameter of interest to attach to every injected sample.
            If ``None``, the first POI of the first analysis is used.
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
        and all likelihood references in :attr:`patch`.

        Parameters
        ----------
        dist_name:
            Name of the ``histfactory_dist`` to remove.
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
        """Clear all injected signal samples and the remove list."""
        self._signal_dict = {}
        self._to_remove = []

    # ------------------------------------------------------------------
    # Workspace patching
    # ------------------------------------------------------------------

    @property
    def patch(self) -> Dict:
        """
        Deep copy of the workspace with injected signals applied.

        The returned dict can be passed directly to
        :class:`~spey_hs3.HS3Interface` as ``hs3_dict``, or saved to a new
        JSON file.

        Raises
        ------
        ValueError
            If no signal has been injected and no distribution is being removed.

        Returns
        -------
        Dict
            Patched HS3 workspace dictionary.
        """
        if not self._signal_dict and not self._to_remove:
            raise ValueError(
                "Nothing to patch: use inject_signal() or remove_distribution() first."
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

        return data

    def save_patch(self, path: str) -> None:
        """
        Save the patched workspace to a JSON file.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``"patched_workspace.json"``).
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

        Parameters
        ----------
        analysis_name:
            Restrict the distribution listing to one analysis.
        show_samples:
            Also list each sample name per distribution.
        show_parameters:
            Print free / const parameter counts.
        max_dists:
            Maximum number of distributions to print per analysis (default 50).
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
                removed_str = "  [REMOVED]" if dname in self._to_remove else ""
                obs_str = f"  obs={n_obs}" if n_obs else ""
                print(
                    f"    {i+1:4d}. {tag} {dname}"
                    f"  ({n} bin{'s' if n != 1 else ''}){obs_str}{inj_str}{removed_str}"
                )
                if show_samples:
                    for sname in self.samples.get(dname, []):
                        injected_flag = (
                            " [injected]"
                            if sname in (self._signal_dict.get(dname) or {})
                            else ""
                        )
                        print(f"              sample: {sname}{injected_flag}")

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

        if self._to_remove:
            print(f"Distributions to remove ({len(self._to_remove)}):")
            for dname in self._to_remove:
                print(f"  {dname}")
            print()

        print(sep)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_poi(self) -> str:
        """Return the first POI of the first analysis, or raise."""
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
