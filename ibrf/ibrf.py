# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 18:27:46 2025
"""

__version__ = "0.1.0"
# -*- coding: utf-8 -*-
"""
Improved Balanced Random Forest (iBRF)
Per-tree pipeline: NC -> partial SMOTE (balance_split) -> RUS
Author: Asif Newaz 
"""

import numbers
from warnings import warn
from copy import deepcopy

import numpy as np
from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE
from scipy.sparse import issparse

from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.ensemble._forest import _get_n_samples_bootstrap
from sklearn.ensemble._forest import _parallel_build_trees
from sklearn.exceptions import DataConversionWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import validate_data  # sklearn >=1.6 public API

from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule
from imblearn.utils import check_target_type
from sklearn.utils import check_X_y


# -----------------------------
# Utilities
# -----------------------------
def _class_counts(y):
    """Return dict {label: count} preserving label dtype."""
    uniq, cnt = np.unique(y, return_counts=True)
    return {label: int(c) for label, c in zip(uniq, cnt)}

def _majority_label(counts):
    """Return the label with the maximum count (ties resolved by first occurrence)."""
    return max(counts.items(), key=lambda kv: kv[1])[0]


# -----------------------------
# Composite sampler with tracking and balance split
# -----------------------------
class NCRUSSMOTE(BaseSampler):
    """
    Composite sampler: NC -> partial SMOTE -> RUS

    - After NC, let n_M be the size of the (single) majority class and n_c the size
      of any non-majority class c. We set SMOTE targets as:

          n_c_target = n_c + balance_split * (n_M - n_c)

      rounded to int, for each non-majority class. Majority class is untouched by SMOTE.

    - Then RUS reduces ONLY the majority class down to:
          max_c n_c_target
      while leaving other classes unchanged.

    This lets SMOTE handle only a fraction of the gap, with the remainder handled by RUS.
    """

    _sampling_type = "over-sampling"
    _parameter_constraints: dict = {}  # for sklearn >=1.5 compatibility

    def __init__(
        self,
        *,
        balance_split=0.65,           # fraction of gap filled by SMOTE (0..1)
        random_state=None,
        n_jobs=-1,
        smote=None,
        rus=None,
        nc=None,
        verbose=False,
    ):
        super().__init__()
        if not (0.0 <= balance_split <= 1.0):
            raise ValueError("balance_split must be in [0, 1].")
        self.balance_split = float(balance_split)

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.smote = smote
        self.rus = rus
        self.nc = nc
        self.verbose = verbose

        # per-instance stats (for this sampler / tree)
        self.stats_ = {"nc_removed": 0, "smote_generated": 0, "rus_removed": 0}

    def _validate_estimators(self):
        # NC (supports n_jobs)
        if self.nc is not None:
            if not isinstance(self.nc, NeighbourhoodCleaningRule):
                raise ValueError(f"nc must be NeighbourhoodCleaningRule, got {type(self.nc)}.")
            self.nc_ = clone(self.nc)
        else:
            self.nc_ = NeighbourhoodCleaningRule(sampling_strategy="auto", n_jobs=self.n_jobs)

        # SMOTE (we'll pass a dict strategy per-batch; keep an instance prototype)
        if self.smote is not None:
            if not isinstance(self.smote, SMOTE):
                raise ValueError(f"smote must be SMOTE, got {type(self.smote)}.")
            self.smote_proto_ = clone(self.smote)
        else:
            # Use a prototype; we will clone with dict strategy at call time
            try:
                self.smote_proto_ = SMOTE(random_state=self.random_state, n_jobs=self.n_jobs)
            except TypeError:
                self.smote_proto_ = SMOTE(random_state=self.random_state)

        # RUS (we will pass a dict strategy per-batch)
        if self.rus is not None:
            if not isinstance(self.rus, RandomUnderSampler):
                raise ValueError(f"rus must be RandomUnderSampler, got {type(self.rus)}.")
            self.rus_proto_ = clone(self.rus)
        else:
            self.rus_proto_ = RandomUnderSampler()

    def _fit_resample(self, X, y):
        self._validate_estimators()
        y = check_target_type(y)
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"])

        n0 = len(y)

        # 1) Neighbourhood Cleaning
        X_nc, y_nc = self.nc_.fit_resample(X, y)
        n1 = len(y_nc)
        self.stats_["nc_removed"] = int(n0 - n1)

        # 2) Partial SMOTE toward balance by balance_split
        counts_nc = _class_counts(y_nc)
        maj_label = _majority_label(counts_nc)
        nM = counts_nc[maj_label]

        # Build SMOTE target dict for NON-majority classes only
        smote_target = {}
        smote_generated_total = 0
        for lbl, ncnt in counts_nc.items():
            if lbl == maj_label:
                continue
            target = int(round(ncnt + self.balance_split * (nM - ncnt)))
            # ensure monotonic (at least current size)
            target = max(target, ncnt)
            smote_target[lbl] = target
            smote_generated_total += (target - ncnt)

        if smote_target:
            # Clone with per-batch dict strategy
            try:
                smote_ = clone(self.smote_proto_).set_params(sampling_strategy=smote_target)
            except (TypeError, ValueError):
                # older imblearn may require passing in ctor
                try:
                    smote_ = SMOTE(sampling_strategy=smote_target, random_state=self.random_state)
                except TypeError:
                    smote_ = SMOTE(sampling_strategy=smote_target, random_state=self.random_state)
            X_sm, y_sm = smote_.fit_resample(X_nc, y_nc)
        else:
            # Degenerate (everything already equal)
            X_sm, y_sm = X_nc, y_nc

        n2 = len(y_sm)
        # cross-check/record generated count (prefer computed target sum for stability)
        self.stats_["smote_generated"] = int(smote_generated_total)

        # 3) RUS to finish the balance: reduce majority to max(minority targets)
        counts_sm = _class_counts(y_sm)
        maj_label_after = _majority_label(counts_sm)  # usually same, but recompute
        max_minority_after = max([cnt for lbl, cnt in counts_sm.items() if lbl != maj_label_after], default=counts_sm[maj_label_after])

        # Build RUS dict: keep all non-majority at their current counts; reduce majority to max minority
        rus_target = {}
        for lbl, cnt in counts_sm.items():
            if lbl == maj_label_after:
                rus_target[lbl] = int(max_minority_after)
            else:
                rus_target[lbl] = int(cnt)

        rus_ = clone(self.rus_proto_).set_params(sampling_strategy=rus_target)
        X_res, y_res = rus_.fit_resample(X_sm, y_sm)
        n3 = len(y_res)

        # Only majority can be reduced; compute removed count as delta in majority
        rus_removed = max(0, counts_sm[maj_label_after] - rus_target[maj_label_after])
        self.stats_["rus_removed"] = int(rus_removed)

        if self.verbose:
            print(
                f"[NCRUSSMOTE] NC removed: {self.stats_['nc_removed']}, "
                f"SMOTE generated: {self.stats_['smote_generated']}, "
                f"RUS removed: {self.stats_['rus_removed']}"
            )

        return X_res, y_res

    def get_sampling_report(self):
        """Return a dict with counts of removed/generated samples for this sampler instance."""
        return dict(self.stats_)


# ---------------------------------------
# Internal helper for parallel tree build
# ---------------------------------------
def _local_parallel_build_trees(
    sampler,
    tree,
    forest,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
):
    # Resample (NC -> partial SMOTE -> RUS) for this tree
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # Ensure bootstrap doesn't exceed resampled size
    if _get_n_samples_bootstrap is not None:
        n_samples_bootstrap = min(n_samples_bootstrap, X_resampled.shape[0])

    # Build tree on the resampled set
    tree = _parallel_build_trees(
        tree,
        forest,
        X_resampled,
        y_resampled,
        sample_weight=None,  # synthetic samples break direct mapping of original weights
        tree_idx=tree_idx,
        n_trees=n_trees,
        verbose=verbose,
        class_weight=class_weight,
        n_samples_bootstrap=n_samples_bootstrap,
    )
    return sampler, tree


# --------------------------------------
# The iBRF classifier (drop-in estimator)
# --------------------------------------
class ImprovedBalancedRandomForestClassifier(RandomForestClassifier):
    """
    iBRF: For each tree, apply NC -> partial SMOTE (balance_split) -> RUS on that tree's
    effective training subset, then fit the tree. OOB is not supported.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",  # 'auto' deprecated; 'sqrt' is the modern default for classification
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,  # forbidden (SMOTE makes OOB undefined)
        balance_split=0.65,   # <-- key control knob (0..1)
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        smote=None,
        rus=None,
        nc=None,
        sampler_verbose=False,
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        self.balance_split = float(balance_split)
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.smote = smote
        self.rus = rus
        self.nc = nc
        self.sampler_verbose = sampler_verbose

        # storage
        self.base_sampler_ = None
        self.samplers_ = []   # one per fitted tree

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        """Validate n_estimators and set base estimator + base sampler prototypes."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError(f"n_estimators must be an integer, got {type(self.n_estimators)}.")
        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators must be greater than zero, got {self.n_estimators}.")

        # Base tree
        be = getattr(self, "base_estimator", None)
        self.base_estimator_ = clone(be) if be is not None else clone(default)

        # Prototype sampler used to clone per-tree samplers
        self.base_sampler_ = NCRUSSMOTE(
            balance_split=self.balance_split,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            smote=self.smote,
            rus=self.rus,
            nc=self.nc,
            verbose=self.sampler_verbose,
        )

    def _make_sampler_estimator(self, random_state=None):
        """Instantiate a fresh (estimator, sampler) pair for a tree."""
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})

        sampler = clone(self.base_sampler_)
        if random_state is not None:
            _set_random_states(estimator, random_state)
            _set_random_states(sampler, random_state)
        return estimator, sampler

    def fit(self, X, y, sample_weight=None):
        """Build the forest of trees with per-tree NC -> partial SMOTE -> RUS resampling."""
        if self.oob_score:
            raise ValueError(
                "oob_score=True is not supported in iBRF because SMOTE introduces "
                "synthetic samples that invalidate OOB accounting."
            )

        # sklearn >=1.6: use public validate_data; fall back if needed
        try:
            X, y = validate_data(self, X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE)
        except TypeError:
            X, y = self._validate_data(X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        self._n_features = X.shape[1]

        if issparse(X):
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was expected. "
                "Please reshape y to (n_samples,), e.g., using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        y_encoded, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y_encoded = np.ascontiguousarray(y_encoded, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Bootstrap sample size per tree
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=self.max_samples
        )

        self._validate_estimator()
        rng = check_random_state(self.random_state)

        # Reset containers (unless warm_start and adding more trees)
        if not self.warm_start or not hasattr(self, "estimators_"):
            self.estimators_ = []
            self.samplers_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)
        if n_more_estimators < 0:
            raise ValueError(
                f"n_estimators={self.n_estimators} must be >= "
                f"len(estimators_)={len(self.estimators_)} when warm_start=True"
            )
        elif n_more_estimators == 0:
            warn("Warm-start called without increasing n_estimators; no new trees fitted.")
            return self

        if self.warm_start and len(self.estimators_) > 0:
            rng.randint(np.iinfo(np.int32).max, size=len(self.estimators_))

        # Create (tree, sampler) pairs
        trees = []
        samplers = []
        for _ in range(n_more_estimators):
            t, s = self._make_sampler_estimator(random_state=rng)
            trees.append(t)
            samplers.append(s)

        # Fit trees in parallel
        samplers_trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_local_parallel_build_trees)(
                s, t, self, X, y_encoded, sample_weight,
                i, len(trees), verbose=self.verbose,
                class_weight=self.class_weight,
                n_samples_bootstrap=n_samples_bootstrap,
            )
            for i, (s, t) in enumerate(zip(samplers, trees))
        )
        samplers, trees = zip(*samplers_trees)

        # Collect fitted components
        self.estimators_.extend(trees)
        self.samplers_.extend(samplers)

        # Decapsulate classes_ for single-output
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    # Convenience: aggregate per-forest sampling stats
    def get_forest_sampling_report(self, reduce="sum"):
        """
        Aggregate NC/SMOTE/RUS counts across all fitted trees.
        reduce='sum' | 'mean'
        """
        if not hasattr(self, "samplers_") or len(self.samplers_) == 0:
            return {"nc_removed": 0, "smote_generated": 0, "rus_removed": 0}

        stats = np.array(
            [
                [
                    s.stats_.get("nc_removed", 0),
                    s.stats_.get("smote_generated", 0),
                    s.stats_.get("rus_removed", 0),
                ]
                for s in self.samplers_
            ],
            dtype=float,
        )
        if reduce == "mean":
            agg = stats.mean(axis=0)
        else:
            agg = stats.sum(axis=0)

        return {
            "nc_removed": float(agg[0]),
            "smote_generated": float(agg[1]),
            "rus_removed": float(agg[2]),
        }

    @property
    def n_features_(self):
        """Number of features when fitting the estimator (back-compat shim)."""
        return getattr(self, "n_features_in_", getattr(self, "_n_features", None))

    def _more_tags(self):
        return {"multioutput": False, "multilabel": False}


# Alias
iBRF = ImprovedBalancedRandomForestClassifier


