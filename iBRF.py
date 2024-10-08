# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:47:53 2024

@author: asifn
"""

"""Proposed Algorithm: iBRF"""

# The code is written using the original implementation of the BRF classier from the imblearn library.

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
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.exceptions import DataConversionWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing
#from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import _check_sample_weight

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import EditedNearestNeighbours

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.base import BaseSampler

from imblearn.utils import Substitution
from imblearn.utils._docstring import _n_jobs_docstring
from imblearn.utils._docstring import _random_state_docstring
from imblearn.utils._validation import check_sampling_strategy
from imblearn.utils._validation import _deprecate_positional_args

#from smote_nc import SMOTENC
#from smote_rus import SMOTERUS

MAX_INT = np.iinfo(np.int32).max

from sklearn.utils import check_X_y

from imblearn.base import BaseSampler
from imblearn.base import BaseCleaningSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import RandomUnderSampler

from imblearn.utils import check_target_type
from imblearn.utils import Substitution
from imblearn.utils._docstring import _n_jobs_docstring
from imblearn.utils._docstring import _random_state_docstring
from imblearn.utils._validation import _deprecate_positional_args


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SMOTENCRUS(BaseSampler):
    
    """Over-sampling using SMOTE, undersampling using RUS and cleaning using NC.
    
    Parameters
    ----------
    {sampling_strategy}
    {random_state}
    
    smote : sampler object, default=None
        The :class:`~imblearn.over_sampling.SMOTE` object to use. If not given,
        a :class:`~imblearn.over_sampling.SMOTE` object with default parameters
        will be given.
    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.
    smote_ : sampler object
        The validated :class:`~imblearn.over_sampling.SMOTE` instance.
    nc_ : sampler object
        The validated :class:`~imblearn.under_sampling.NeighbourhoodCleaningRule`
        instance.
    rus_ : sampler object
        The validated :class:`~imblearn.under_sampling.RandomUnderSampler`
        instance.
    n_features_in_ : int
        Number of features in the input dataset.
        
    """

    _sampling_type = "over-sampling"

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        smote=None,
        rus= None,
        nc= None,
        n_jobs=-1,
    ):
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = smote
        self.rus= rus
        self.nc= nc
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        
        if self.smote is not None:
            if isinstance(self.smote, SMOTE):
                self.smote_ = clone(self.smote)
            else:
                raise ValueError(
                    f"smote needs to be a SMOTE object."
                    f"Got {type(self.smote)} instead."
                )
        # Otherwise create a default SMOTE
        else:
            self.smote_ = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

            
        if self.rus is not None:
            if isinstance(self.rus, RandomUnderSampler):
                self.rus_ = clone(self.rus)
            else:
                raise ValueError(
                    f"enn needs to be an RUS."
                    f" Got {type(self.rus)} instead."
                )
        # Otherwise create a default RUS
        else:
            self.rus_ = RandomUnderSampler(
                sampling_strategy="auto", n_jobs=self.n_jobs
            )
            
            
        if self.nc is not None:
            if isinstance(self.nc, NeighbourhoodCleaningRule):
                self.nc_ = clone(self.nc)
            else:
                raise ValueError(
                    f"enn needs to be an NeighbourhoodCleaningRule."
                    f" Got {type(self.rus)} instead."
                )
        # Otherwise create a default NC
        else:
            self.nc_ = NeighbourhoodCleaningRule(
                sampling_strategy="auto", n_jobs=self.n_jobs
            )
        

    def _fit_resample(self, X, y):
        self._validate_estimator()
        y = check_target_type(y)
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"])
        self.sampling_strategy_ = self.sampling_strategy

        X_res, y_res = self.nc_.fit_resample(X, y)
        X_res, y_res = self.rus_.fit_resample(X_res, y_res)
                                              
        return self.smote_.fit_resample(X_res, y_res)
    
    # Creating the forest

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
    # resample before to fit the tree
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    if sample_weight is not None:
        sample_weight = _safe_indexing(sample_weight, sampler.sample_indices_)
    if _get_n_samples_bootstrap is not None:
        n_samples_bootstrap = min(n_samples_bootstrap, X_resampled.shape[0])
    tree = _parallel_build_trees(
        tree,
        forest,
        X_resampled,
        y_resampled,
        sample_weight,
        tree_idx,
        n_trees,
        verbose=verbose,
        class_weight=class_weight,
        n_samples_bootstrap=n_samples_bootstrap,
    )
    return sampler, tree


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)


class ImprovedBalancedRandomForestClassifier(RandomForestClassifier):
    """An improved balanced random forest classifier.

    A balanced random forest randomly under-samples each boostrap sample to
    balance it. The iBRF approach uses a hybrid sampling approach to balance it.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {{"gini", "entropy"}}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node:

        - If int, then consider ``min_samples_leaf`` as the minimum number.
        - If float, then ``min_samples_leaf`` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {{"auto", "sqrt", "log2"}}, int, float, or None, \
            default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    {sampling_strategy}

    replacement : bool, default=False
        Whether or not to sample randomly with replacement or not.

    {n_jobs}

    {random_state}

    verbose : int, default=0
        Controls the verbosity of the tree building process.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    class_weight : dict, list of dicts, {{"balanced", "balanced_subsample"}}, \
            default=None
        Weights associated with classes in the form dictionary with the key
        being the class_label and the value the weight.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{{0: 1, 1: 1}}, {{0: 1, 1: 5}}, {{0: 1, 1: 1}}, {{0: 1, 1: 1}}]
        instead of [{{1:1}}, {{2:5}}, {{3:1}}, {{4:1}}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.

        .. versionadded:: 0.6
           Added in `scikit-learn` in 0.22

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
            - If None (default), then draw `X.shape[0]` samples.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples. Thus,
              `max_samples` should be in the interval `(0, 1)`.
        Be aware that the final number samples used will be the minimum between
        the number of samples given in `max_samples` and the number of samples
        obtained after resampling.

        .. versionadded:: 0.6
           Added in `scikit-learn` in 0.22

    Attributes
    ----------
    base_estimator_ : :class:`~sklearn.tree.DecisionTreeClassifier` instance
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of :class:`~sklearn.tree.DecisionTreeClassifier`
        The collection of fitted sub-estimators.

    base_sampler_ : :class:`~imblearn.under_sampling.RandomUnderSampler`
        The base sampler used to construct the subsequent list of samplers.

    samplers_ : list of :class:`~imblearn.under_sampling.RandomUnderSampler`
        The collection of fitted samplers.

    pipelines_ : list of Pipeline.
        The collection of fitted pipelines (samplers + trees).

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

        .. deprecated:: 1.0
           `n_features_` is deprecated in `scikit-learn` 1.0 and will be removed
           in version 1.2. Depending of the version of `scikit-learn` installed,
           you will get be warned or not.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.9

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances (the higher, the more important the feature).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    References
    ----------
    .. [1] Chen, Chao, Andy Liaw, and Leo Breiman. "Using random forest to
       learn imbalanced data." University of California, Berkeley 110 (2004):
       1-12.


    """

    @_deprecate_positional_args
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        sampling_strategy="auto",
        replacement=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        k_neighbors= 5,
        smote= None,
        rus= None,
        nc= None
        
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

        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors=k_neighbors
        self.n_jobs=n_jobs
        self.replacement= replacement
        self.smote= smote
        self.rus= rus
        self.nc= nc

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError(
                f"n_estimators must be an integer, " f"got {type(self.n_estimators)}."
            )

        if self.n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be greater than zero, " f"got {self.n_estimators}."
            )

        if self.base_estimator is not None:
            self.base_estimator_ = clone(self.base_estimator)
        else:
            self.base_estimator_ = clone(default)

        self.base_sampler_ = SMOTENCRUS(
            sampling_strategy= self._sampling_strategy,
            random_state= self.random_state,
            smote= self.smote,
            rus= self.rus,
            nc= self.nc,
            n_jobs= self.n_jobs
        )

    def _make_sampler_estimator(self, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})
        sampler = clone(self.base_sampler_)

        if random_state is not None:
            _set_random_states(estimator, random_state)
            _set_random_states(sampler, random_state)

        return estimator, sampler

    def fit(self, X, y, sample_weight=None):
        """Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            The fitted instance.
        """

        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = self._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        self._n_features = X.shape[1]

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y_encoded, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y_encoded = np.ascontiguousarray(y_encoded, dtype=DOUBLE)

        if isinstance(self.sampling_strategy, dict):
            self._sampling_strategy = {
                np.where(self.classes_[0] == key)[0][0]: value
                for key, value in check_sampling_strategy(
                    self.sampling_strategy,
                    y,
                    "under-sampling",
                ).items()
            }
        else:
            self._sampling_strategy = self.sampling_strategy

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.samplers_ = []
            self.pipelines_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = []
            samplers = []
            for _ in range(n_more_estimators):
                tree, sampler = self._make_sampler_estimator(random_state=random_state)
                trees.append(tree)
                samplers.append(sampler)

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, we respect any parallel_backend contexts set
            # at a higher level, since correctness does not rely on using
            # threads.
            samplers_trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
               # **_joblib_parallel_args(prefer="threads"),
            )(
                delayed(_local_parallel_build_trees)(
                    s,
                    t,
                    self,
                    X,
                    y_encoded,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, (s, t) in enumerate(zip(samplers, trees))
            )
            samplers, trees = zip(*samplers_trees)

            # Collect newly grown trees
            self.estimators_.extend(trees)
            self.samplers_.extend(samplers)

            # Create pipeline with the fitted samplers and trees
            self.pipelines_.extend(
                [
                    make_pipeline(deepcopy(s), deepcopy(t))
                    for s, t in zip(samplers, trees)
                ]
            )

        if self.oob_score:
            self._set_oob_score(X, y_encoded)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def _set_oob_score(self, X, y):
        """Compute out-of-bag score."""
        X = check_array(X, dtype=DTYPE, accept_sparse="csr")

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = [
            np.zeros((n_samples, n_classes_[k])) for k in range(self.n_outputs_)
        ]

        for sampler, estimator in zip(self.samplers_, self.estimators_):
            X_resample = X[sampler.sample_indices_]
            y_resample = y[sampler.sample_indices_]

            n_sample_subset = y_resample.shape[0]
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_sample_subset, self.max_samples
            )

            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_sample_subset, n_samples_bootstrap
            )
            p_estimator = estimator.predict_proba(
                X_resample[unsampled_indices, :], check_input=False
            )

            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                indices = sampler.sample_indices_[unsampled_indices]
                predictions[k][indices, :] += p_estimator[k]

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn(
                    "Some inputs do not have OOB scores. "
                    "This probably means too few trees were used "
                    "to compute any reliable oob estimates."
                )

            with np.errstate(invalid="ignore", divide="ignore"):
                # with the resampling, we are likely to have rows not included
                # for the OOB score leading to division by zero
                decision = predictions[k] / predictions[k].sum(axis=1)[:, np.newaxis]
            mask_scores = np.isnan(np.sum(decision, axis=1))
            oob_decision_function.append(decision)
            oob_score += np.mean(
                y[~mask_scores, k] == np.argmax(predictions[k][~mask_scores], axis=1),
                axis=0,
            )

        if self.n_outputs_ == 1:
            self.oob_decision_function_ = oob_decision_function[0]
        else:
            self.oob_decision_function_ = oob_decision_function

        self.oob_score_ = oob_score / self.n_outputs_

    @property
    def n_features_(self):
        """Number of features when fitting the estimator."""
        return getattr(self.n_features_in_, "n_features_", self._n_features)

    def _more_tags(self):
        return {
            "multioutput": False,
            "multilabel": False,
        }