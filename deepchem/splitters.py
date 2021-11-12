"""
Contains an abstract base class that supports chemically aware data splits.
"""
import inspect
import os
import random
import tempfile
import itertools
import logging
from typing import Any, Dict, List, Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#import deepchem as dc
from datasets import Dataset, DiskDataset

logger = logging.getLogger(__name__)


def randomize_arrays(array_list):
  # assumes that every array is of the same dimension
  num_rows = array_list[0].shape[0]
  perm = np.random.permutation(num_rows)
  permuted_arrays = []
  for array in array_list:
    permuted_arrays.append(array[perm])
  return permuted_arrays


class Splitter(object):
  """Splitters split up Datasets into pieces for training/validation/testing.
  In machine learning applications, it's often necessary to split up a dataset
  into training/validation/test sets. Or to k-fold split a dataset (that is,
  divide into k equal subsets) for cross-validation. The `Splitter` class is
  an abstract superclass for all splitters that captures the common API across
  splitter classes.
  Note that `Splitter` is an abstract superclass. You won't want to
  instantiate this class directly. Rather you will want to use a concrete
  subclass for your application.
  """

  def k_fold_split(self,
                   dataset: Dataset,
                   k: int,
                   directories: Optional[List[str]] = None,
                   **kwargs) -> List[Tuple[Dataset, Dataset]]:
    """
    Parameters
    ----------
    dataset: Dataset
      Dataset to do a k-fold split
    k: int
      Number of folds to split `dataset` into.
    directories: List[str], optional (default None)
      List of length 2*k filepaths to save the result disk-datasets.
    Returns
    -------
    List[Tuple[Dataset, Dataset]]
      List of length k tuples of (train, cv) where `train` and `cv` are both `Dataset`.
    """
    logger.info("Computing K-fold split")
    if directories is None:
      directories = [tempfile.mkdtemp() for _ in range(2 * k)]
    else:
      assert len(directories) == 2 * k
    cv_datasets = []
    train_ds_base = None
    train_datasets = []
    # rem_dataset is remaining portion of dataset
    if isinstance(dataset, DiskDataset):
      rem_dataset = dataset
    else:
      rem_dataset = DiskDataset.from_numpy(dataset.X, dataset.y, dataset.w,
                                           dataset.ids)

    for fold in range(k):
      # Note starts as 1/k since fold starts at 0. Ends at 1 since fold goes up
      # to k-1.
      frac_fold = 1. / (k - fold)
      train_dir, cv_dir = directories[2 * fold], directories[2 * fold + 1]
      fold_inds, rem_inds, _ = self.split(
          rem_dataset,
          frac_train=frac_fold,
          frac_valid=1 - frac_fold,
          frac_test=0,
          **kwargs)
      cv_dataset = rem_dataset.select(fold_inds, select_dir=cv_dir)
      cv_datasets.append(cv_dataset)
      # FIXME: Incompatible types in assignment (expression has type "Dataset", variable has type "DiskDataset")
      rem_dataset = rem_dataset.select(rem_inds)  # type: ignore

      train_ds_to_merge: Iterator[Dataset] = filter(
          None, [train_ds_base, rem_dataset])
      train_ds_to_merge = filter(lambda x: len(x) > 0, train_ds_to_merge)
      train_dataset = DiskDataset.merge(train_ds_to_merge, merge_dir=train_dir)
      train_datasets.append(train_dataset)

      update_train_base_merge: Iterator[Dataset] = filter(
          None, [train_ds_base, cv_dataset])
      train_ds_base = DiskDataset.merge(update_train_base_merge)
    return list(zip(train_datasets, cv_datasets))

  def train_valid_test_split(self,
                             dataset: Dataset,
                             train_dir: Optional[str] = None,
                             valid_dir: Optional[str] = None,
                             test_dir: Optional[str] = None,
                             frac_train: float = 0.8,
                             frac_valid: float = 0.1,
                             frac_test: float = 0.1,
                             seed: Optional[int] = None,
                             log_every_n: int = 1000,
                             **kwargs) -> Tuple[Dataset, Dataset, Dataset]:
    """ Splits self into train/validation/test sets.
    Returns Dataset objects for train, valid, test.
    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    train_dir: str, optional (default None)
      If specified, the directory in which the generated
      training dataset should be stored. This is only
      considered if `isinstance(dataset, dc.data.DiskDataset)`
    valid_dir: str, optional (default None)
      If specified, the directory in which the generated
      valid dataset should be stored. This is only
      considered if `isinstance(dataset, dc.data.DiskDataset)`
      is True.
    test_dir: str, optional (default None)
      If specified, the directory in which the generated
      test dataset should be stored. This is only
      considered if `isinstance(dataset, dc.data.DiskDataset)`
      is True.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default 1000)
      Controls the logger by dictating how often logger outputs
      will be produced.
    Returns
    -------
    Tuple[Dataset, Optional[Dataset], Dataset]
      A tuple of train, valid and test datasets as dc.data.Dataset objects.
    """
    logger.info("Computing train/valid/test indices")
    train_inds, valid_inds, test_inds = self.split(
        dataset,
        frac_train=frac_train,
        frac_test=frac_test,
        frac_valid=frac_valid,
        seed=seed,
        log_every_n=log_every_n)
    if train_dir is None:
      train_dir = tempfile.mkdtemp()
    if valid_dir is None:
      valid_dir = tempfile.mkdtemp()
    if test_dir is None:
      test_dir = tempfile.mkdtemp()
    train_dataset = dataset.select(train_inds, train_dir)
    valid_dataset = dataset.select(valid_inds, valid_dir)
    test_dataset = dataset.select(test_inds, test_dir)
    if isinstance(train_dataset, DiskDataset):
      train_dataset.memory_cache_size = 40 * (1 << 20)  # 40 MB

    return train_dataset, valid_dataset, test_dataset

  def train_test_split(self,
                       dataset: Dataset,
                       train_dir: Optional[str] = None,
                       test_dir: Optional[str] = None,
                       frac_train: float = 0.8,
                       seed: Optional[int] = None,
                       **kwargs) -> Tuple[Dataset, Dataset]:
    """Splits self into train/test sets.
    Returns Dataset objects for train/test.
    Parameters
    ----------
    dataset: data like object
      Dataset to be split.
    train_dir: str, optional (default None)
      If specified, the directory in which the generated
      training dataset should be stored. This is only
      considered if `isinstance(dataset, dc.data.DiskDataset)`
      is True.
    test_dir: str, optional (default None)
      If specified, the directory in which the generated
      test dataset should be stored. This is only
      considered if `isinstance(dataset, dc.data.DiskDataset)`
      is True.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    seed: int, optional (default None)
      Random seed to use.
    Returns
    -------
    Tuple[Dataset, Dataset]
      A tuple of train and test datasets as dc.data.Dataset objects.
    """
    valid_dir = tempfile.mkdtemp()
    train_dataset, _, test_dataset = self.train_valid_test_split(
        dataset,
        train_dir,
        valid_dir,
        test_dir,
        frac_train=frac_train,
        frac_test=1 - frac_train,
        frac_valid=0.,
        seed=seed,
        **kwargs)
    return train_dataset, test_dataset

  def split(self,
            dataset: Dataset,
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1,
            seed: Optional[int] = None,
            log_every_n: Optional[int] = None) -> Tuple:
    """Return indices for specified split
    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to be split.
    seed: int, optional (default None)
      Random seed to use.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    log_every_n: int, optional (default None)
      Controls the logger by dictating how often logger outputs
      will be produced.
    Returns
    -------
    Tuple
      A tuple `(train_inds, valid_inds, test_inds)` of the indices (integers) for
      the various splits.
    """
    raise NotImplementedError

  def __str__(self) -> str:
    """Convert self to str representation.
    Returns
    -------
    str
      The string represents the class.
    Examples
    --------
    >>> import deepchem as dc
    >>> str(dc.splits.RandomSplitter())
    'RandomSplitter'
    """
    args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
    args_names = [arg for arg in args_spec.args if arg != 'self']
    args_num = len(args_names)
    args_default_values = [None for _ in range(args_num)]
    if args_spec.defaults is not None:
      defaults = list(args_spec.defaults)
      args_default_values[-len(defaults):] = defaults

    override_args_info = ''
    for arg_name, default in zip(args_names, args_default_values):
      if arg_name in self.__dict__:
        arg_value = self.__dict__[arg_name]
        # validation
        # skip list
        if isinstance(arg_value, list):
          continue
        if isinstance(arg_value, str):
          # skip path string
          if "\\/." in arg_value or "/" in arg_value or '.' in arg_value:
            continue
        # main logic
        if default != arg_value:
          override_args_info += '_' + arg_name + '_' + str(arg_value)
    return self.__class__.__name__ + override_args_info

  def __repr__(self) -> str:
    """Convert self to repr representation.
    Returns
    -------
    str
      The string represents the class.
    Examples
    --------
    >>> import deepchem as dc
    >>> dc.splits.RandomSplitter()
    RandomSplitter[]
    """
    args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
    args_names = [arg for arg in args_spec.args if arg != 'self']
    args_info = ''
    for arg_name in args_names:
      value = self.__dict__[arg_name]
      # for str
      if isinstance(value, str):
        value = "'" + value + "'"
      # for list
      if isinstance(value, list):
        threshold = 10
        value = np.array2string(np.array(value), threshold=threshold)
      args_info += arg_name + '=' + str(value) + ', '
    return self.__class__.__name__ + '[' + args_info[:-2] + ']'


class ScaffoldSplitter(Splitter):
  """Class for doing data splits based on the scaffold of small molecules.
  Group  molecules  based on  the Bemis-Murcko scaffold representation, which identifies rings,
  linkers, frameworks (combinations between linkers and rings) and atomic properties  such as
  atom type, hibridization and bond order in a dataset of molecules. Then split the groups by
  the number of molecules in each group in decreasing order.
  It is necessary to add the smiles representation in the ids field during the
  DiskDataset creation.
  Examples
  ---------
  >>> import deepchem as dc
  >>> # creation of demo data set with some smiles strings
  ... data_test= ["CC(C)Cl" , "CCC(C)CO" ,  "CCCCCCCO" , "CCCCCCCC(=O)OC" , "c3ccc2nc1ccccc1cc2c3" , "Nc2cccc3nc1ccccc1cc23" , "C1CCCCCC1" ]
  >>> Xs = np.zeros(len(data_test))
  >>> Ys = np.ones(len(data_test))
  >>> # creation of a deepchem dataset with the smile codes in the ids field
  ... dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=Ys,w=np.zeros(len(data_test)),ids=data_test)
  >>> scaffoldsplitter = dc.splits.ScaffoldSplitter()
  >>> train,test = scaffoldsplitter.train_test_split(dataset)
  >>> train
  <DiskDataset X.shape: (5,), y.shape: (5,), w.shape: (5,), ids: ['CC(C)Cl' 'CCC(C)CO' 'CCCCCCCO' 'CCCCCCCC(=O)OC' 'C1CCCCCC1'], task_names: [0]>
  References
  ----------
  .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
     1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.
  Note
  ----
  This class requires RDKit to be installed.
  """

  def split(self,
            dataset: Dataset,
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1,
            seed: Optional[int] = None,
            log_every_n: Optional[int] = 1000
           ) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits internal compounds into train/validation/test by scaffold.
    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default 1000)
      Controls the logger by dictating how often logger outputs
      will be produced.
    Returns
    -------
    Tuple[List[int], List[int], List[int]]
      A tuple of train indices, valid indices, and test indices.
      Each indices is a list of integers.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffold_sets = self.generate_scaffolds(dataset)

    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    logger.info("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
          test_inds += scaffold_set
        else:
          valid_inds += scaffold_set
      else:
        train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

  def generate_scaffolds(self, dataset: Dataset,
                         log_every_n: int = 1000) -> List[List[int]]:
    """Returns all scaffolds from the dataset.
    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    log_every_n: int, optional (default 1000)
      Controls the logger by dictating how often logger outputs
      will be produced.
    Returns
    -------
    scaffold_sets: List[List[int]]
      List of indices of each scaffold in the dataset.
    """
    scaffolds = {}
    data_len = len(dataset)

    logger.info("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.ids):
      if ind % log_every_n == 0:
        logger.info("Generating scaffold %d/%d" % (ind, data_len))
      scaffold = _generate_scaffold(smiles)
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


class RandomStratifiedSplitter(Splitter):
  """RandomStratified Splitter class.
  For sparse multitask datasets, a standard split offers no guarantees
  that the splits will have any active compounds. This class tries to
  arrange that each split has a proportional number of the actives for each
  task. This is strictly guaranteed only for single-task datasets, but for
  sparse multitask datasets it usually manages to produces a fairly accurate
  division of the actives for each task.
  Note
  ----
  This splitter is primarily designed for boolean labeled data. It considers
  only whether a label is zero or non-zero. When labels can take on multiple
  non-zero values, it does not try to give each split a proportional fraction
  of the samples with each value.
  """

  def split(self,
            dataset: Dataset,
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1,
            seed: Optional[int] = None,
            log_every_n: Optional[int] = None) -> Tuple:
    """Return indices for specified split
    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to be split.
    seed: int, optional (default None)
      Random seed to use.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    log_every_n: int, optional (default None)
      Controls the logger by dictating how often logger outputs
      will be produced.
    Returns
    -------
    Tuple
      A tuple `(train_inds, valid_inds, test_inds)` of the indices (integers) for
      the various splits.
    """
    y_present = (dataset.y != 0) * (dataset.w != 0)
    if len(y_present.shape) == 1:
      y_present = np.expand_dims(y_present, 1)
    elif len(y_present.shape) > 2:
      raise ValueError(
          'RandomStratifiedSplitter cannot be applied when y has more than two dimensions'
      )
    if seed is not None:
      np.random.seed(seed)

    # Figure out how many positive samples we want for each task in each dataset.

    n_tasks = y_present.shape[1]
    indices_for_task = [
        np.random.permutation(np.nonzero(y_present[:, i])[0])
        for i in range(n_tasks)
    ]
    count_for_task = np.array([len(x) for x in indices_for_task])
    train_target = np.round(frac_train * count_for_task).astype(int)
    valid_target = np.round(frac_valid * count_for_task).astype(int)
    test_target = np.round(frac_test * count_for_task).astype(int)

    # Assign the positive samples to datasets.  Since a sample may be positive
    # on more than one task, we need to keep track of the effect of each added
    # sample on each task.  To try to keep everything balanced, we cycle through
    # tasks, assigning one positive sample for each one.

    train_counts = np.zeros(n_tasks, int)
    valid_counts = np.zeros(n_tasks, int)
    test_counts = np.zeros(n_tasks, int)
    set_target = [train_target, valid_target, test_target]
    set_counts = [train_counts, valid_counts, test_counts]
    set_inds: List[List[int]] = [[], [], []]
    assigned = set()
    max_count = np.max(count_for_task)
    for i in range(max_count):
      for task in range(n_tasks):
        indices = indices_for_task[task]
        if i < len(indices) and indices[i] not in assigned:
          # We have a sample that hasn't been assigned yet.  Assign it to
          # whichever set currently has the lowest fraction of its target for
          # this task.

          index = indices[i]
          set_frac = [
              1 if set_target[i][task] == 0 else
              set_counts[i][task] / set_target[i][task] for i in range(3)
          ]
          s = np.argmin(set_frac)
          set_inds[s].append(index)
          assigned.add(index)
          set_counts[s] += y_present[index]

    # The remaining samples are negative for all tasks.  Add them to fill out
    # each set to the correct total number.

    n_samples = y_present.shape[0]
    set_size = [
        int(np.round(n_samples * f))
        for f in (frac_train, frac_valid, frac_test)
    ]
    s = 0
    for i in np.random.permutation(range(n_samples)):
      if i not in assigned:
        while s < 2 and len(set_inds[s]) >= set_size[s]:
          s += 1
        set_inds[s].append(i)
    return tuple(sorted(x) for x in set_inds)

