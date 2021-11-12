"""
Contains an abstract base class that supports data transformations.
"""
import os
import logging
import time
import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy
import scipy.ndimage

#import deepchem as dc
from datasets import Dataset, DiskDataset

logger = logging.getLogger(__name__)


def undo_grad_transforms(grad, tasks, transformers):
  """DEPRECATED. DO NOT USE."""
  logger.warning(
      "undo_grad_transforms is DEPRECATED and will be removed in a future version of DeepChem. "
      "Manually implement transforms to perform force calculations.")
  for transformer in reversed(transformers):
    if transformer.transform_y:
      grad = transformer.untransform_grad(grad, tasks)
  return grad


def get_grad_statistics(dataset):
  """Computes and returns statistics of a dataset
  DEPRECATED DO NOT USE.
  This function assumes that the first task of a dataset holds the
  energy for an input system, and that the remaining tasks holds the
  gradient for the system.
  """
  logger.warning(
      "get_grad_statistics is DEPRECATED and will be removed in a future version of DeepChem. Manually compute force/energy statistics."
  )
  if len(dataset) == 0:
    return None, None, None, None
  y = dataset.y
  energy = y[:, 0]
  grad = y[:, 1:]
  for i in range(energy.size):
    grad[i] *= energy[i]
  ydely_means = np.sum(grad, axis=0) / len(energy)
  return grad, ydely_means


class Transformer(object):
  """Abstract base class for different data transformation techniques.
  A transformer is an object that applies a transformation to a given
  dataset. Think of a transformation as a mathematical operation which
  makes the source dataset more amenable to learning. For example, one
  transformer could normalize the features for a dataset (ensuring
  they have zero mean and unit standard deviation). Another
  transformer could for example threshold values in a dataset so that
  values outside a given range are truncated. Yet another transformer
  could act as a data augmentation routine, generating multiple
  different images from each source datapoint (a transformation need
  not necessarily be one to one).
  Transformers are designed to be chained, since data pipelines often
  chain multiple different transformations to a dataset. Transformers
  are also designed to be scalable and can be applied to
  large `dc.data.Dataset` objects. Not that Transformers are not
  usually thread-safe so you will have to be careful in processing
  very large datasets.
  This class is an abstract superclass that isn't meant to be directly
  instantiated. Instead, you will want to instantiate one of the
  subclasses of this class inorder to perform concrete
  transformations.
  """
  # Hack to allow for easy unpickling:
  # http://stefaanlippens.net/pickleproblem
  __module__ = os.path.splitext(os.path.basename(__file__))[0]

  def __init__(self,
               transform_X: bool = False,
               transform_y: bool = False,
               transform_w: bool = False,
               transform_ids: bool = False,
               dataset: Optional[Dataset] = None):
    """Initializes transformation based on dataset statistics.
    Parameters
    ----------
    transform_X: bool, optional (default False)
      Whether to transform X
    transform_y: bool, optional (default False)
      Whether to transform y
    transform_w: bool, optional (default False)
      Whether to transform w
    transform_ids: bool, optional (default False)
      Whether to transform ids
    dataset: dc.data.Dataset object, optional (default None)
      Dataset to be transformed
    """
    if self.__class__.__name__ == "Transformer":
      raise ValueError(
          "Transformer is an abstract superclass and cannot be directly instantiated. You probably want to instantiate a concrete subclass instead."
      )
    self.transform_X = transform_X
    self.transform_y = transform_y
    self.transform_w = transform_w
    self.transform_ids = transform_ids
    # Some transformation must happen
    assert transform_X or transform_y or transform_w or transform_ids

  def transform_array(
      self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
      ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform the data in a set of (X, y, w, ids) arrays.
    Parameters
    ----------
    X: np.ndarray
      Array of features
    y: np.ndarray
      Array of labels
    w: np.ndarray
      Array of weights.
    ids: np.ndarray
      Array of identifiers.
    Returns
    -------
    Xtrans: np.ndarray
      Transformed array of features
    ytrans: np.ndarray
      Transformed array of labels
    wtrans: np.ndarray
      Transformed array of weights
    idstrans: np.ndarray
      Transformed array of ids
    """
    raise NotImplementedError(
        "Each Transformer is responsible for its own transform_array method.")

  def untransform(self, transformed):
    """Reverses stored transformation on provided data.
    Depending on whether `transform_X` or `transform_y` or `transform_w` was
    set, this will perform different un-transformations. Note that this method
    may not always be defined since some transformations aren't 1-1.
    Parameters
    ----------
    transformed: np.ndarray
      Array which was previously transformed by this class.
    """
    raise NotImplementedError(
        "Each Transformer is responsible for its own untransform method.")

  def transform(self,
                dataset: Dataset,
                parallel: bool = False,
                out_dir: Optional[str] = None,
                **kwargs) -> Dataset:
    """Transforms all internally stored data in dataset.
    This method transforms all internal data in the provided dataset by using
    the `Dataset.transform` method. Note that this method adds X-transform,
    y-transform columns to metadata. Specified keyword arguments are passed on
    to `Dataset.transform`.
    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset object to be transformed.
    parallel: bool, optional (default False)
      if True, use multiple processes to transform the dataset in parallel.
      For large datasets, this might be faster.
    out_dir: str, optional
      If `out_dir` is specified in `kwargs` and `dataset` is a `DiskDataset`,
      the output dataset will be written to the specified directory.
    Returns
    -------
    Dataset
      A newly transformed Dataset object
    """
    # Add this case in to handle non-DiskDataset that should be written to disk
    if out_dir is not None:
      if not isinstance(dataset, dc.data.DiskDataset):
        dataset = dc.data.DiskDataset.from_numpy(dataset.X, dataset.y,
                                                 dataset.w, dataset.ids)
    _, y_shape, w_shape, _ = dataset.get_shape()
    if y_shape == tuple() and self.transform_y:
      raise ValueError("Cannot transform y when y_values are not present")
    if w_shape == tuple() and self.transform_w:
      raise ValueError("Cannot transform w when w_values are not present")
    return dataset.transform(self, out_dir=out_dir, parallel=parallel)

  def transform_on_array(
      self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
      ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transforms numpy arrays X, y, and w
    DEPRECATED. Use `transform_array` instead.
    Parameters
    ----------
    X: np.ndarray
      Array of features
    y: np.ndarray
      Array of labels
    w: np.ndarray
      Array of weights.
    ids: np.ndarray
      Array of identifiers.
    Returns
    -------
    Xtrans: np.ndarray
      Transformed array of features
    ytrans: np.ndarray
      Transformed array of labels
    wtrans: np.ndarray
      Transformed array of weights
    idstrans: np.ndarray
      Transformed array of ids
    """
    warnings.warn(
        "transform_on_array() is deprecated and has been renamed to transform_array()."
        "transform_on_array() will be removed in DeepChem 3.0", FutureWarning)
    X, y, w, ids = self.transform_array(X, y, w, ids)
    return X, y, w, ids


def undo_transforms(y: np.ndarray,
                    transformers: List[Transformer]) -> np.ndarray:
  """Undoes all transformations applied.
  Transformations are reversed using `transformer.untransform`.
  Transformations will be assumed to have been applied in the order specified,
  so transformations will be reversed in the opposite order. That is if
  `transformers = [t1, t2]`, then this method will do `t2.untransform`
  followed by `t1.untransform`.
  Parameters
  ----------
  y: np.ndarray
    Array of values for which transformations have to be undone.
  transformers: list[dc.trans.Transformer]
    List of transformations which have already been applied to `y` in the
    order specifed.
  Returns
  -------
  y_out: np.ndarray
    The array with all transformations reversed.
  """
  # Note that transformers have to be undone in reversed order
  for transformer in reversed(transformers):
    if transformer.transform_y:
      y = transformer.untransform(y)
  return y


class BalancingTransformer(Transformer):
  """Balance positive and negative (or multiclass) example weights.
  This class balances the sample weights so that the sum of all example
  weights from all classes is the same. This can be useful when you're
  working on an imbalanced dataset where there are far fewer examples of some
  classes than others.
  Examples
  --------
  Here's an example for a binary dataset.
  >>> n_samples = 10
  >>> n_features = 3
  >>> n_tasks = 1
  >>> n_classes = 2
  >>> ids = np.arange(n_samples)
  >>> X = np.random.rand(n_samples, n_features)
  >>> y = np.random.randint(n_classes, size=(n_samples, n_tasks))
  >>> w = np.ones((n_samples, n_tasks))
  >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
  >>> transformer = dc.trans.BalancingTransformer(dataset=dataset)
  >>> dataset = transformer.transform(dataset)
  And here's a multiclass dataset example.
  >>> n_samples = 50
  >>> n_features = 3
  >>> n_tasks = 1
  >>> n_classes = 5
  >>> ids = np.arange(n_samples)
  >>> X = np.random.rand(n_samples, n_features)
  >>> y = np.random.randint(n_classes, size=(n_samples, n_tasks))
  >>> w = np.ones((n_samples, n_tasks))
  >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
  >>> transformer = dc.trans.BalancingTransformer(dataset=dataset)
  >>> dataset = transformer.transform(dataset)
  See Also
  --------
  deepchem.trans.DuplicateBalancingTransformer: Balance by duplicating samples.
  Note
  ----
  This transformer is only meaningful for classification datasets where `y`
  takes on a limited set of values. This class can only transform `w` and does
  not transform `X` or `y`.
  Raises
  ------
  ValueError
    if `transform_X` or `transform_y` are set. Also raises or if `y` or `w` aren't of shape `(N,)` or `(N, n_tasks)`.
  """

  def __init__(self, dataset: Dataset):
    # BalancingTransformer can only transform weights.
    super(BalancingTransformer, self).__init__(
        transform_w=True, dataset=dataset)

    # Compute weighting factors from dataset.
    y = dataset.y
    w = dataset.w
    # Handle 1-D case
    if len(y.shape) == 1:
      y = np.reshape(y, (len(y), 1))
    if len(w.shape) == 1:
      w = np.reshape(w, (len(w), 1))
    if len(y.shape) != 2:
      raise ValueError("y must be of shape (N,) or (N, n_tasks)")
    if len(w.shape) != 2:
      raise ValueError("w must be of shape (N,) or (N, n_tasks)")
    self.classes = sorted(np.unique(y))
    weights = []
    for ind, task in enumerate(dataset.get_task_names()):
      task_w = w[:, ind]
      task_y = y[:, ind]
      # Remove labels with zero weights
      task_y = task_y[task_w != 0]
      N_task = len(task_y)
      class_counts = []
      # Note that we may have 0 elements of a given class since we remove those
      # labels with zero weight. This typically happens in multitask datasets
      # where some datapoints only have labels for some tasks.
      for c in self.classes:
        # this works because task_y is 1D
        num_c = len(np.where(task_y == c)[0])
        class_counts.append(num_c)
      # This is the right ratio since N_task/num_c * num_c = N_task
      # for all classes
      class_weights = [
          N_task / float(num_c) if num_c > 0 else 0 for num_c in class_counts
      ]
      weights.append(class_weights)
    self.weights = weights

  def transform_array(
      self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
      ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform the data in a set of (X, y, w) arrays.
    Parameters
    ----------
    X: np.ndarray
      Array of features
    y: np.ndarray
      Array of labels
    w: np.ndarray
      Array of weights.
    ids: np.ndarray
      Array of weights.
    Returns
    -------
    Xtrans: np.ndarray
      Transformed array of features
    ytrans: np.ndarray
      Transformed array of labels
    wtrans: np.ndarray
      Transformed array of weights
    idstrans: np.ndarray
      Transformed array of ids
    """
    w_balanced = np.zeros_like(w)
    if len(y.shape) == 1 and len(w.shape) == 2 and w.shape[1] == 1:
      y = np.expand_dims(y, 1)
    if len(y.shape) == 1:
      n_tasks = 1
    elif len(y.shape) == 2:
      n_tasks = y.shape[1]
    else:
      raise ValueError("y must be of shape (N,) or (N, n_tasks)")
    for ind in range(n_tasks):
      if n_tasks == 1:
        task_y = y
        task_w = w
      else:
        task_y = y[:, ind]
        task_w = w[:, ind]
      for i, c in enumerate(self.classes):
        class_indices = np.logical_and(task_y == c, task_w != 0)
        # Set to the class weight computed previously
        if n_tasks == 1:
          w_balanced[class_indices] = self.weights[ind][i]
        else:
          w_balanced[class_indices, ind] = self.weights[ind][i]
    return (X, y, w_balanced, ids)
