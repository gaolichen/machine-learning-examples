
"""
Process an input dataset into a format suitable for machine learning.
"""
import os
import tempfile
import zipfile
import time
import logging
import warnings
from typing import List, Optional, Tuple, Any, Sequence, Union, Iterator

import pandas as pd
import numpy as np

from dctyping import OneOrMany
#from deepchem.utils.data_utils import load_image_files, load_csv_files, load_json_files, load_sdf_files
from base_classes import Featurizer
from datasets import Dataset, DiskDataset#, NumpyDataset, ImageDataset
#from deepchem.feat.molecule_featurizers import OneHotFeaturizer
#from deepchem.utils.genomics_utils import encode_bio_sequence

logger = logging.getLogger(__name__)

def load_csv_files(input_files: List[str],
                   shard_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
  """Load data as pandas dataframe from CSV files.
  Parameters
  ----------
  input_files: List[str]
    List of filenames
  shard_size: int, default None
    The shard size to yield at one time.
  Returns
  -------
  Iterator[pd.DataFrame]
    Generator which yields the dataframe which is the same shard size.
  """
  # First line of user-specified CSV *must* be header.
  shard_num = 1
  for input_file in input_files:
    if shard_size is None:
      yield pd.read_csv(input_file)
    else:
      logger.info("About to start loading CSV from %s" % input_file)
      for df in pd.read_csv(input_file, chunksize=shard_size):
        logger.info(
            "Loading shard %d of size %s." % (shard_num, str(shard_size)))
        df = df.replace(np.nan, str(""), regex=True)
        shard_num += 1
        yield df


def _convert_df_to_numpy(df: pd.DataFrame,
                         tasks: List[str]) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms a dataframe containing deepchem input into numpy arrays
  This is a private helper method intended to help parse labels and
  weights arrays from a pandas dataframe. Here `df` is a dataframe
  which has columns for each task in `tasks`. These labels are
  extracted into a labels array `y`. Weights `w` are initialized to
  all ones, but weights for any missing labels are set to 0.
  Parameters
  ----------
  df: pd.DataFrame
    Pandas dataframe with columns for all tasks
  tasks: List[str]
    List of tasks
  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
    The tuple is `(w, y)`.
  """
  n_samples = df.shape[0]
  n_tasks = len(tasks)

  y = np.hstack(
      [np.reshape(np.array(df[task].values), (n_samples, 1)) for task in tasks])
  w = np.ones((n_samples, n_tasks))
  if y.dtype.kind in ['O', 'U']:
    missing = (y == '')
    y[missing] = 0
    w[missing] = 0

  return y.astype(float), w.astype(float)


class DataLoader(object):
  """Handles loading/featurizing of data from disk.
  The main use of `DataLoader` and its child classes is to make it
  easier to load large datasets into `Dataset` objects.`
  `DataLoader` is an abstract superclass that provides a
  general framework for loading data into DeepChem. This class should
  never be instantiated directly.  To load your own type of data, make
  a subclass of `DataLoader` and provide your own implementation for
  the `create_dataset()` method.
  To construct a `Dataset` from input data, first instantiate a
  concrete data loader (that is, an object which is an instance of a
  subclass of `DataLoader`) with a given `Featurizer` object. Then
  call the data loader's `create_dataset()` method on a list of input
  files that hold the source data to process. Note that each subclass
  of `DataLoader` is specialized to handle one type of input data so
  you will have to pick the loader class suitable for your input data
  type.
  Note that it isn't necessary to use a data loader to process input
  data. You can directly use `Featurizer` objects to featurize
  provided input into numpy arrays, but note that this calculation
  will be performed in memory, so you will have to write generators
  that walk the source files and write featurized data to disk
  yourself. `DataLoader` and its subclasses make this process easier
  for you by performing this work under the hood.
  """

  def __init__(self,
               tasks: List[str],
               featurizer: Featurizer,
               id_field: Optional[str] = None,
               log_every_n: int = 1000):
    """Construct a DataLoader object.
    This constructor is provided as a template mainly. You
    shouldn't ever call this constructor directly as a user.
    Parameters
    ----------
    tasks: List[str]
      List of task names
    featurizer: Featurizer
      Featurizer to use to process data.
    id_field: str, optional (default None)
      Name of field that holds sample identifier. Note that the
      meaning of "field" depends on the input data type and can have a
      different meaning in different subclasses. For example, a CSV
      file could have a field as a column, and an SDF file could have
      a field as molecular property.
    log_every_n: int, optional (default 1000)
      Writes a logging statement this often.
    """
    if self.__class__ is DataLoader:
      raise ValueError(
          "DataLoader should never be instantiated directly. Use a subclass instead."
      )
    if not isinstance(tasks, list):
      raise ValueError("tasks must be a list.")
    self.tasks = tasks
    self.id_field = id_field
    self.user_specified_features = None
#    if isinstance(featurizer, UserDefinedFeaturizer):
#      self.user_specified_features = featurizer.feature_fields
    self.featurizer = featurizer
    self.log_every_n = log_every_n

  def featurize(self,
                inputs: OneOrMany[Any],
                data_dir: Optional[str] = None,
                shard_size: Optional[int] = 8192) -> Dataset:
    """Featurize provided files and write to specified location.
    DEPRECATED: This method is now a wrapper for `create_dataset()`
    and calls that method under the hood.
    For large datasets, automatically shards into smaller chunks
    for convenience. This implementation assumes that the helper
    methods `_get_shards` and `_featurize_shard` are implemented and
    that each shard returned by `_get_shards` is a pandas dataframe.
    You may choose to reuse or override this method in your subclass
    implementations.
    Parameters
    ----------
    inputs: List
      List of inputs to process. Entries can be filenames or arbitrary objects.
    data_dir: str, default None
      Directory to store featurized dataset.
    shard_size: int, optional (default 8192)
      Number of examples stored in each shard.
    Returns
    -------
    Dataset
      A `Dataset` object containing a featurized representation of data
      from `inputs`.
    """
    warnings.warn(
        "featurize() is deprecated and has been renamed to create_dataset()."
        "featurize() will be removed in DeepChem 3.0", FutureWarning)
    return self.create_dataset(inputs, data_dir, shard_size)

  def create_dataset(self,
                     inputs: OneOrMany[Any],
                     data_dir: Optional[str] = None,
                     shard_size: Optional[int] = 8192) -> Dataset:
    """Creates and returns a `Dataset` object by featurizing provided files.
    Reads in `inputs` and uses `self.featurizer` to featurize the
    data in these inputs.  For large files, automatically shards
    into smaller chunks of `shard_size` datapoints for convenience.
    Returns a `Dataset` object that contains the featurized dataset.
    This implementation assumes that the helper methods `_get_shards`
    and `_featurize_shard` are implemented and that each shard
    returned by `_get_shards` is a pandas dataframe.  You may choose
    to reuse or override this method in your subclass implementations.
    Parameters
    ----------
    inputs: List
      List of inputs to process. Entries can be filenames or arbitrary objects.
    data_dir: str, optional (default None)
      Directory to store featurized dataset.
    shard_size: int, optional (default 8192)
      Number of examples stored in each shard.
    Returns
    -------
    DiskDataset
      A `DiskDataset` object containing a featurized representation of data
      from `inputs`.
    """
    logger.info("Loading raw samples now.")
    logger.info("shard_size: %s" % str(shard_size))

    # Special case handling of single input
    if not isinstance(inputs, list):
      inputs = [inputs]

    def shard_generator():
      for shard_num, shard in enumerate(self._get_shards(inputs, shard_size)):
        time1 = time.time()
        X, valid_inds = self._featurize_shard(shard)
        ids = shard[self.id_field].values
        ids = ids[valid_inds]
        if len(self.tasks) > 0:
          # Featurize task results iff they exist.
          y, w = _convert_df_to_numpy(shard, self.tasks)
          # Filter out examples where featurization failed.
          y, w = (y[valid_inds], w[valid_inds])
          assert len(X) == len(ids) == len(y) == len(w)
        else:
          # For prospective data where results are unknown, it
          # makes no sense to have y values or weights.
          y, w = (None, None)
          assert len(X) == len(ids)

        time2 = time.time()
        logger.info("TIMING: featurizing shard %d took %0.3f s" %
                    (shard_num, time2 - time1))
        yield X, y, w, ids

    return DiskDataset.create_dataset(shard_generator(), data_dir, self.tasks)

  def _get_shards(self, inputs: List, shard_size: Optional[int]) -> Iterator:
    """Stub for children classes.
    Should implement a generator that walks over the source data in
    `inputs` and returns a "shard" at a time. Here a shard is a
    chunk of input data that can reasonably be handled in memory. For
    example, this may be a set of rows from a CSV file or a set of
    molecules from a SDF file. To re-use the
    `DataLoader.create_dataset()` method, each shard must be a pandas
    dataframe.
    If you chose to override `create_dataset()` directly you don't
    need to override this helper method.
    Parameters
    ----------
    inputs: list
      List of inputs to process. Entries can be filenames or arbitrary objects.
    shard_size: int, optional
      Number of examples stored in each shard.
    """
    raise NotImplementedError

  def _featurize_shard(self, shard: Any):
    """Featurizes a shard of input data.
    Recall a shard is a chunk of input data that can reasonably be
    handled in memory. For example, this may be a set of rows from a
    CSV file or a set of molecules from a SDF file. Featurize this
    shard in memory and return the results.
    Parameters
    ----------
    shard: Any
      A chunk of input data
    """
    raise NotImplementedError



class CSVLoader(DataLoader):
  """
  Creates `Dataset` objects from input CSV files.
  This class provides conveniences to load data from CSV files.
  It's possible to directly featurize data from CSV files using
  pandas, but this class may prove useful if you're processing
  large CSV files that you don't want to manipulate directly in
  memory.
  Examples
  --------
  Let's suppose we have some smiles and labels
  >>> smiles = ["C", "CCC"]
  >>> labels = [1.5, 2.3]
  Let's put these in a dataframe.
  >>> import pandas as pd
  >>> df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])
  Let's now write this to disk somewhere. We can now use `CSVLoader` to
  process this CSV dataset.
  >>> import tempfile
  >>> import deepchem as dc
  >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
  ...   df.to_csv(tmpfile.name)
  ...   loader = dc.data.CSVLoader(["task1"], feature_field="smiles",
  ...                              featurizer=dc.feat.CircularFingerprint())
  ...   dataset = loader.create_dataset(tmpfile.name)
  >>> len(dataset)
  2
  Of course in practice you should already have your data in a CSV file if
  you're using `CSVLoader`. If your data is already in memory, use
  `InMemoryLoader` instead.
  Sometimes there will be datasets without specific tasks, for example
  datasets which are used in unsupervised learning tasks. Such datasets
  can be loaded by leaving the `tasks` field empty.
  Example
  -------
  >>> x1, x2 = [2, 3, 4], [4, 6, 8]
  >>> df = pd.DataFrame({"x1":x1, "x2": x2}).reset_index()
  >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
  ...   df.to_csv(tmpfile.name)
  ...   loader = dc.data.CSVLoader(tasks=[], id_field="index", feature_field=["x1", "x2"],
  ...                              featurizer=dc.feat.DummyFeaturizer())
  ...   dataset = loader.create_dataset(tmpfile.name)
  >>> len(dataset)
  3
  """

  def __init__(self,
               tasks: List[str],
               featurizer: Featurizer,
               feature_field: Optional[str] = None,
               id_field: Optional[str] = None,
               smiles_field: Optional[str] = None,
               log_every_n: int = 1000):
    """Initializes CSVLoader.
    Parameters
    ----------
    tasks: List[str]
      List of task names
    featurizer: Featurizer
      Featurizer to use to process data.
    feature_field: str, optional (default None)
      Field with data to be featurized.
    id_field: str, optional, (default None)
      CSV column that holds sample identifier
    smiles_field: str, optional (default None) (DEPRECATED)
      Name of field that holds smiles string.
    log_every_n: int, optional (default 1000)
      Writes a logging statement this often.
    """
    if not isinstance(tasks, list):
      raise ValueError("tasks must be a list.")
    if smiles_field is not None:
      logger.warning(
          "smiles_field is deprecated and will be removed in a future version of DeepChem."
          "Use feature_field instead.")
      if feature_field is not None and smiles_field != feature_field:
        raise ValueError(
            "smiles_field and feature_field if both set must have the same value."
        )
      elif feature_field is None:
        feature_field = smiles_field

    self.tasks = tasks
    self.feature_field = feature_field
    self.id_field = id_field
    if id_field is None:
      self.id_field = feature_field  # Use features as unique ids if necessary
    else:
      self.id_field = id_field
    self.user_specified_features = None
#    if isinstance(featurizer, UserDefinedFeaturizer):
#      self.user_specified_features = featurizer.feature_fields
    self.featurizer = featurizer
    self.log_every_n = log_every_n

  def _get_shards(self, input_files: List[str],
                  shard_size: Optional[int]) -> Iterator[pd.DataFrame]:
    """Defines a generator which returns data for each shard
    Parameters
    ----------
    input_files: List[str]
      List of filenames to process
    shard_size: int, optional
      The size of a shard of data to process at a time.
    Returns
    -------
    Iterator[pd.DataFrame]
      Iterator over shards
    """
    return load_csv_files(input_files, shard_size)

  def _featurize_shard(self,
                       shard: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Featurizes a shard of an input dataframe.
    Parameters
    ----------
    shard: pd.DataFrame
      DataFrame that holds a shard of the input CSV file
    Returns
    -------
    features: np.ndarray
      Features computed from CSV file.
    valid_inds: np.ndarray
      Indices of rows in source CSV with valid data.
    """
    logger.info("About to featurize shard.")
    if self.featurizer is None:
      raise ValueError(
          "featurizer must be specified in constructor to featurizer data/")
    features = [elt for elt in self.featurizer(shard[self.feature_field])]
    valid_inds = np.array(
        [1 if np.array(elt).size > 0 else 0 for elt in features], dtype=bool)
    features = [
        elt for (is_valid, elt) in zip(valid_inds, features) if is_valid
    ]
    return np.array(features), valid_inds
