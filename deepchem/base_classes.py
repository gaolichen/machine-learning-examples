"""
Feature calculations.
"""
import inspect
import logging
import numpy as np
from typing import Any, Dict, Iterable, Optional, Tuple, Union, cast

logger = logging.getLogger(__name__)


class Featurizer(object):
  """Abstract class for calculating a set of features for a datapoint.
  This class is abstract and cannot be invoked directly. You'll
  likely only interact with this class if you're a developer. In
  that case, you might want to make a child class which
  implements the `_featurize` method for calculating features for
  a single datapoints if you'd like to make a featurizer for a
  new datatype.
  """

  def featurize(self,
                datapoints: Iterable[Any],
                log_every_n: int = 1000,
                **kwargs) -> np.ndarray:
    """Calculate features for datapoints.
    Parameters
    ----------
    datapoints: Iterable[Any]
      A sequence of objects that you'd like to featurize. Subclassses of
      `Featurizer` should instantiate the `_featurize` method that featurizes
      objects in the sequence.
    log_every_n: int, default 1000
      Logs featurization progress every `log_every_n` steps.
    Returns
    -------
    np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    datapoints = list(datapoints)
    features = []
    for i, point in enumerate(datapoints):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)
      try:
        features.append(self._featurize(point, **kwargs))
      except:
        logger.warning(
            "Failed to featurize datapoint %d. Appending empty array")
        features.append(np.array([]))

    return np.asarray(features)

  def __call__(self, datapoints: Iterable[Any], **kwargs):
    """Calculate features for datapoints.
    `**kwargs` will get passed directly to `Featurizer.featurize`
    Parameters
    ----------
    datapoints: Iterable[Any]
      Any blob of data you like. Subclasss should instantiate this.
    """
    return self.featurize(datapoints, **kwargs)

  def _featurize(self, datapoint: Any, **kwargs):
    """Calculate features for a single datapoint.
    Parameters
    ----------
    datapoint: Any
      Any blob of data you like. Subclass should instantiate this.
    """
    raise NotImplementedError('Featurizer is not defined.')

  def __repr__(self) -> str:
    """Convert self to repr representation.
    Returns
    -------
    str
      The string represents the class.
    Examples
    --------
    >>> import deepchem as dc
    >>> dc.feat.CircularFingerprint(size=1024, radius=4)
    CircularFingerprint[radius=4, size=1024, chiral=False, bonds=True, features=False, sparse=False, smiles=False]
    >>> dc.feat.CGCNNFeaturizer()
    CGCNNFeaturizer[radius=8.0, max_neighbors=12, step=0.2]
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

  def __str__(self) -> str:
    """Convert self to str representation.
    Returns
    -------
    str
      The string represents the class.
    Examples
    --------
    >>> import deepchem as dc
    >>> str(dc.feat.CircularFingerprint(size=1024, radius=4))
    'CircularFingerprint_radius_4_size_1024'
    >>> str(dc.feat.CGCNNFeaturizer())
    'CGCNNFeaturizer'
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

class MolecularFeaturizer(Featurizer):
  """Abstract class for calculating a set of features for a
  molecule.
  The defining feature of a `MolecularFeaturizer` is that it
  uses SMILES strings and RDKit molecule objects to represent
  small molecules. All other featurizers which are subclasses of
  this class should plan to process input which comes as smiles
  strings or RDKit molecules.
  Child classes need to implement the _featurize method for
  calculating features for a single molecule.
  Note
  ----
  The subclasses of this class require RDKit to be installed.
  """

  def featurize(self, datapoints, log_every_n=1000, **kwargs) -> np.ndarray:
    """Calculate features for molecules.
    Parameters
    ----------
    datapoints: rdkit.Chem.rdchem.Mol / SMILES string / iterable
      RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
      strings.
    log_every_n: int, default 1000
      Logging messages reported every `log_every_n` samples.
    Returns
    -------
    features: np.ndarray
      A numpy array containing a featurized representation of `datapoints`.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import rdmolfiles
      from rdkit.Chem import rdmolops
      from rdkit.Chem.rdchem import Mol
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")

    if 'molecules' in kwargs:
      datapoints = kwargs.get("molecules")
      raise DeprecationWarning(
          'Molecules is being phased out as a parameter, please pass "datapoints" instead.'
      )

    # Special case handling of single molecule
    if isinstance(datapoints, str) or isinstance(datapoints, Mol):
      datapoints = [datapoints]
    else:
      # Convert iterables to list
      datapoints = list(datapoints)

    features: list = []
    for i, mol in enumerate(datapoints):
      if i % log_every_n == 0:
        logger.info("Featurizing datapoint %i" % i)

      try:
        if isinstance(mol, str):
          # mol must be a RDKit Mol object, so parse a SMILES
          mol = Chem.MolFromSmiles(mol)
          # SMILES is unique, so set a canonical order of atoms
          new_order = rdmolfiles.CanonicalRankAtoms(mol)
          mol = rdmolops.RenumberAtoms(mol, new_order)

        features.append(self._featurize(mol, **kwargs))
      except Exception as e:
        if isinstance(mol, Chem.rdchem.Mol):
          mol = Chem.MolToSmiles(mol)
        logger.warning(
            "Failed to featurize datapoint %d, %s. Appending empty array", i,
            mol)
        logger.warning("Exception message: {}".format(e))
        features.append(np.array([]))

    return np.asarray(features)