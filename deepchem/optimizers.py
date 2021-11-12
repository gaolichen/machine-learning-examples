
"""Optimizers and related classes for use with TensorGraph."""

import math

from typing import Dict, Union, Optional


class Optimizer(object):
  """An algorithm for optimizing a model.
  This is an abstract class.  Subclasses represent specific optimization algorithms.
  """

  def __init__(self, learning_rate: "Union[float, LearningRateSchedule]"):
    """This constructor should only be called by subclasses.
    Parameters
    ----------
    learning_rate: float or LearningRateSchedule
      the learning rate to use for optimization
    """
    self.learning_rate = learning_rate

  def _create_tf_optimizer(self, global_step):
    """Construct a TensorFlow optimizer.
    Parameters
    ----------
    global_step: tensor
      a tensor containing the global step index during optimization, used for learning rate decay
    Returns
    -------
    a new TensorFlow optimizer implementing the algorithm
    """
    raise NotImplementedError("Subclasses must implement this")

  def _create_pytorch_optimizer(self, params):
    """Construct a PyTorch optimizer.
    Parameters
    ----------
    params: Iterable
      the model parameters to optimize
    Returns
    -------
    a new PyTorch optimizer implementing the algorithm
    """
    raise NotImplementedError("Subclasses must implement this")

  def _create_jax_optimizer(self):
    """Construct a Jax optimizer.
    Returns
    -------
    a new Optax optimizer optax.GradientTransformation implementing the algorithm
    """
    raise NotImplementedError("Subclasses must implement this")


class LearningRateSchedule(object):
  """A schedule for changing the learning rate over the course of optimization.
  This is an abstract class.  Subclasses represent specific schedules.
  """

  def _create_tf_tensor(self, global_step):
    """Construct a tensor that equals the learning rate.
    Parameters
    ----------
    global_step: tensor
      a tensor containing the global step index during optimization
    Returns
    -------
    a tensor that equals the learning rate
    """
    raise NotImplementedError("Subclasses must implement this")

  def _create_pytorch_schedule(self, optimizer):
    """Construct a PyTorch learning rate scheduler.
    Parameters
    ----------
    optimizer: torch.optim.Optimizer
      the Optimizer whose learning rate will be modified
    Returns
    -------
    a PyTorch scheduler implementing the schedule
    """
    raise NotImplementedError("Subclasses must implement this")

  def _create_jax_schedule(self, learning_rate):
    """Construct a Jax learning rate scheduler using optax.
    Parameters
    ----------
    learning_rate: float
      the initial learning rate that will be modified
    Returns
    -------
    a optax scheduler implementing the schedule
    """
    raise NotImplementedError("Subclasses must implement this")

class LearningRateSchedule(object):
  """A schedule for changing the learning rate over the course of optimization.
  This is an abstract class.  Subclasses represent specific schedules.
  """

  def _create_tf_tensor(self, global_step):
    """Construct a tensor that equals the learning rate.
    Parameters
    ----------
    global_step: tensor
      a tensor containing the global step index during optimization
    Returns
    -------
    a tensor that equals the learning rate
    """
    raise NotImplementedError("Subclasses must implement this")

  def _create_pytorch_schedule(self, optimizer):
    """Construct a PyTorch learning rate scheduler.
    Parameters
    ----------
    optimizer: torch.optim.Optimizer
      the Optimizer whose learning rate will be modified
    Returns
    -------
    a PyTorch scheduler implementing the schedule
    """
    raise NotImplementedError("Subclasses must implement this")

  def _create_jax_schedule(self, learning_rate):
    """Construct a Jax learning rate scheduler using optax.
    Parameters
    ----------
    learning_rate: float
      the initial learning rate that will be modified
    Returns
    -------
    a optax scheduler implementing the schedule
    """
    raise NotImplementedError("Subclasses must implement this")


class ExponentialDecay(LearningRateSchedule):
  """A learning rate that decreases exponentially with the number of training steps."""

  def __init__(self,
               initial_rate: float,
               decay_rate: float,
               decay_steps: int,
               staircase: bool = True):
    """Create an exponentially decaying learning rate.
    The learning rate starts as initial_rate.  Every decay_steps training steps, it is multiplied by decay_rate.
    Parameters
    ----------
    initial_rate: float
      the initial learning rate
    decay_rate: float
      the base of the exponential
    decay_steps: int
      the number of training steps over which the rate decreases by decay_rate
    staircase: bool
      if True, the learning rate decreases by discrete jumps every decay_steps.
      if False, the learning rate decreases smoothly every step
    """
    self.initial_rate = initial_rate
    self.decay_rate = decay_rate
    self.decay_steps = decay_steps
    self.staircase = staircase

  def _create_tf_tensor(self, global_step):
    import tensorflow as tf
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=self.initial_rate,
        decay_rate=self.decay_rate,
        decay_steps=self.decay_steps,
        staircase=self.staircase)(global_step)

  def _create_pytorch_schedule(self, optimizer):
    import torch
    if self.staircase:
      return torch.optim.lr_scheduler.StepLR(optimizer, self.decay_steps,
                                             self.decay_rate)
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer, math.pow(self.decay_rate, 1 / self.decay_steps))

  def _create_jax_schedule(self):
    import optax
    return optax.exponential_decay(
        init_value=self.initial_rate,
        transition_steps=self.decay_steps,
        decay_rate=self.decay_rate,
        staircase=self.staircase)

class Adam(Optimizer):
  """The Adam optimization algorithm."""

  def __init__(self,
               learning_rate: Union[float, LearningRateSchedule] = 0.001,
               beta1: float = 0.9,
               beta2: float = 0.999,
               epsilon: float = 1e-08):
    """Construct an Adam optimizer.
    Parameters
    ----------
    learning_rate: float or LearningRateSchedule
      the learning rate to use for optimization
    beta1: float
      a parameter of the Adam algorithm
    beta2: float
      a parameter of the Adam algorithm
    epsilon: float
      a parameter of the Adam algorithm
    """
    super(Adam, self).__init__(learning_rate)
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

  def _create_tf_optimizer(self, global_step):
    import tensorflow as tf
    if isinstance(self.learning_rate, LearningRateSchedule):
      learning_rate = self.learning_rate._create_tf_tensor(global_step)
    else:
      learning_rate = self.learning_rate
    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=self.beta1,
        beta_2=self.beta2,
        epsilon=self.epsilon)

  def _create_pytorch_optimizer(self, params):
    import torch
    if isinstance(self.learning_rate, LearningRateSchedule):
      lr = self.learning_rate.initial_rate
    else:
      lr = self.learning_rate
    return torch.optim.Adam(params, lr, (self.beta1, self.beta2), self.epsilon)

  def _create_jax_optimizer(self):
    import optax
    process = []
    if isinstance(self.learning_rate, LearningRateSchedule):
      scheduler = self.learning_rate._create_jax_schedule()
      process.append(optax.scale_by_schedule(scheduler))
      last_process = optax.scale(-1.0)
    else:
      lr = self.learning_rate
      last_process = optax.scale(-1.0 * lr)

    process.append(
        optax.scale_by_adam(b1=self.beta1, b2=self.beta2, eps=self.epsilon))
    process.append(last_process)
    return optax.chain(*process)
