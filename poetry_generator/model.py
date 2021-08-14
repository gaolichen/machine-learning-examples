import tensorflow as tf
import tensorflow.keras as keras

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current, '../../simplebert/src'))

from simplebert.models import model_from_pretrained

def poet_loss(from_logits = False):
  loss = keras.losses.SparseCategoricalCrossentropy(from_logits = from_logits, reduction = 'none')

  def eval(target, logits):
    attention_mask = tf.cast(tf.not_equal(target, 0), dtype='float32')
    val = loss(target, logits)
    val = tf.math.multiply(val, attention_mask)
    return tf.math.reduce_sum(val) / (tf.math.reduce_sum(attention_mask) + 1e-6)

  return eval


def build_model(settings):
    input_ids = tf.keras.layers.Input(shape=(settings.input_dim,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(settings.input_dim, ), dtype=tf.int32)
    inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

    bert_model = model_from_pretrained(settings.model_name, model_head = 'lm', causal_attention = True, name = 'bert')
    output = bert_model(inputs)
    
    model = keras.models.Model(inputs, output['logits'], name = 'poetry_generator')

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = settings.learning_rate),
                  loss = poet_loss(from_logits = True))

    return model
