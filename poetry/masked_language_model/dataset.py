# -*- coding: utf-8 -*-
# @File    : dataset.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

import random
import numpy as np
import tensorflow.keras as keras

def to_masked_sentence(sentences):
  masked_sentences = []
  masked_words = []

  for sen in sentences:
    while True:
      w_index = random.randint(0, len(sen) - 1)
      if sen[w_index] not in ['。', '，', '？', '！']:
        break

    masked_words.append(sen[w_index])
    masked_sentences.append(sen[:w_index] + '[MASK]' + sen[w_index + 1:])
  return masked_sentences, masked_words

class MaskedPoetryDataSet(keras.utils.Sequence):
  def __init__(self, sentences, tokenizer, settings):
    self.tokenizer = tokenizer
    self.sentences = sentences
    self.input_dim = settings.input_dim
    self.batch_size = settings.batch_size
    self.input_dim = settings.input_dim
  
  def __len__(self):
    return (len(self.sentences) + self.batch_size - 1) // self.batch_size
  
  def __getitem__(self, index):
    end_pos = min((index + 1) * self.batch_size, len(self.sentences))
    batch = self.sentences[index * self.batch_size: end_pos]
    masked_sentences, masked_words = to_masked_sentence(batch)
    input_ids = self.tokenizer.encode(masked_sentences, maxlen = self.input_dim)
    masked_words_ids = self.tokenizer.tokens_to_ids(masked_words)

    return input_ids, np.array(masked_words_ids)
  
  def on_epoch_end(self):
    random.shuffle(self.sentences)
