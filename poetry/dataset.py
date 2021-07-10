# -*- coding: utf-8 -*-
# @File    : dataset.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

import numpy as np
import random
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import settings
import utils

def random_poetry_sentences(size, type = 'tang'):
  max_index = settings.max_tang_index if type == 'tang' else settings.max_song_index
  indices = np.arange(1, max_index + 1)
  np.random.shuffle(indices)
  sentences = []

  for index in indices:
    peotry_list = utils.load_json(index, type)
    for poetry in peotry_list:
      if not poetry['paragraphs']:
        continue

      utils.convert_to_simplify(poetry)
      p_index = random.randint(0, len(poetry['paragraphs']) - 1)
      text = utils.filter_non_poetry_part(poetry['paragraphs'][p_index])
      length = len(text)
      if length < 10 or length > settings.input_dim - 2:
        continue

      sentences.append(text)
      if len(sentences) >= size:
        return sentences 

  return sentences


def load_poetry_sentences(tang_poetry_size, song_poetry_size):
  tang_sentences = random_poetry_sentences(tang_poetry_size, 'tang')
  song_sentences = random_poetry_sentences(song_poetry_size, 'song')

  all_poetry_sentences = tang_sentences + song_sentences
  #all_poetry_sentences = utils.filter_unknown_words(all_poetry_sentences)

  tr_indices, val_indices = train_test_split(list(range(len(all_poetry_sentences))), test_size = settings.val_split)

  tr_poetry_sentences = [all_poetry_sentences[i] for i in tr_indices]
  val_poetry_sentences = [all_poetry_sentences[i] for i in val_indices]

  return tr_poetry_sentences, val_poetry_sentences


class MaskedPoetryDataSet(keras.utils.Sequence):
  def __init__(self, sentences, input_dim, batch_size):
    self.sentences = sentences
    self.input_dim = input_dim
    self.batch_size = batch_size
  
  def __len__(self):
    return (len(self.sentences) + self.batch_size - 1) // self.batch_size
  
  def __getitem__(self, index):
    end_pos = min((index + 1) * self.batch_size, len(self.sentences))
    batch = self.sentences[index * self.batch_size: end_pos]
    masked_sentences, masked_words = utils.to_masked_sentence(batch)
    input_ids = utils.convert_texts_to_input_ids(masked_sentences)
    masked_words_ids = utils.convert_tokens_to_ids(masked_words)

    return input_ids, np.array(masked_words_ids)
  
  def on_epoch_end(self):
    random.shuffle(self.sentences)
