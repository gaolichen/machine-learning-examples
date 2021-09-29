# -*- coding: utf-8 -*-
# @File    : ci_generator.py
# @Author  : Gaoli Chen
# @Time    : 2021/09/28
# @Desc    :

import tensorflow as tf
import numpy as np

class CiGenerator:
  def __init__(self, model, tokenizer, input_dim):
    self.model = model
    self.tokenizer = tokenizer
    self.input_dim = input_dim
  
  def _next_token_id(self, input_ids, topk = 5):
    attention_mask = [1] * len(input_ids) + [0] * (self.input_dim - len(input_ids))
    last_pos = len(input_ids) - 1
    input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.input_dim - len(input_ids))    

    res = self.model.predict({'input_ids': np.array([input_ids]), 'attention_mask': np.array([attention_mask])})
    res = res[0][last_pos]
    token_ids = tf.argsort(res, direction = 'DESCENDING')[:topk]
    res = tf.gather(res, token_ids)
    prob = tf.nn.softmax(res)
    return np.random.choice(token_ids, p = prob.numpy())
  
  def make_ci(self, rhy, topk = 5):
    token_ids1 = self.tokenizer.encode(rhy)
    token_ids2 = []
    maxlen_to_gen = self.input_dim - len(token_ids1)
    
    for i in range(maxlen_to_gen):
      tok_id = self._next_token_id(input_ids = token_ids1 + token_ids2, topk = topk)
      if tok_id == self.tokenizer.sep_token_id:
        break
          
      elif tok_id == self.tokenizer.unk_token_id:
        continue

      token_ids2 += [tok_id]
        
    return self.tokenizer.decode(token_ids2)
  
