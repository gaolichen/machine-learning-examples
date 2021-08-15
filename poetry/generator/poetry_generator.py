# -*- coding: utf-8 -*-
# @File    : poetry_generator.py
# @Author  : Gaoli Chen
# @Time    : 2021/08/12
# @Desc    :

import tensorflow as tf
import numpy as np

class PoetryGenerator:
  def __init__(self, model, tokenizer, input_dim):
    self.model = model
    self.tokenizer = tokenizer
    self.input_dim = input_dim
    self.punc_ids = tokenizer.tokens_to_ids([u'。', u'，', u'？', u'！'])
  
  def _next_token_id(self, head_ids, topk = 5):
    attention_mask = [1] * len(head_ids) + [0] * (self.input_dim - len(head_ids))
    input_ids = head_ids + [0] * (self.input_dim - len(head_ids))    

    res = self.model.predict({'input_ids': np.array([input_ids]), 'attention_mask': np.array([attention_mask])})
    res = res[0][len(head_ids) - 1]
    token_ids = tf.argsort(res, direction = 'DESCENDING')[:topk]
    res = tf.gather(res, token_ids)
    prob = tf.nn.softmax(res)
    return np.random.choice(token_ids, p = prob.numpy())

  def next_token(self, head = '', topk = 5):
    head_ids = self.tokenizer.encode(head)[:-1]
    tok_id = self._next_token_id(head_ids, topk = topk)
    tok = self.tokenizer.ids_to_tokens([tok_id])
    return tok[0]
  
  def new_peot_with_heads(self, heads, topk = 5):
    head_index = 0
    maxlen = 20
    
    poet_ids = [self.tokenizer.cls_token_id]
    head_ids = self.tokenizer.tokens_to_ids(heads)
    
    for head_id in head_ids:
      poet_ids += [head_id]
      for i in range(maxlen):
        tok_id = self._next_token_id(head_ids = poet_ids, topk = topk)
        if tok_id == self.tokenizer.unk_token_id:
          continue
        
        poet_ids += [tok_id]
        if tok_id in self.punc_ids:
          break

    return self.tokenizer.decode(poet_ids[1:])

  def new_peot_with_start(self, start = '', topk = 5):
    periods = 0
    poet_ids = self.tokenizer.encode(start)[:-1]
    maxlen_to_gen = self.input_dim - len(poet_ids)
    
    for i in range(maxlen_to_gen):
      tok_id = self._next_token_id(head_ids = poet_ids, topk = topk)
      if tok_id == self.tokenizer.sep_token_id:
        if periods >= 4:
          break
        else:
          continue
          
      elif tok_id == self.tokenizer.unk_token_id:
        continue

      poet_ids += [tok_id]
      if tok_id in self.punc_ids:
        periods += 1
        
    return self.tokenizer.decode(poet_ids[1:])
  
  def random(self, topk = 5):
    start = self.tokenizer.unk_token
    # the first token should not be [UNK]
    while start == self.tokenizer.unk_token:
      start = self.next_token(head = '', topk = 1000)
      
    return self.new_peot_with_start(start = start, topk = topk)
