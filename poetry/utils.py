# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

import os
import json
import random
from zhconv import convert
from transformers import TFBertForMaskedLM, BertConfig
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import settings

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

pad_token_id = tokenizer.pad_token_id
mask_token_id = tokenizer.mask_token_id

#Unicdoe4E00~9FFF表示中文
def is_ch(ch):
  if '\u4e00' <= ch <= '\u9fff':
    return True
  else:
    return False

def text_width(text):
  w = len(text)
  for ch in text:
    if is_ch(ch):
      w += 1
  return w

def rjust(text, width):
  len = text_width(text)
  return (' ' * (width - len)) + text

def ljust(text, width):
  len = text_width(text)
  return text + (' ' * (width - len))


def load_json(index, type):
  path = os.path.join(settings.data_root, str.format('poet.{0}.{1}.json', type, index * 1000))
  with open(path, 'r', encoding='utf8') as f:
    return json.load(f)


def convert_to_simplify(json_obj):
  json_obj['author'] = convert(json_obj['author'], 'zh-cn')
  json_obj['title'] = convert(json_obj['title'], 'zh-cn')
  json_obj['paragraphs'] = [convert(p, 'zh-cn') for p in json_obj['paragraphs']]
  return json_obj


def filter_non_poetry_part(text):
  filtered  = ''

  in_parenthsis = 0
  part = 0
  for ch in text.strip():
    if ch == '（':
      in_parenthsis += 1
    elif ch == '）':
      in_parenthsis -= 1
    if in_parenthsis:
      continue
    
    filtered += ch
    if ch == '，':
      part += 1

    elif ch in ['。', '？', '！']:
      if part != 1:
        filtered = ''
      break
    
    elif not is_ch(ch):
      filtered = ''
      break
    
  return filtered if part == 1 else ''


def filter_unknown_words(sentences):
  ret = []
  for sen in sentences:
    token_ids = tokenizer.encode(sen)
    if not tokenizer.unk_token_id in token_ids:
      ret.append(sen)

  return ret

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


def convert_texts_to_input_ids(texts):
  input_ids = [tokenizer.encode(text) for text in texts]
  input_ids = pad_sequences(input_ids, maxlen = settings.input_dim, padding = 'post', truncating = 'post', value = tokenizer.pad_token_id)
  return input_ids

def convert_ids_to_tokens(ids):
    return tokenizer.convert_ids_to_tokens(ids)

def convert_tokens_to_ids(tokens):
    return tokenizer.convert_tokens_to_ids(tokens)

def load_bert_model():
    config = BertConfig.from_pretrained('bert-base-chinese',
                                        output_attentions = False,
                                        output_hidden_states = False,
                                        use_cache = True,
                                        return_dict = True)
    bert_mlm = TFBertForMaskedLM.from_pretrained('bert-base-chinese', config = config)
    
    return bert_mlm
