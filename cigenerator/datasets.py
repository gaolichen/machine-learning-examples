# -*- coding: utf-8 -*-
# @File    : datasets.py
# @Author  : Gaoli Chen
# @Time    : 2021/09/28
# @Desc    :

import os
import json
import random
from dataclasses import dataclass
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import get_file

download_url_base = 'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master'

@dataclass
class Ci:
  author: str
  
  rhythmic: str

  content: str


@dataclass
class DownloadInfo:
    """Class for keeping track information of poetry."""
    dynasty: str
    max_index: int
    index_step: int = 1000

    def get_download_url(self, index) -> str:
        return download_url_base + f'/ci/ci.{self.dynasty}.{index * self.index_step}.json'        


download_infos = [
    DownloadInfo(dynasty = "song", max_index = 21)]

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
    if not is_ch(ch):
      filtered = ''
      break

  return filtered

#Unicdoe4E00~9FFF表示中文
def is_ch(ch):
  if '\u4e00' <= ch <= '\u9fff':
    return True
  else:
    return False


class CiCollection(object):
    def __init__(self):
        self.cache_dir = '~/.cache/'
        self.ci_list = []
      
    def download(self, download_info, download_size = None):
        index_range = download_info.max_index + 1
        download_ratio = 1.1

        if isinstance(download_size, int):
            index_range = min(index_range, download_size)
        elif isinstance(download_size, float) and download_size < 1.0:
            download_ratio = download_size
        
        len_dict = dict()
        rhy_to_index = dict()
        index_to_rhy = dict()
        rhy_freq = dict()
        curr_index = 0

        found = False
        for index in range(index_range):
            if download_ratio < 1.0:
                if random.random() > download_ratio:
                    continue

            file_path = self._download_file(download_info, index)
            
            with open(file_path, 'r', encoding='utf8') as f:
              obj_list = json.load(f)
              for obj in obj_list:
                rhy = obj['rhythmic']
                content = self._join(obj['paragraphs'])

                if len(content) == 0:
                  continue
                
                if u'・' in rhy:
                  parts = [part.strip() for part in rhy.split(u'・')]
                else:
                  parts = [rhy]

                idx = curr_index

                for part in parts:
                  rhy_freq[part] = rhy_freq.get(part, 0) + 1
                  if part in rhy_to_index:
                    idx = rhy_to_index[part]
                
                for part in parts:
                  rhy_to_index[part] = idx
                
                if idx == curr_index:
                  index_to_rhy[idx] = parts
                  curr_index += 1
                else:
                  for part in parts:
                    if not part in index_to_rhy[idx]:
                      index_to_rhy[idx].append(part)
                
                rhy = parts[-1]

                self.ci_list.append(Ci(author = obj['author'], rhythmic = rhy, content = content))
                len_dict[len(content)] = len_dict.get(len(content), 0) + 1
        
        # build rhy name mapping.
        rhy_name_mapping = dict()
        for idx in range(curr_index):
          rhy = index_to_rhy[idx][0]
          for r in index_to_rhy[idx][1:]:
            if rhy_freq[r] > rhy_freq[rhy]:
              rhy = r
          for r in index_to_rhy[idx]:
            if r != rhy:
              rhy_name_mapping[r] = rhy
        
        rhy_dict = dict()
        for ci in self.ci_list:
          if ci.rhythmic in rhy_name_mapping:
            ci.rhythmic = rhy_name_mapping[ci.rhythmic]
          rhy_dict[ci.rhythmic] = rhy_dict.get(ci.rhythmic, 0) + 1

        len_stat = sorted(list(len_dict.items()))
        rhy_stat = sorted(list(rhy_dict.items()), key = lambda x : (-x[1], x[0]))
    
    def train_test_data(self, val_split, maxlen = None):
      tr_data = []
      val_data = []

      if maxlen is None:
        maxlen = 10000
      
      for ci in self.ci_list:
        if len(ci.content) > maxlen:
          continue

        if random.random() < val_split:
          val_data.append(ci)
        else:
          tr_data.append(ci)
      
      random.shuffle(tr_data)
      random.shuffle(val_data)
      return tr_data, val_data

    def _download_file(self, info, index) -> str:
        url = info.get_download_url(index)
        fname = url.split('/')[-1]
        
        return get_file(fname = fname,
                        origin = url,
                        extract = False,
                        cache_subdir = 'ci',
                        cache_dir = self.cache_dir)

    @staticmethod
    def _join(text_list, maxlen = None):
        if maxlen is None:
            return ''.join(text_list)
        else:
            ret = ''
            for text in text_list:
                if len(ret) + len(text) <= maxlen:
                    ret += text
                else:
                    break
            return ret


class CiDataGenerator(keras.utils.Sequence):
  def __init__(self, ci_data, input_dim, batch_size, tokenizer, shuffle = True):
    self.input_dim = input_dim
    self.ci_data = ci_data
    self.batch_size = batch_size
    self.tokenizer = tokenizer
    self.shuffle = shuffle
    
    self.on_epoch_end()

  def __len__(self):
    return (len(self.ci_data) + self.batch_size - 1) // self.batch_size
  
  def __getitem__(self, index):
    last_pos = min(len(self.ci_data), (index + 1) * self.batch_size)

    contents = [ci.content for ci in self.ci_data[index * self.batch_size : last_pos]]
    rhys = [ci.rhythmic for ci in self.ci_data[index * self.batch_size : last_pos]]
      
    contents_ids = self.tokenizer.encode(contents)
    rhys_ids = self.tokenizer.encode(rhys)

    input_ids = [r + c[0:-1] for r, c in zip(rhys_ids, contents_ids)]
    label_ids = [[self.tokenizer.pad_token_id] * (len(r) - 1) + c for r, c in zip(rhys_ids, contents_ids)]
    
    input_ids = pad_sequences(input_ids, maxlen = self.input_dim, padding = 'post', truncating = 'post')
    label_ids = pad_sequences(label_ids, maxlen = self.input_dim, padding = 'post', truncating = 'post')
    attention_mask = (input_ids != 0).astype('int32')
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, label_ids
  
  def on_epoch_end(self):
    if self.shuffle:
      random.shuffle(self.ci_data)


if __name__ == '__main__':
    cc = CiCollection()
    cc.download(download_infos[0], download_size = None)
    tr_data, val_data = cc.train_test_data(val_split = 0.1, maxlen = 250)
    print(len(tr_data), len(val_data))
