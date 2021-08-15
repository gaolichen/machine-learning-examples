# -*- coding: utf-8 -*-
# @File    : datasets.py
# @Author  : Gaoli Chen
# @Time    : 2021/08/12
# @Desc    :

import os
import json
import random
from zhconv import convert
from dataclasses import dataclass
from zhconv import convert
from tensorflow.keras.utils import get_file

download_url_base = 'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master'


@dataclass
class PoetryInfo:
    """Class for keeping track information of poetry."""
    dynasty: str
    max_index: int
    index_step: int = 1000

    def get_download_url(self, index) -> str:
        return download_url_base + f'/json/poet.{self.dynasty}.{index * self.index_step}.json'        


poetry_infos = [
    PoetryInfo(dynasty = "tang", max_index = 57),
    PoetryInfo(dynasty = "song", max_index = 254)]


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

#Unicdoe4E00~9FFF表示中文
def is_ch(ch):
  if '\u4e00' <= ch <= '\u9fff':
    return True
  else:
    return False


class ChinesePoetry(object):
    def __init__(self):
        self.info_dict = { info.dynasty:info for info in poetry_infos}
        self.cache_dir = '~/.cache/'

    def load_dataset(self, dynasty,
                     whole_poetry = False,
                     maxlen = None,
                     val_split = None,
                     download_size = None):
        if not dynasty in self.info_dict:
            raise ValueError(f'Cannot find the poetry of {dynasty}.')

        texts = []
        info = self.info_dict[dynasty]

        index_range = info.max_index + 1
        download_ratio = 1.1

        if isinstance(download_size, int):
            index_range = min(index_range, download_size)
        elif isinstance(download_size, float) and download_size < 1.0:
            download_ratio = download_size


        tr_texts = []
        val_texts = []

        for index in range(index_range):
            if download_ratio < 1.0:
                if random.random() > download_ratio:
                    continue
            
            file_path = self._download_file(info, index)
            with open(file_path, 'r', encoding='utf8') as f:
                obj_list = json.load(f)
            for obj in obj_list:
                obj = self._convert_to_simplify(obj)
                if val_split is None or random.random() >  val_split:
                    text_list = tr_texts
                else:
                    text_list = val_texts
                    
                if whole_poetry:
                    text_list.append(self._join(obj['paragraphs'], maxlen = maxlen))
                else:
                    text_list.extend(obj['paragraphs'])

        if val_split is None:
            return tr_texts
        else:
            return tr_texts, val_texts

    def _download_file(self, info, index) -> str:
        url = info.get_download_url(index)
        fname = url.split('/')[-1]
        
        return get_file(fname = fname,
                        origin = url,
                        extract = False,
                        cache_subdir = 'poetry',
                        cache_dir = self.cache_dir)

    @staticmethod
    def _convert_to_simplify(json_obj):
      json_obj['author'] = convert(json_obj['author'], 'zh-cn')
      json_obj['title'] = convert(json_obj['title'], 'zh-cn')
      l = [filter_non_poetry_part(convert(p, 'zh-cn')) for p in json_obj['paragraphs']]
      json_obj['paragraphs'] = [s for s in l if len(s) > 0]
      return json_obj

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

if __name__ == '__main__':
    cp = ChinesePoetry()
    tr_data, val_data = cp.load_dataset('tang', whole_poetry = True, maxlen = 35, download_size = 0.1, val_split = 0.2)
    print(len(tr_data), len(val_data))
    print(tr_data[:10])
    print(val_data[:10])
    #print(filter_non_poetry_part(convert(u'棘樹寒雲色，茵蔯春藕香。', 'zh-cn')))
