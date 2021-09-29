# -*- coding: utf-8 -*-
# @File    : train.py
# @Author  : Gaoli Chen
# @Time    : 2021/09/28
# @Desc    :

from datetime import datetime
import tensorflow.keras as keras

from dataclasses import dataclass
from cigenerator.datasets import CiDataGenerator, CiCollection, download_infos
from cigenerator.ci_generator import CiGenerator
from cigenerator.model import build_model

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
if os.environ.get('SIMPLEBERT_LOCAL_SOURCE', '') == "1":
    sys.path.append(os.path.join(current, '../../simplebert/src'))

from simplebert import tokenizer_from_pretrained, model_from_pretrained, config_from_pretrained
from simplebert.models import BertModel

@dataclass
class Settings(object):
    epochs: int = 5
    batch_size: int = 32
    input_dim: int = 256
    learning_rate: float = 5e-5
    model_name: str = 'bert-base-chinese'
    save_dir: str = '.'
    download_size: int = None
    val_split: float = 0.1

class CiGenCallback(keras.callbacks.Callback):
  def __init__(self, ci_gen, file_name = None):
    super(CiGenCallback, self).__init__()

    self.gen = ci_gen
    self.file_name = file_name

  def _print(self, content):
    print(content)
    if not self.file_name is None:
      with open(self.file_name, 'a') as f:
        print(content, file = f)
  
  def _generate(self):
    print()
    print('generating test rhy...')

    rhys = [u'浣溪沙', u'水调歌头', u'念奴娇', u'江城子', u'满江红', u'虞美人']

    for rhy in rhys:
        self._print(rhy + '\t' + self.gen.make_ci(rhy))
  
  def on_epoch_end(self, epoch, logs = None):
    self._generate()

def train(settings):
    tokenizer = tokenizer_from_pretrained(settings.model_name)

    cc = CiCollection()
    cc.download(download_infos[0], download_size = settings.download_size)
    tr_ci, val_ci = cc.train_test_data(val_split = settings.val_split, maxlen = settings.input_dim - 10)

    tr_data = CiDataGenerator(tr_ci, tokenizer = tokenizer,
                                input_dim = settings.input_dim,
                                batch_size = settings.batch_size, shuffle = True)
    
    val_data = CiDataGenerator(val_ci, tokenizer = tokenizer,
                                input_dim = settings.input_dim,
                                batch_size = settings.batch_size)
        
    model = build_model(settings)
    print(model.summary())

    gen = CiGenerator(model = model, tokenizer = tokenizer, input_dim = settings.input_dim)

    timestr = datetime.now().strftime('%y-%m-%d-%H')
    output_path = os.path.join(settings.save_dir, f'{model.name}_{timestr}_output.txt')
    print('training...')
    hist = model.fit(tr_data,
                     validation_data = val_data,
                     epochs = settings.epochs,
                     callbacks = [CiGenCallback(gen, output_path)])
    return model

def eval(settings, checkpoint_path):
    tokenizer = tokenizer_from_pretrained(settings.model_name)
    config = config_from_pretrained(settings.model_name)
    model = BertModel(config, model_head = 'lm', causal_attention = True, name = 'bert')
    model.load_weights(checkpoint_path)
    gen = CiGenCallback(model = model, tokenizer = tokenizer, input_dim = settings.input_dim)

    while True:
        cmd = input(u'输入词牌名或者按q退出:')
        if cmd == 'q':
            break
        else:
            print(gen.make_ci(cmd))

if __name__ == "__main__":
    settings = Settings(epochs = 10)
    
    n = len(sys.argv)
    if n >= 2:
        cmd = sys.argv[1]
    else:
        cmd = '-train'
    
    if cmd == '-train':
        train(settings)
    elif cmd == '-eval':
        checkpoint_path = None if n <3 else sys.argv[2]
        if checkpoint_path == None:
            raise ValueError('Invalid checkpoint path {checkpoint_path}')
        eval(settings, checkpoint_path)
    else:
        raise ValueError(f'Invalid command {cmd}')
