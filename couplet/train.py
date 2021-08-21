# -*- coding: utf-8 -*-
# @File    : train.py
# @Author  : Gaoli Chen
# @Time    : 2021/08/21
# @Desc    :

from datetime import datetime
import tensorflow.keras as keras

from dataclasses import dataclass
from couplet.dataset import CoupletDataGenerator
from couplet.couplet_generator import CoupletGenerator
from couplet.model import build_model

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
    input_dim: int = 64
    learning_rate: float = 5e-5
    model_name: str = 'bert-base-chinese'
    save_dir: str = '.'
    train_tsv: str = os.path.join(current, './datasets/train.tsv')
    dev_tsv: str = os.path.join(current, './datasets/dev.tsv')
    test_tsv: str = os.path.join(current, './datasets/test.tsv')

class CoupletGenCallback(keras.callbacks.Callback):
  def __init__(self, couplet_gen, file_name = None):
    super(CoupletGenCallback, self).__init__()

    self.gen = couplet_gen
    self.file_name = file_name

  def _print(self, content):
    print(content)
    if not self.file_name is None:
      with open(self.file_name, 'a') as f:
        print(content, file = f)
  
  def _generate(self):
    print()
    print('generating test couplets...')

    couplets = [u'愿景天成无墨迹',
                u'日近京都多俊迈',
                u'请自今指海为盟,告我先生,所不同心如此水',
                u'通揽八方客',
                u'美稼秋登万石',
                u'上海自来水来自海上']

    for couplet in couplets:
        self._print(couplet + '\t' + self.gen.make_couplet(couplet))

  def on_train_batch_end(self, batch, logs = None):
    if (batch + 1) % 1000 == 0:
      self._generate()
  
  def on_epoch_end(self, epoch, logs = None):
    self._generate()

def train(settings):
    tokenizer = tokenizer_from_pretrained(settings.model_name)

    tr_data = CoupletDataGenerator(settings.train_tsv, tokenizer = tokenizer,
                                input_dim = settings.input_dim,
                                batch_size = settings.batch_size, shuffle = True)
    
    val_data = CoupletDataGenerator(settings.dev_tsv, tokenizer = tokenizer,
                                input_dim = settings.input_dim,
                                batch_size = settings.batch_size)

    print(tr_data.first_texts[:10])
    print(tr_data.second_texts[:10])
    for feature, label in tr_data:
        print('feature:')
        for i in range(2):
            print('input_ids:', feature['input_ids'][i])
            print('label:',label[i])
            print(tokenizer.decode(feature['input_ids'][i]))
            print(tokenizer.decode(label[i]))
            print('attention_mask:', feature['attention_mask'][i])
        break
        
    model = build_model(settings)
    print(model.summary())

    gen = CoupletGenerator(model = model, tokenizer = tokenizer, input_dim = settings.input_dim)

    timestr = datetime.now().strftime('%y-%m-%d-%H')
    output_path = os.path.join(settings.save_dir, f'{model.name}_{timestr}_output.txt')
    print('training...')
    hist = model.fit(tr_data,
                     validation_data = val_data,
                     epochs = settings.epochs,
                     callbacks = [CoupletGenCallback(gen, output_path)])

def eval(settings, checkpoint_path):
    tokenizer = tokenizer_from_pretrained(settings.model_name)
    config = config_from_pretrained(settings.model_name)
    model = BertModel(config, model_head = 'lm', causal_attention = True, name = 'bert')
    model.load_weights(checkpoint_path)
    gen = CoupletGenCallback(model = model, tokenizer = tokenizer, input_dim = settings.input_dim)

    while True:
        cmd = input('Enter a couplet or quit(q):')
        if cmd == 'q':
            break
        else:
            print(gen.make_couplet(cmd))
    

if __name__ == "__main__":
    settings = Settings()
    
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
