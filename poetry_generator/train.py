
from datetime import datetime
import tensorflow.keras as keras
from dataclasses import dataclass
from poetry_generator.dataset import PoetDataGenerator
from poetry_generator.poetry_generator import PoetryGenerator
from poetry_generator.model import build_model

import sys
import os

#current = os.path.dirname(os.path.realpath(__file__))
#print(f'current={current}')
#sys.path.append(os.path.join(current, '../../simplebert/src'))
#sys.path.append(os.path.join(current, '..'))

from datasets.poetry import ChinesePoetry
from simplebert.tokenizers import tokenizer_from_pretrained


@dataclass
class Settings(object):
    epochs: int = 5
    batch_size: int = 32
    input_dim: int = 64
    learning_rate: float = 5e-5
    model_name: str = 'bert-base-chinese'
    val_split: float = 0.1
    save_dir: str = '.'
    download_size: float = 1.0

class RandomPoetCallback(keras.callbacks.Callback):
  def __init__(self, poetry_gen, file_name = None):
    super(RandomPoetCallback, self).__init__()

    self.gen = poetry_gen
    self.file_name = file_name

  def _print(self, content):
    print(content)
    if not self.file_name is None:
      with open(self.file_name, 'a') as f:
        print(content, file = f)
  
  def on_epoch_end(self, epoch, logs = None):
    print('generating random poetries...')

    for i in range(5):
      self._print(f'poetry {i+1} ===============')
      poet = self.gen.random(topk = 5)
      self._print(poet)
    
    heads = u'好好学习天天向上'
    self._print(f'head with {heads} =======')
    self._print(self.gen.new_peot_with_heads(heads))

    start = u'床前明月光,'
    self._print(f'start with {start} =======')
    self._print(self.gen.new_peot_with_start(start))

    start = u'万水千山总是情,'
    self._print(f'start with {start} =======')
    self._print(self.gen.new_peot_with_start(start))

    start = u'天若有情天亦老,'
    self._print(f'start with {start} =======')
    self._print(self.gen.new_peot_with_start(start))

def train(settings):
    cp = ChinesePoetry()
    tr_texts, val_texts = cp.load_dataset('song',
                                          download_size = settings.download_size,
                                          whole_poetry = True,
                                          maxlen = settings.input_dim - 2,
                                          val_split = settings.val_split)
    
    tr_texts2, val_texts2 = cp.load_dataset('tang',
                                          download_size = settings.download_size,
                                          whole_poetry = True,
                                          maxlen = settings.input_dim - 2,
                                          val_split = settings.val_split)
    tr_texts.extend(tr_texts2)
    val_texts.extend(val_texts2)
    del tr_texts2
    del val_texts2
    
    tokenizer = tokenizer_from_pretrained(settings.model_name)

    tr_data = PoetDataGenerator(tr_texts, tokenizer = tokenizer,
                                input_dim = settings.input_dim,
                                batch_size = settings.batch_size)
    
    val_data = PoetDataGenerator(val_texts, tokenizer = tokenizer,
                                input_dim = settings.input_dim,
                                batch_size = settings.batch_size)
        
    model = build_model(settings)
    print(model.summary())

    pg = PoetryGenerator(model = model, tokenizer = tokenizer, input_dim = settings.input_dim)

    timestr = datetime.now().strftime('%y-%m-%d-%H')
    output_path = os.path.join(settings.save_dir, f'{model.name}_{timestr}_output.txt')
    
    hist = model.fit(tr_data,
                     validation_data = val_data,
                     epochs = settings.epochs,
                     callbacks = [RandomPoetCallback(pg, output_path)])
    

if __name__ == "__main__":
    settings = Settings()
    train(settings)
