import random
from keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras


def read_tsv(file_path):
  first = []
  second = []
  with open(file_path, encoding = 'utf-8') as f:
    for line in f.readlines():
      parts = line.split('\t')
      if len(parts) != 2:
        print(f'line = {line}, file={file_path}')
        continue
      first.append(parts[0].strip())
      second.append(parts[1].strip())

  return first, second


class CoupletDataGenerator(keras.utils.Sequence):
  def __init__(self, file_path, input_dim, batch_size, tokenizer, shuffle = False):
    self.input_dim = input_dim
    self.first_texts, self.second_texts = read_tsv(file_path)
    self.indices = list(range(len(self.first_texts)))
    self.batch_size = batch_size
    self.tokenizer = tokenizer
    self.shuffle = shuffle
    
    self.on_epoch_end()

  def __len__(self):
    return (len(self.first_texts) + self.batch_size - 1) // self.batch_size
  
  def __getitem__(self, index):
    last_pos = min(len(self.first_texts), (index + 1) * self.batch_size)

    token_ids1 = self.tokenizer.encode(self.first_texts[index * self.batch_size : last_pos])
    token_ids2 = self.tokenizer.encode(self.second_texts[index * self.batch_size : last_pos])
    
    input_ids = [ids1 + ids2[1:-1] for ids1, ids2 in zip(token_ids1, token_ids2)]
    label_ids = [[self.tokenizer.pad_token_id] * (len(ids1) - 1) + ids2[1:] for ids1, ids2 in zip(token_ids1, token_ids2)]

    input_ids = pad_sequences(input_ids, maxlen = self.input_dim, padding = 'post', truncating = 'post')
    label_ids = pad_sequences(label_ids, maxlen = self.input_dim, padding = 'post', truncating = 'post')
    attention_mask = (input_ids != self.tokenizer.pad_token_id).astype('int32')
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, label_ids
  
  def on_epoch_end(self):
    if self.shuffle:
      random.shuffle(self.indices)


