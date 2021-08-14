import random
from keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras


class PoetDataGenerator(keras.utils.Sequence):
  def __init__(self, texts, input_dim, batch_size, tokenizer):
    self.input_dim = input_dim
    self.texts = texts
    self.indices = list(range(len(texts)))
    self.batch_size = batch_size
    self.tokenizer = tokenizer
    
    self.on_epoch_end()

  def __len__(self):
    return (len(self.texts) + self.batch_size - 1) // self.batch_size
  
  def __getitem__(self, index):
    last_pos = min(len(self.texts), (index + 1) * self.batch_size)
    all_input_ids = self.tokenizer.encode(self.texts[index * self.batch_size : last_pos])
    
    input_ids = [id_list[:-1] for id_list in all_input_ids]
    label_ids = [id_list[1:] for id_list in all_input_ids]

    input_ids = pad_sequences(input_ids, maxlen = self.input_dim, padding = 'post', truncating = 'post')
    label_ids = pad_sequences(label_ids, maxlen = self.input_dim, padding = 'post', truncating = 'post')
    attention_mask = (input_ids != 0).astype('int32')
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, label_ids
  
  def on_epoch_end(self):
    random.shuffle(self.indices)


