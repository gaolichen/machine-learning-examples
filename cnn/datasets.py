import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataDownloader(object):
    def __init__(self, settings):
        self.data_dir = settings.data_dir
        self.train_download_path = settings.train_download_path
        self.test_download_path = settings.test_download_path
    
    def _download_file(self, file_name, download_path):
        if not os.path.exists(os.path.join(self.data_dir, file_name)):
            keras.utils.get_file(file_name, download_path, extract=True, cache_dir = '.')
            print('finished downloading {}'.format(file_name))
        else:
            print('{0} already downloaded'.format(file_name))

    def download(self):
        self._download_file('train.zip', self.train_download_path)
        self._download_file('test.zip', self.test_download_path)


def get_paths_from_dir(dire):
    return [os.path.join(dire, file) for file in os.listdir(dire)]

def get_labels_from_dir(dire):
    class_names = ['cat', 'dog']
    return [str(class_names.index(file.split('.')[0])) for file in os.listdir(dire)]

def get_preprocessing_function(model_name):
    if model_name.startswith('ResNet'):
        return keras.applications.resnet.preprocess_input
    elif model_name == 'Xception':
        return keras.applications.xception.preprocess_input
    elif model_name == 'InceptionV3':
        return keras.applications.inception_v3.preprocess_input
    elif model_name == 'VGG19':
        return keras.applications.vgg19.preprocess_input
    elif model_name == 'VGG16':
        return keras.applications.vgg16.preprocess_input
    elif model_name == 'InceptionResNetV2':
        return keras.applications.inception_resnet_v2.preprocess_input
    else:
        return None

class DataGenCreator(object):
    def __init__(self, settings):
        self.data_dir = settings.data_dir
        self.preprocessing_function = get_preprocessing_function(settings.model_name)
        self.val_split = settings.val_split
        self.data_augment = settings.data_augment
        self.seed = settings.seed
        self.image_size = (settings.image_height, settings.image_width)
        self.batch_size = settings.batch_size
    
    def _create(self, data_generator, df):
        return data_generator.flow_from_dataframe(df, x_col = 'filename', y_col = 'label',
            target_size = self.image_size, batch_size = self.batch_size,
            class_mode = 'binary', shuffle=False)
    
    def create(self):
        """
        """
        train_dir = os.path.join(self.data_dir, './datasets/train')
        df = pd.DataFrame({'filename' : get_paths_from_dir(train_dir), 'label' : get_labels_from_dir(train_dir)})
        
        val_df = df.sample(frac = self.val_split, random_state = self.seed)
        train_df = df.drop(val_df.index)

        val_data_gen = ImageDataGenerator(rescale = 1.0/255, preprocessing_function = self.preprocessing_function)
        if self.data_augment:
            tr_data_gen = ImageDataGenerator(
                rescale = 1.0/255,
                rotation_range=30,
                #width_shift_range=0.1,
                #height_shift_range=0.1,
                #shear_range = 0.2,
                #zoom_range = 0.2,
                #samplewise_center = True,
                horizontal_flip = True,
                preprocessing_function = self.preprocessing_function)
        else:
            tr_data_gen = ImageDataGenerator(rescale = 1.0/255, preprocessing_function = self.preprocessing_function)
        
        return self._create(tr_data_gen, train_df), self._create(val_data_gen, val_df)
