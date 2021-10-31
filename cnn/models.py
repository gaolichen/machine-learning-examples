
import tensorflow as tf
import tensorflow.keras as keras

class ModelBuilder(object):
    def __init__(self, settings):
        self.setting = settings
        self.image_size = (settings.image_height, settings.image_width)
    
    def build(self):
        raise NotImplemented
    

class CnnModelBuilder(ModelBuilder):
    def __init__(self, settings, conv_layers = None, dense_layers = None):
        super(CnnModelBuilder, self).__init__(settings)

        self.kernal_size = (3, 3)
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers

    
    def build(self):
        model = keras.models.Sequential()
        first = True

        # build convolution layers
        for i, layers in enumerate(self.conv_layers):
            for j, channels in enumerate(layers):
                layer_name = 'conv2D_block{0}_{1}'.format(i + 1, j + 1)
                if not first:
                    model.add(keras.layers.Conv2D(channels, self.kernal_size, name = layer_name, activation = 'relu', padding = 'valid'))
                else:
                    model.add(keras.layers.Conv2D(channels, self.kernal_size, name = layer_name, activation = 'relu',
                        input_shape = self.image_size + (3, ), padding = 'same'))
                    first = False

            # add batch normalization
            if len(layers) > 0:
                model.add(keras.layers.BatchNormalization(axis = -1))            
            model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding = 'valid'))
    
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.5))

        # add dense layers
        for units in self.dense_layers:
            model.add(keras.layers.Dense(units, activation = 'relu'))
        
        # output layer.
        model.add(keras.layers.Dense(1, activation = 'sigmoid'))
        return model

model_mapping = dict([('VGG16', keras.applications.VGG16), 
    ('VGG19', keras.applications.VGG19),
    ('ResNet50', keras.applications.ResNet50), 
    ('ResNet50V2', keras.applications.ResNet50V2), 
    ('Xception', keras.applications.Xception), 
    ('InceptionV3', keras.applications.InceptionResNetV2), 
    ('InceptionResNetV2', keras.applications.InceptionResNetV2)])

class PretrainedModelBuilder(ModelBuilder):
    def __init__(self, settings, top_layers = []):
        super(PretrainedModelBuilder, self).__init__(settings)

        self.settings = settings 
        self.top_layers = top_layers
        self.first_turning_layer = settings.first_turning_layer
    
    @staticmethod
    def _set_trainable(base_model, first_trainable_layer_index):
        for layer in base_model.layers:
            layer.trainable = False
        if first_trainable_layer_index > 0:
            for layer in base_model.layers[-first_trainable_layer_index:]:
                layer.trainable = True
    
    @staticmethod
    def _load_pretrained_model(model_name, input_shape, random_weights = False):
        if random_weights:
            return model_mapping[model_name](include_top=False, input_shape = input_shape, weights = None)
        else:
            return model_mapping[model_name](include_top=False, input_shape = input_shape, weights = 'imagenet')

    def build(self):
        self.base_model = self._load_pretrained_model(self.settings.model_name, input_shape = self.image_size + (3,))
        print(self.base_model.summary())
        self._set_trainable(self.base_model, self.first_turning_layer)

        model = keras.models.Sequential()
        model.add(self.base_model)

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.5))
        for units in self.top_layers:
            model.add(keras.layers.Dense(units, activation = 'relu'))

        model.add(keras.layers.Dense(1, activation = 'sigmoid'))

        return model