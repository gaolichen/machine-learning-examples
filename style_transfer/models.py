import tensorflow as tf
import tensorflow.keras as keras
from datasets import image_to_tensor

class FeatureExtractor(keras.models.Model):
    def __init__(self, base_model, layer_names):
        super(FeatureExtractor, self).__init__()

        self.base_model = base_model
        self.layer_names = layer_names

        outputs = [base_model.get_layer(layer_name).output for layer_name in layer_names]
        self.model = keras.models.Model([base_model.input], outputs)
        
    def call(self, input_tensor):
        outputs = self.model(input_tensor)

        return {layer_name: output for output, layer_name in zip(outputs, self.layer_names)}

model_mapping = dict([('VGG16', keras.applications.VGG16), 
    ('VGG19', keras.applications.VGG19),
    ('ResNet50', keras.applications.ResNet50), 
    ('ResNet50V2', keras.applications.ResNet50V2), 
    ('Xception', keras.applications.Xception), 
    ('InceptionV3', keras.applications.InceptionResNetV2), 
    ('InceptionResNetV2', keras.applications.InceptionResNetV2)])

class StyleTransfer(object):
    def __init__(self, settings, style_layer_names, content_layer_names):
        super(StyleTransfer, self).__init__()

        self.settings = settings
        self.style_weight = settings.style_weight
        self.content_weight = settings.content_weight

        base_model = self._load_pretrained_model(settings.model_name)
        print(base_model.summary())
        self.style_layer_names = style_layer_names
        self.content_layer_names = content_layer_names
        self.feature_extractor = FeatureExtractor(base_model, self.style_layer_names + self.content_layer_names)

        self.optimizer = keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
       
    def _get_outputs(self, image):
        outputs = self.feature_extractor(image)
        style_outputs = [self._gram_matrix(outputs[layer_name]) for layer_name in self.style_layer_names]
        content_outputs = [outputs[layer_name] for layer_name in self.content_layer_names]

        return style_outputs, content_outputs
    
    @staticmethod
    def _gram_matrix(x):
        res = tf.linalg.einsum('bijc,bijd->bcd', x, x)
        shape = tf.shape(res)
        n = tf.cast(shape[1] * shape[2], tf.float32)
        return res / n
    
    @staticmethod
    def _loss(output, target):
        loss = tf.add_n(tf.reduce_mean((x - y)**2) for x, y in zip(output, target))
        return loss / len(output)

    @staticmethod
    def _load_pretrained_model(model_name, random_weights = False):
        if random_weights:
            return model_mapping[model_name](include_top=False, weights = None)
        else:
            return model_mapping[model_name](include_top=False, weights = 'imagenet')
    
    @staticmethod
    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def train_step(self, image, style_target, content_target):
        with tf.GradientTape() as tape:
            style_output, content_output = self._get_outputs(image)
            loss = self.style_weight * self._loss(style_output, style_target) + self.content_weight * self._loss(content_output, content_target)

        grad = tape.gradient(loss, image)
#        print(f'grad.shape=', grad.shape)
#        print(grad[0,1,:])
        self.optimizer.apply_gradients([(grad, image)])
        image.assign(self.clip_0_1(image))
        return image, loss
    
    def train(self, style_image_path, content_image_path):
        # content_image.shape = (1, H, W, 3)
        content_image = image_to_tensor(content_image_path)
        style_target, _ = self._get_outputs(image_to_tensor(style_image_path))
        _, content_target = self._get_outputs(content_image)

        image = tf.Variable(content_image)
        for epoch in range(1, self.settings.epochs + 1):
            print(f'Epoch {epoch} ...')
            for step in range(self.settings.steps_per_epoch):
                image, loss = self.train_step(image, style_target, content_target)
            print(f'\tloss={loss}')
        
        return image