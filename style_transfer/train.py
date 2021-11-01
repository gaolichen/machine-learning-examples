
from dataclasses import dataclass
from models import StyleTransfer
from datasets import imshow

@dataclass
class Settings(object):

    # training settings
    epochs: int = 5
    steps_per_epoch: int = 10

    style_weight: float = 1e2
    content_weight: float = 1e2

    model_name: str = 'VGG16'


def run(settings, style_image_path, content_image_path, style_layer_names, content_layer_names):
    transfer = StyleTransfer(settings,
        style_layer_names = style_layer_names,
        content_layer_names = content_layer_names)

    image = transfer.train(style_image_path, content_image_path)
    imshow(image)

if __name__ == '__main__':
    settings = Settings()
    style_image_path = ''
    content_image_path = ''
    style_layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', ]
    content_layer_names = ['block5_conv2']
    run(settings, style_image_path, content_image_path, style_layer_names, content_layer_names)