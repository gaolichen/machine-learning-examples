import numpy as np
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt

def image_to_tensor(path, max_dim = 512):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype = np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]

    return PIL.Image.fromarray(tensor)

def image_to_variable(path):
    pass

def imshow(image, title = None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis = 0)

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

if __name__ == '__main__':
    style_path = './style.jpg'
    img = image_to_tensor(style_path)
    print(img.shape)
    #imshow(img)

    img = tensor_to_image(img)

