
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datasets import DataDownloader, DataGenCreator
from models import CnnModelBuilder, PretrainedModelBuilder, model_mapping

@dataclass
class Settings(object):

    # training settings
    epochs: int = 5
    batch_size: int = 32
    val_split: float = 0.2
    data_augment: bool = True
    seed: int = 13421

    # model settings

    #model_name: str = 'cnn'
    #model_name: str = 'resnet50'
    model_name: str = 'resnet50V2'
    first_turning_layer: int = 11

    # image size
    image_height: int = 150
    image_width: int = 150
    
    # data setttings
    data_dir: str = '.'
    save_dir: str = '.'
    train_download_path: str = ''
    test_download_path: str = ''

def show_generator_samples(gen):
    class_names = ['cat', 'dog']
    plt.figure(figsize=(10, 10))
    for images, labels in gen:
        print(images.shape, labels.shape)
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(class_names[labels[i].astype('uint8')])
            plt.axis('off')
        break

def plot_hist(histgram, model_name):
    if histgram is None:
        return

    hist = histgram.history
    x = list(range(1, len(hist['acc']) + 1))
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, hist['acc'])
    plt.plot(x, hist['val_acc'])
    plt.legend(['train', 'validation'])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')

    plt.subplot(1, 2, 2)
    plt.plot(x, hist['loss'])
    plt.plot(x, hist['val_loss'])
    plt.legend(['train', 'validation'])
    plt.ylabel('loss')
    plt.xlabel('epochs')

    plt.suptitle(f'{model_name} model')

def run(settings, preview_only = False):
    # prepare dataset
    downloader = DataDownloader(settings)
    downloader.download()
    tr_datagen, val_datagen = DataGenCreator(settings).create()
    #show_generator_samples(tr_datagen)

    # build model
    if settings.model_name in model_mapping:
        model_builder = PretrainedModelBuilder(settings, top_layers = [256])
    else:
        model_builder = CnnModelBuilder(settings, conv_layers=[[16, 16], [32], [], [64], []], dense_layers = [256])

    model = model_builder.build()
    print(model.summary())

    if preview_only:
        return None

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # train
    hist = model.fit(tr_datagen, validation_data = val_datagen,
        batch_size = settings.batch_size, epochs = settings.epochs)
    return hist

if __name__ == '__main__':
    train_file_url = 'https://storage.googleapis.com/kagglesdsdata/competitions/5441/38425/train.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1635749634&Signature=W6MdzagFAfB0ww9A8U5fR5sJe8ScjTCxdvKtovOREnJaPy%2FwIvLDNmEt4y4CRqpjxq%2FqPXppQT%2B1koVqj1qe36oklfffu5yYke4gECQIloR%2FuRUIBUvweEIKGOJIah0faK0INjgUVWSXv9yaUn4Ar5%2BpyxqLLU4pGs14XAyHyoByNotloWf1MP%2FED1hNGY45e80rDVr%2BAYdaVQc1u0RLAw%2B4Fsa%2BD2ujbWmomY6Pmp1gudSZIZ6K0yxnCiMxEDa3R1ovY5T3c%2BgJixoy9Ls1G4JGXS2dsAxNAtQ8gihC1ToEiK%2F7ZIT%2F1LYnmw8rxZY8E%2BwU4XJ0wljH83PfVUS9iA%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.zip'
    test_file_url = 'https://storage.googleapis.com/kagglesdsdata/competitions/5441/38425/test.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1635749907&Signature=W2%2BZWFnd8MyabMUn8xoN8YkPMq%2BmILaJSw5leNfHt%2BtxqXdmLC2%2BDFpA4kAxQdMvGHve4W9U2tMHItpawlnoz6s74GFgW9rZ58lOjhl%2BNxJ09o045oaF4vASld9a0mAmPJMW2K3Y6GfUORTighuc4pNxpeKvUvVQTwHjzsn11Zr8xmU1uS0D0%2Fx14bEOd%2Bdiopg0AphUk4ANLG7qfOj5OOIFCL6UtQh4e7gwjB45YBZjsMDlUo9Cspjcf2yEYcdU8Z8R4viSJn82L1Ch2w0hw6piL08Io5N72BgEjvEaMo42n3CRk4B9fKdXSvSMHUFUDp1woAVOjO6xVBSHDv53XQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dtest.zip'
    #model_names = ['VGG16', 'VGG19', 'ResNet50', 'ResNet50V2', 'Xception', 'InceptionV3', 'InceptionResNetV2']

    model_name = 'custom cnn'
    settings = Settings(data_dir = '.',
        model_name = model_name,
        train_download_path = train_file_url, 
        test_download_path = test_file_url)

    hist = run(settings)
    plot_hist(hist, settings.model_name)