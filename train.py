import string

import numpy as np
# import skimage
import tensorflow as tf
import tensorflow.keras.metrics
from tensorflow.keras.activations import softmax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Dropout,
                                     GlobalAveragePooling2D, Input, MaxPool2D)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array, load_img)
from argparse import ArgumentParser
from custom import cat_acc, cce, plate_acc, top_3_k
from layer_blocks import block_bn
import pandas as pd
from extra_augmentation import cut_out, blur
import matplotlib.pyplot as plt
import os


def modelo_2m(h, w):
    '''
    Modelo de 2 millones de parametros
    '''
    input_tensor = Input((h, w, 1))
    # Backbone
    x, _ = block_bn(input_tensor)
    x, _ = block_bn(x, k=3, n_c=32, s=1, padding='same')
    x, _ = block_bn(x, k=3, n_c=32, s=1, padding='same')
    x, _ = block_bn(x, k=1, n_c=64, s=1, padding='same')
    x = MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='same')(x)
    x, _ = block_bn(x, k=3, n_c=64, s=1, padding='same')
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding='same')
    x, _ = block_bn(x, k=1, n_c=128, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding='same')
    x, _ = block_bn(x, k=3, n_c=128, s=1, padding='same')
    x, _ = block_bn(x, k=1, n_c=256, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn(x, k=3, n_c=256, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn(x, k=1, n_c=512, s=1, padding='same')
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x, _ = block_bn(x, k=1, n_c=1024, s=1, padding='same')
    # Clasificador
    x = GlobalAveragePooling2D()(x)
    # dropout for more robust learning
    x = Dropout(0.5)(x)
    x1 = Dense(units=37)(x)
    x2 = Dense(units=37)(x)
    x3 = Dense(units=37)(x)
    x4 = Dense(units=37)(x)
    x5 = Dense(units=37)(x)
    x6 = Dense(units=37)(x)
    x7 = Dense(units=37)(x)
    # Softmax act.
    x1 = Activation(softmax)(x1)
    x2 = Activation(softmax)(x2)
    x3 = Activation(softmax)(x3)
    x4 = Activation(softmax)(x4)
    x5 = Activation(softmax)(x5)
    x6 = Activation(softmax)(x6)
    x7 = Activation(softmax)(x7)
    x = Concatenate()([x1, x2, x3, x4, x5, x6, x7])
    return Model(inputs=input_tensor, outputs=x)


def data_aug(do_blur=False, do_cut_out=False):
    if do_blur:
        def pf(img):
            if np.random.rand() > .5:
                return blur(img)
            else:
                return img
    elif do_cut_out:
        def pf(img):
            if np.random.rand() > .5:
                return cut_out(img)
            else:
                return img
    elif do_blur and do_cut_out:
        def pf(img):
            '''
            Aplicar 33.3% de las veces cut_out, blur
            y la imagen aumentada normalmente
            '''
            rand = np.random.rand()
            if rand < 1 / 3:
                return cut_out(img)
            elif rand > 1 / 3 and rand < 2 / 3:
                return blur(img)
            else:
                return img
    else:
        pf = None
    datagen = ImageDataGenerator(
        rescale=1 / 255.,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.10,
        brightness_range=(0.5, 1.5),
        shear_range=8,
        zoom_range=0.12,
        preprocessing_function=pf
    )

    datagen_validator = ImageDataGenerator(
        rescale=1 / 255.
    )

    return datagen, datagen_validator


def txt_to_numpy(anots_path, h, w):
    # Ejemplos de interpolacion:
    #  ["bilinear", "bicubic", "nearest", "lanczos", "box", "hamming"]
    interpolation = "bilinear"
    df = pd.read_csv(anots_path, sep='\t', names=['path', 'plate'])
    preprocess_df(df)
    # df.path = df.path.str.replace('imgs', 'val_imgs')
    x, y = df_to_x_y(df, target_h=h, target_w=w, interpolation=interpolation)
    return x, y


def df_to_x_y(df, target_h=70, target_w=140, interpolation="bilinear"):
    '''
    Loads all the imgs to memory (by col name='path')
    with the corresponding y labels (one-hot encoded)
    '''
    # Load all images in numpy array
    x_imgs = []
    for img in df.path.values:
        img = load_img(img, color_mode="grayscale", target_size=(
            target_h, target_w), interpolation=interpolation)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        x_imgs.append(img)
    x_imgs = np.vstack(x_imgs)

    y_imgs = []
    for one_hot in df.labels.values:
        # label = np.expand_dims(one_hot, axis=0)
        one_hot = one_hot.reshape((259))
        y_imgs.append(one_hot)
    y_imgs = np.vstack(y_imgs)

    return x_imgs, y_imgs


def preprocess_df(df):
    # Pad 6-len plates with '_'
    df.loc[df.plate.str.len() == 6, 'plate'] += '_'

    def string_vectorizer(plate_str):
        alphabet = string.digits + string.ascii_uppercase + '_'
        vector = [[0 if char != letter else 1 for char in alphabet]
                  for letter in plate_str]
        return vector

    # Convert to one-hot
    df['labels'] = df.plate.apply(lambda x: np.array(string_vectorizer(x)))


def save_accuracy_plot(save_name, history, metric='plate_acc'):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'model {metric}')
    plt.ylabel(f'{metric}')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    plt.savefig(save_name)


def save_loss_plot(save_name, history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    plt.savefig(save_name)


def save_stats(save_name, history):
    with open(save_name, 'w') as out:
        for key, val in history.history.items():
            out.write(f'---{key}---\n')
            out.write(f'max\t{max(val)}\n')
            out.write(f'max\t{min(val)}\n')
            out.write(f'mean\t{np.mean(val)}\n')
            out.write(f'std\t{np.std(val)}\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-vis", "--visualizar-aug", dest="visualizar_aug",
                        action='store_true', help="Visualizar Data Augmentation (no entrenar)")
    parser.add_argument("-i", "--anotaciones", dest="anotaciones_path",
                        default='train_val_set/train_anotaciones.txt',
                        type=str, help="Path del .txt que contiene las anotaciones")
    parser.add_argument("-v", "--val-anotaciones", dest="val_anotaciones_path",
                        default='train_val_set/valid_anotaciones.txt', type=str, help="Path del .txt que contiene las anotaciones")
    parser.add_argument("-a", "--altura", dest="height",
                        default=70, type=int, help="Alto de imagen a utilizar")
    parser.add_argument("-ancho", "--ancho", dest="width",
                        default=140, type=int, help="Ancho de imagen a utilizar")
    parser.add_argument("-l", "--learning-rate", dest="lr",
                        default=1e-3, type=float, help="Valor del learning rate")
    parser.add_argument("-b", "--batch-size", dest="batch_size",
                        default=64, type=int, help="Tamaño del batch, predeterminado 1")

    parser.add_argument("-o", "--output-dir", dest="output_path",
                        default='', type=str, help="Path para guarda el modelo")
    parser.add_argument("-e", "--epochs", dest="epochs",
                        default=500, type=int, help="Cantidad de Epochs(cuantas veces se ve el dataset completo")

    parser.add_argument("-ca", "--cut-out", dest="cut_out",
                        action='store_true', help="Aplicar cut out a las imagenes, adicionalmente al Augmentation normal")

    parser.add_argument("-ba", "--blur", dest="blur",
                        action='store_true', help="Aplicar blur a las imagenes, adicionalmente al Augmentation normal")
    parser.add_argument("-g", "--graficos", dest="graphics",
                        action='store_true', help="Guardar imagenes graficos de entrenamiento (loss, cat_acc, etc...)")

    args = parser.parse_args()

    if args.visualizar_aug:
        datagen, _ = data_aug(do_blur=args.blur, do_cut_out=args.cut_out)
        # Por defecto se aumenta las imagenes de benchmark/imgs
        x_imgs, _ = txt_to_numpy(
            'benchmark/anotaciones.txt', args.height, args.width)
        aug_generator = datagen.flow(x_imgs, batch_size=1, shuffle=True)

        fig, ax = plt.subplots(nrows=5, ncols=5)
        for row in ax:
            for col in row:
                img = aug_generator.next() * 255.
                col.imshow(np.squeeze(img), cmap='gray')
        # show the figure
        plt.show()
    else:
        # Entrenar
        modelo = modelo_2m(args.height, args.width)
        modelo.compile(loss=cce, optimizer=tf.keras.optimizers.Adam(args.lr),
                       metrics=[cat_acc, plate_acc, top_3_k])

        datagen, datagen_validator = data_aug(args.cut_out, args.blur)
        # Pre-procesar .txt -> arrays de numpy listo para model.fit(...)
        x_imgs, y_imgs = txt_to_numpy(
            args.anotaciones_path, args.height, args.width)
        train_generator = datagen.flow(
            x_imgs, y_imgs, batch_size=args.batch_size, shuffle=True)
        train_steps = train_generator.n // train_generator.batch_size

        if args.val_anotaciones_path is not None:
            X_test, y_test = txt_to_numpy(
                args.val_anotaciones_path, args.height, args.width)
            validation_generator = datagen_validator.flow(
                X_test, y_test, batch_size=args.batch_size, shuffle=False)
            validation_steps = validation_generator.n // validation_generator.batch_size
        else:
            validation_generator = None
            validation_steps = None

        callbacks = [
            # Si en 5 epochs no baja val_loss, disminuir por un
            # factor de 0.3 el lr (cambiar la patience para esperar mas antes de bajarlo)
            ReduceLROnPlateau('val_loss'),
            # Parar de entrenar si val_cat_acc no aumenta en 50 epochs
            EarlyStopping(monitor='val_cat_acc', patience=50)
        ]

        history = modelo.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=args.epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        # modelo.save(f'model2m_trained_{args.epochs}.h5')
        modelo.save(os.path.join(args.output_path, 'model2m_trained.h5'))

        if args.graphics:
            # Save graphs to current dir.
            save_accuracy_plot('top3_acc.png', history, metric='top_3_k')
            save_accuracy_plot('cat_acc.png', history, metric='cat_acc')
            save_accuracy_plot('plate_acc.png', history, metric='plate_acc')
            save_loss_plot('loss.png', history)
            save_stats('stats.txt', history)
