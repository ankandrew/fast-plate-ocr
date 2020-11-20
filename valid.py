import tensorflow as tf
import string
import numpy as np

# Custom metris / losses
from custom import cat_acc, cce, plate_acc, top_3_k
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.python.keras.activations import softmax
from argparse import ArgumentParser
import pandas as pd


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


def df_to_x_y(df, target_h=70, target_w=140):
    '''
    Loads all the imgs to memory (by col name='path')
    with the corresponding y labels (one-hot encoded)
    '''
    # Load all images in numpy array
    x_imgs = []
    for img in df.path.values:
        img = load_img(img, color_mode="grayscale", target_size=(
            target_h, target_w), interpolation="bilinear")
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        x_imgs.append(img)
    x_imgs = np.vstack(x_imgs)

    y_imgs = []
    for one_hot in df.labels.values:
        one_hot = one_hot.reshape((259))
        y_imgs.append(one_hot)
    y_imgs = np.vstack(y_imgs)

    return x_imgs, y_imgs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_path",
                        default='models/m1_93_vpa_2.0M-i2.h5', metavar="FILE",
                        help="Path del modelo, predeterminado es el model_4m.h5")
    parser.add_argument("-b", "--batch-size", dest="batch_size",
                        default=1, type=int,
                        help="Tamaño del batch, predeterminado 1")

    args = parser.parse_args()
    custom_objects = {
        'cce': cce,
        'cat_acc': cat_acc,
        'plate_acc': plate_acc,
        'top_3_k': top_3_k,
        'softmax': softmax
    }
    model = tf.keras.models.load_model(
        args.model_path, custom_objects=custom_objects)

    df_val = pd.read_csv('benchmark/anotaciones.txt',
                         sep='\t', names=['path', 'plate'])
    preprocess_df(df_val)
    x_val, y_val = df_to_x_y(df_val, target_h=70, target_w=140)
    datagen_val = ImageDataGenerator(
        rescale=1 / 255.
    )
    val_generator = datagen_val.flow(
        x_val, y_val, batch_size=args.batch_size, shuffle=False)
    model.evaluate(val_generator)
