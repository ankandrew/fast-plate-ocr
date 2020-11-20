import tensorflow as tf
import string
import numpy as np
import random
import os
import cv2
from argparse import ArgumentParser

from tensorflow.python.keras.activations import softmax
# import statistics as stat
# Custom metris / losses
from custom import cat_acc, cce, plate_acc, top_3_k

# For measuring inference time
from timeit import default_timer as timer

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def check_low_conf(probs, thresh=.3):
    '''
    Add position of chars. that are < thresh
    '''
    return [i for i, prob in enumerate(probs) if prob < thresh]


@tf.function
def predict_from_array(img, model):
    pred = model(img, training=False)
    return pred


def probs_to_plate(prediction):
    prediction = prediction.reshape((7, 37))
    probs = np.max(prediction, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    plate = list(map(lambda x: alphabet[x], prediction))
    return plate, probs


def visualize_predictions(model, imgs_path='benchmark/imgs/', shuffle=False, print_time=False):
    # generate samples and plot
    val_imgs = [os.path.join(imgs_path, f) for f in os.listdir(imgs_path)
                if os.path.isfile(os.path.join(imgs_path, f))]
    if shuffle:
        random.shuffle(val_imgs)
    for im in val_imgs:
        # Start time
        start = timer()
        #
        im = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
        # resize dsize (w, h) -> (140, 70)
        img = cv2.resize(im, dsize=(140, 70), interpolation=cv2.INTER_LINEAR)
        img = img[np.newaxis, ..., np.newaxis] / 255.
        img = tf.constant(img, dtype=tf.float32)
        start_inference = timer()
        prediction = predict_from_array(img, model).numpy()
        end_inference = timer()
        plate, probs = probs_to_plate(prediction)
        plate_str = ''.join(plate)
        # End timer
        end = timer()
        #
        if print_time:
            delta_time = end - start
            fps = 1 / delta_time
            delta_time_inference = end_inference - start_inference
            fps_inference = 1 / delta_time_inference
            # Timing con y sin preprocessing (rescale, add axis, cast)
            print(
                f'Time taken, including reading, resize, prediction is\t{delta_time:.5f}\tFPS: {fps:.0f}\
                \nTime taken just inference: {delta_time_inference:.5f}\tFPS: {fps_inference:.0f}', flush=True)
        print(f'Plate: {plate_str}', flush=True)
        print(f'Confidence: {probs}', flush=True)
        im_to_show = cv2.resize(im, dsize=(
            140 * 3, 70 * 3), interpolation=cv2.INTER_LINEAR)
        # Converting to BGR for color text
        im_to_show = cv2.cvtColor(im_to_show, cv2.COLOR_GRAY2RGB)
        # Avg. probabilities
        avg_prob = np.mean(probs) * 100
        # Agrego borde negro para que se vea mejor
        cv2.putText(
            im_to_show, f'{plate_str}  {avg_prob:.{2}f}%',
            org=(5, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 0),
            lineType=1,
            thickness=6
            #  bottomLeftOrigin=True
        )
        cv2.putText(
            im_to_show, f'{plate_str}  {avg_prob:.{2}f}%',
            org=(5, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            lineType=1,
            thickness=2
            #  bottomLeftOrigin=True
        )
        # Display character with low confidence
        low_conf_chars = 'Low conf. on: ' + \
            ' '.join([plate[i] for i in check_low_conf(probs, thresh=.15)])
        cv2.putText(
            im_to_show, low_conf_chars,
            org=(5, 200),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 220),
            lineType=1,
            thickness=2
            #  bottomLeftOrigin=True
        )
        cv2.imshow(f'plates', im_to_show)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_path",
                        default='models/m1_93_vpa_2.0M-i2.h5', metavar="FILE",
                        help="Path del modelo, predeterminado es el model_4m.h5")
    parser.add_argument("-i", "--imgs-dir", dest="imgs_dir",
                        default='benchmark/imgs/', metavar="FILE",
                        help="Path de la carpeta contenedora de las imagenes")
    parser.add_argument("-t", "--time", dest='do_time', action='store_true',
                        help="Mostrar el tiempo de inferencia (c/procesamiento)")

    args = parser.parse_args()
    custom_objects = {
        'cce': cce,
        'cat_acc': cat_acc,
        'plate_acc': plate_acc,
        'top_3_k': top_3_k,
        'softmax': softmax
    }

    alphabet = string.digits + string.ascii_uppercase + '_'
    model = tf.keras.models.load_model(
        args.model_path, custom_objects=custom_objects)

    visualize_predictions(model, imgs_path=args.imgs_dir,
                          print_time=args.do_time)
    cv2.destroyAllWindows()
