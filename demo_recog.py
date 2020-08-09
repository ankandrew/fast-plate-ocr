import tensorflow as tf
import string
import numpy as np
import random
import os
import cv2
from argparse import ArgumentParser
import statistics as stat
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


def predict_from_array(img):
    # img = np.expand_dims(img, axis=0)
    img = img[np.newaxis, ..., np.newaxis]
    img = img.astype("float32") / 255
    # Make prediction
    return model.predict(img)


def probs_to_plate(prediction):
    alphabet = string.digits + string.ascii_uppercase + '_'
    i = 0
    plate = []
    probs = []
    for x in range(7):
        id = np.argmax(prediction[0][i:i + 37])
        probs.append(np.max(prediction[0][i:i + 37]))
        plate.append(alphabet[id])
        i += 37
    return plate, probs


def visualize_predictions(model, imgs_path='val_set/imgs/', shuffle=False, print_time=False):
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
        im = cv2.resize(im, dsize=(140, 70), interpolation=cv2.INTER_LINEAR)
        prediction = predict_from_array(im)
        plate, probs = probs_to_plate(prediction)
        plate_str = ''.join(plate)
        # End timer
        end = timer()
        #
        if print_time:
            print(
                f'Time taken, including reading, resize, prediction is\t{end - start}'
            )
        print(f'Plate: {plate_str}')
        print(f'Confidence: {probs}')
        im_to_show = cv2.resize(im, dsize=(
            140 * 3, 70 * 3), interpolation=cv2.INTER_LINEAR)
        # Converting to BGR for color text
        im_to_show = cv2.cvtColor(im_to_show, cv2.COLOR_GRAY2RGB)
        # Avg. probabilities
        avg_prob = stat.mean(probs) * 100
        # Show plate
        im_to_show = cv2.putText(
            im_to_show, f'{plate_str}  {avg_prob:.{2}f}%',
            org=(5, 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 100, 0),
            lineType=1,
            thickness=2
            #  bottomLeftOrigin=True
        )
        # Display character with low confidence
        low_conf_chars = 'Low conf. on: ' + \
            ' '.join([plate[i] for i in check_low_conf(probs, thresh=.15)])
        im_to_show = cv2.putText(
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
                        default='models/model_4m.h5', metavar="FILE",
                        help="Path del modelo, predeterminado es el model_4m.h5")

    args = parser.parse_args()
    custom_objects = {
        'cce': cce,
        'cat_acc': cat_acc,
        'plate_acc': plate_acc,
        'top_3_k': top_3_k
    }

    model = tf.keras.models.load_model(
        args.model_path, custom_objects=custom_objects)

    visualize_predictions(model, imgs_path='val_set/imgs/')
    cv2.destroyAllWindows()
