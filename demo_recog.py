import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
import string
import numpy as np
import random
import os
# from matplotlib import pyplot as plt
import cv2

# Custom metris / losses
from custom import cat_acc, cce, plate_acc, top_3_k

# For measuring inference time
from timeit import default_timer as timer

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


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


def visualize_predictions(path='val_set/imgs/', shuffle=False, print_time=False):
    # generate samples and plot
    val_imgs = [os.path.join(path, f) for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))]
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
        # Show plate
        im = cv2.putText(im, f'{plate_str}',
                         org=(5, 20),
                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale=0.7,
                         color=(0, 255, 0),
                         lineType=1,
                         thickness=3
                         #  bottomLeftOrigin=True
                         )

        print(f'Plate: {plate}')
        print(f'Confidence: {probs}')
        im_to_show = cv2.resize(im, dsize=(
            300, 400), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(f'plates', im_to_show)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return


if __name__ == "__main__":
    custom_objects = {
        'cce': cce,
        'cat_acc': cat_acc,
        'plate_acc': plate_acc,
        'top_3_k': top_3_k
    }

    model = tf.keras.models.load_model(
        'models/model_4m.h5', custom_objects=custom_objects)

    visualize_predictions(path='val_set/imgs/')
    cv2.destroyAllWindows()
