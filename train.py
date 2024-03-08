"""
Script for training the License Plate OCR models.
"""
import os
import string
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import img_to_array, load_img

from augmentation import DataAugmentation
from fast_lp_ocr.custom import cat_acc, cce, plate_acc, top_3_k
from fast_lp_ocr.models import modelo_1m_cpu, modelo_2m

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


class Preprocess:
    def __init__(self, annots_path, h, w, interpolation="bilinear") -> None:
        self.height = h
        self.width = w
        self.annotations = annots_path
        #  ["bilinear", "bicubic", "nearest", "lanczos", "box", "hamming"]
        self.interpolation = interpolation
        self.alphabet = string.digits + string.ascii_uppercase + "_"
        self.df = pd.read_csv(self.annotations, sep="\t", names=["path", "plate"])
        self.__preprocess_df()

    def __call__(self):
        # Return numpy array: features, labels
        return self.__df_to_x_y()

    def __df_to_x_y(self):
        """
        Loads all the imgs to memory (by col name='path')
        with the corresponding y labels (one-hot encoded)
        """
        # Load all images in numpy array
        x_imgs = []
        for img_path in self.df.path.values:
            img = load_img(
                img_path,
                color_mode="grayscale",
                target_size=(self.height, self.width),
                interpolation=self.interpolation,
            )
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            x_imgs.append(img)
        x_imgs = np.vstack(x_imgs)
        y_imgs = [one_hot.reshape(7 * 37) for one_hot in self.df.plate.values]
        y_imgs = np.vstack(y_imgs)
        return x_imgs, y_imgs

    def __preprocess_df(self):
        # Pad 6-len plates with '_'
        self.df.loc[self.df.plate.str.len() == 6, "plate"] += "_"
        # Convert to one-hot
        self.df["labels"] = self.df.plate.apply(lambda x: np.array(self.__string_vectorizer(x)))

    def __string_vectorizer(self, plate_str):
        vector = [[0 if char != letter else 1 for char in self.alphabet] for letter in plate_str]
        return vector


class Graph:
    def __init__(self, history, output_folder) -> None:
        self.history = history
        self.output_folder = output_folder
        plt.style.use("seaborn")

    def __call__(self):
        self.__save_accuracy_plot(
            os.path.join(self.output_folder, "top3_acc.png"), metric="top_3_k"
        )
        self.__save_accuracy_plot(os.path.join(self.output_folder, "cat_acc.png"), metric="cat_acc")
        self.__save_accuracy_plot(
            os.path.join(self.output_folder, "plate_acc.png"), metric="plate_acc"
        )
        self.__save_lr_plot(os.path.join(self.output_folder, "learning_rate.png"))
        self.__save_loss_plot(os.path.join(self.output_folder, "loss.png"))
        self.__save_stats(os.path.join(self.output_folder, "stats.csv"))

    def __save_accuracy_plot(self, save_name, metric="plate_acc"):
        plt.plot(self.__smooth_curve(self.history.history[metric]))
        plt.plot(self.__smooth_curve(self.history.history[f"val_{metric}"]))
        plt.title(f"Model {metric}")
        plt.ylabel(f"{metric}")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper left")
        # plt.show()
        plt.savefig(save_name)
        # Clear points/axis
        plt.clf()

    def __save_lr_plot(self, save_name):
        plt.plot(self.history.history["lr"])
        plt.title("LR during training")
        plt.ylabel("learning rate")
        plt.xlabel("epoch")
        plt.savefig(save_name)
        # Clear points/axis
        plt.clf()

    def __save_loss_plot(self, save_name):
        plt.plot(self.__smooth_curve(self.history.history["loss"]))
        plt.plot(self.__smooth_curve(self.history.history["val_loss"]))
        plt.title("Model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper left")
        # plt.show()
        plt.savefig(save_name)

    def __save_stats(self, save_name):
        metricl, maxl, minl, meanl, stdl = [], [], [], [], []
        for key, val in self.history.history.items():
            metricl.append(key)
            maxl.append(max(val))
            minl.append(min(val))
            meanl.append(np.mean(val))
            stdl.append(np.std(val))
        pd.DataFrame(
            {"metric": metricl, "max": maxl, "min": minl, "mean": meanl, "std": stdl}
        ).to_csv(save_name, index=False)

    # Suavizado Exponencial
    # Sacado de 'Deep Learning with Python' por François Chollet
    def __smooth_curve(self, points, factor=0.6):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--show-aug",
        dest="show_aug",
        action="store_true",
        help="Visualizar Data Augmentation (no entrenar)",
    )
    parser.add_argument(
        "--modelo-cpu",
        dest="cpu",
        action="store_true",
        help="Alternativamente elegir entrenar el modelo para CPU",
    )
    parser.add_argument(
        "--dense",
        dest="dense",
        action="store_true",
        help="Incluir Dense Layers para la parte final(head) de clasificacion",
    )
    parser.add_argument(
        "--annotations",
        dest="anotaciones_path",
        default="./train_val_set/train_anotaciones.txt",
        type=str,
        help="Path del .txt que contiene las anotaciones",
    )
    parser.add_argument(
        "--val-annotations",
        dest="val_anotaciones_path",
        type=str,
        help="Path del .txt que contiene las anotaciones",
    )
    parser.add_argument(
        "--height", dest="height", default=70, type=int, help="Alto de imagen a utilizar"
    )
    parser.add_argument(
        "--width", dest="width", default=140, type=int, help="Ancho de imagen a utilizar"
    )
    parser.add_argument(
        "--lr", dest="lr", default=1e-3, type=float, help="Valor del learning rate inicial"
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=64,
        type=int,
        help="Tamaño del batch, predeterminado 1",
    )

    parser.add_argument(
        "--output-dir",
        dest="output_path",
        default=None,
        type=str,
        help="Path para guarda el modelo",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        default=500,
        type=int,
        help="Cantidad de Epochs(cuantas veces se ve el dataset completo",
    )

    parser.add_argument(
        "--cut-out",
        dest="cut_out",
        action="store_true",
        help="Aplicar cut out a las imagenes, adicionalmente al Augmentation normal",
    )

    parser.add_argument(
        "--blur",
        dest="blur",
        action="store_true",
        help="Aplicar motion blur a las imagenes, adicionalmente al Augmentation normal",
    )
    parser.add_argument(
        "--graficos",
        dest="graphics",
        action="store_true",
        help="Guardar imagenes graficos de entrenamiento (loss, cat_acc, etc...)",
    )

    args = parser.parse_args()
    da = DataAugmentation(do_blur=args.blur, do_cut_out=args.cut_out)

    if args.show_aug:
        datagen, _ = da.data_aug()
        # Por defecto se aumenta las imagenes de benchmark/imgs
        x_imgs, _ = Preprocess("./benchmark/anotaciones.txt", args.height, args.width)()
        aug_generator = datagen.flow(x_imgs, batch_size=1, shuffle=True)

        fig, ax = plt.subplots(nrows=6, ncols=6)
        for row in ax:
            for col in row:
                img = aug_generator.next() * 255.0
                col.imshow(np.squeeze(img), cmap="gray")
        # show the figure
        plt.show()
    else:
        # Entrenar
        if args.cpu:
            modelo = modelo_1m_cpu(args.height, args.width, args.dense)
        else:
            modelo = modelo_2m(args.height, args.width, args.dense)
        modelo.compile(
            loss=cce,
            optimizer=tf.keras.optimizers.Adam(args.lr),
            metrics=[cat_acc, plate_acc, top_3_k],
        )

        datagen, datagen_validator = da.data_aug()
        # Pre-procesar .txt -> arrays de numpy listo para model.fit(...)
        x_imgs, y_imgs = Preprocess(args.anotaciones_path, args.height, args.width)()
        train_generator = datagen.flow(x_imgs, y_imgs, batch_size=args.batch_size, shuffle=True)
        train_steps = train_generator.n // train_generator.batch_size

        if args.val_anotaciones_path is not None:
            X_test, y_test = Preprocess(args.val_anotaciones_path, args.height, args.width)()
            validation_generator = datagen_validator.flow(
                X_test, y_test, batch_size=args.batch_size, shuffle=False
            )
            validation_steps = validation_generator.n // validation_generator.batch_size
        else:
            validation_generator = None
            validation_steps = None

        callbacks = [
            # Si en 35 epochs no mejora val_plate_acc reducir
            # lr por un factor de 0.5x
            ReduceLROnPlateau("val_plate_acc", verbose=1, patience=35, factor=0.5, min_lr=1e-5),
            # Parar de entrenar si val_plate_acc no aumenta en 50 epochs
            # Guardo la mejor version con restore_best_weights
            EarlyStopping(monitor="val_plate_acc", patience=50, restore_best_weights=True),
        ]

        history = modelo.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=args.epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
        )

        # modelo.save(f'model2m_trained_{args.epochs}.h5')
        best_vpa = max(history.history["val_plate_acc"])
        epochs = len(history.epoch)
        model_name = f"cnn-ocr_{best_vpa:.4}-vpa_epochs-{epochs}"
        # Make dir for trained model
        if args.output_path is None:
            model_folder = f"./trained/{model_name}"
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            output_path = model_folder
        else:
            output_path = args.output_path
        modelo.save(os.path.join(output_path, f"{model_name}.h5"))

        if args.graphics:
            # Save graphs to current dir.
            Graph(history, output_path)()
