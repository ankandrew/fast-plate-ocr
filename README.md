# Reconocedor de Patentes OCR de Patentes (Arg)

![Demo](extra/local_recog_demo.png)

**OCR** implementado con solo Redes Convolucionales (**CNN**) de Patentes Argentinas. Los modelos son entrenados con patentes de 6 digitos (viejas) y patentes del Mercosur de 7 digitos (las nuevas). Este repo esta dedicado solamente al modulo que se encarga de *f(imagen patente recortada) = texto de patente*

Es común que se aplique una **ConvNet(CNN)** y una **Recurrent Neural Net. (LSTM/GRU)** para modelar este tipo de problema de secuencia de caracteres a partir de una imagen. En este caso se implementan solo ConvNets debido a:
* Se busca deployear en **sistemas embebidos** como RaspBerry Pi + Accelerator, por ende tiene que ser ligero.
* No tenemos el problema de una **secuencia variable de longitud**. El máximo de caracteres posibles es 7 (para Argentina) por ende las patentes de 6 digitos se le asigna una caracter extra para indicar el faltante.

## Uso

### Instalar dependencias

Contar con **python 3.x**, instalar los requerimientos:

```pip install requirements.txt```

### Visualizar predicciones

```python demo_recog.py -m models/m2_85_vpc_3.9M.h5```

*Se visualizaran las predicciones hechas a patentes que se encuentren en la carpeta val_set/imgs/*

### Calcular precisión

```python valid.py -m models/m2_85_vpc_3.9M.h5```

Ejemplo de salida:

```
loss: 1.4920 - cat_acc: 0.9592 - plate_acc: 0.8503 - top_3_k: 0.9796
```

## Caracteristicas

Los modelos son una tipica ConvNet. Las capas/layers estan formadas por `Convolution -> BatchNorm -> Activation -> MaxPooling` ... hasta formar un volumen de AxHx1024 *(altura x ancho x canales)* ... se le aplica GlobalAvgPooling para formar un volumen de 1x1x1024 que se conecta (mediante una Fully Conected Layer) con 37 x 7 unidades con activacion `softmax`. El numero 37 viene de 26 (vocabulario) + 10 digitos + simbolo de faltante `'_'`, por 7 porque por cada posición tiene una probabilidad de 37 caracteres.

![model head](extra/FCN.png)

* **Regularización**: Se probo DropBlock, DropOut y l2 reg. a los filtros. Este ultimo dio los mejores resultados
   * **Label Smoothing**: le da un 10% notorio de aumento de `plate_acc`. Se suavizan los one-hot encoding y pasan de ser (por ejemplo) ```[0, 0, 0, 1]``` a ```[0.01, 0.01, 0.01, 0.90]```
* **Data Augmentation**: Se usa la augmentacion estandard de Keras y se aplica:
    * Cambios de brillo
    * Leve rotaciones
    * Shearing (tambien leve)
    * Zoom
    * Desplazamiento Vertical/Horizontal
* **Input**
   * Imagen **blanco & negro**
       * 70x140 *(altura x ancho)*
       * Interpolacion **bilineal**

## Validación

Para validar la calidad de los modelos se utilizara *principalmente* una metrica personalizada `plate_acc`. Esta simplemente calcula el porcentaje de patentes bien categorizadas **en su totalidad**.

Ejemplo si se tiene 2 patentes: { `AB 123 DC`, `GKO 697` } y se predice { `AB 123 CC`, `GKO 697` } la precisión es de 50%, una patente correctamente reconocida y la otra no.
Métrica definida en Keras:
```python
from tensorflow.keras import backend as K

def plate_acc(y_true, y_pred):
    et = K.equal(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(
        K.cast(
          K.all(et, axis=-1, keepdims=False),
          dtype='float32'
        )
    )
```

Ninguna imagen (como corresponde) del val_set fue usada para entrenar el modelo. Para evaluar mejor la precisión se necesita un validation-set publico con mas imagenes

## Benchmarks

| modelo  | cat_acc | plate_acc | top_3_k |
| -------  | ----------- | ------ | ------ |
| m2_85_vpc_3.9M |   **0.9592**    | **0.8503** | **0.9796** |
| m1_78_vpc_3.5M |   0.9446    | 0.7823 | 0.9757 |

* **top_3_k** calcula que tan seguido el caracter verdadero se encuentra en las 3 predicciones con mayor probabilidades
* **cat_acc** es simplemente la [CategoricalAccuracy](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy) para problemas de multi-class labels. **Ejemplo** si el label correcto es `ABC123` y se predice `ABC133` no va a dar una precisión de 0% como plate_acc *(no clasificada correctamente en su totalidad)*, sino de 83.3% (5/6)

*Estas metricas estan ubicadas en el archivo custom.py*

## Set - Validación

La fuente principal del set de validación proviene de estos de [video night drive](https://www.youtube.com/watch?v=75X9vSFCh14) y [video morning drive](https://www.youtube.com/watch?v=-TPJot7-HTs). Créditos a [J Utah](https://www.youtube.com/channel/UCBcVQr-07MH-p9e2kRTdB3A).

Formato de *val_set/anotaciones.txt* (separado por tab):
```
imgs/nombre_imagen.png  ABC 123 DE
```

*Si desean colaborar para expandir el set de validación, mejor aún.*

## Active Learning

Hacer las anotaciones de miles de patentes resulta un trabajo **largo** e **impractico**. Por eso se propone, **"Active Learning"** que simplemente con el modelo base entrenado (con 1800~ imagenes) se predicen patentes no vistas. Luego anotan solo las patentes de **baja confianza**. Este es un proceso **iterativo** y se repite hasta llegar a la precisión deseada (en el test de validación)

![Active Learning](extra/Active_Learning.jpg)

Para decidir si la prediccion tiene poca confianza, se utiliza:

* Si por lo menos un caracter esta poco seguro
```python
def on_any(probs, thresh=.3):
  return any([True if prob < thresh else False for prob in probs])
```

* Si el promedio de todos los caracteres esta por debajo de un valor
```python
def on_avg(probs, avg_thresh=.2):
  return statistics.mean(probs) < avg_thresh 
```

*Métodos no optimizados, solo para ilustración*

En la siguiente tabla se va a mostrar los modelos (misma arquitectura) pero con más imagenes de entrenamiento, basado en el criterio anterior. Para eliminar la varianza en los resultados, y ver el impacto de agregar mas imagenes al dataset de entrenamiento: la arquitectura, método de optimizacion, Data Augmentation ... no cambia en absoluto.

#### Modelo 1 (3.5 M params)

| iteracion  | Set-Entrenamiento | cat_acc | plate_acc | top_3_k |
| -------  | ---------- | ----------- | ------ | ------ |
| 1 |   0.9446   | 1800 | 0.7823 | 0.9757 |
| 2 |   -   | - | - | - |

#### Modelo 2 (3.9 M params)

| iteracion  | Set-Entrenamiento | cat_acc | plate_acc | top_3_k |
| -------  | ---------- | ----------- | ------ | ------ |
| 1 |   0.9592   | 1800 | 0.8503 | 0.9796 |
| 2 |   -   | - | - | - |


## Data Augmentation

![Data Aug](extra/data_aug_ejemplo.png)

Configuracion de Data Aug en Keras:

```python
datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.10,
    brightness_range=(0.5, 1.5),
    shear_range=8,
    zoom_range=0.12
)
```

*Aclaracion: A proposito se busco, manualmente, que de vez en cuando los caracteres salgan **un poco** del frame. Esto ayuda a que generalice mejor y que no se espere una patente recortada perfectamente.

## TODO

- [x] Publicar modelo experimental
- [x] Label Smoothing
- [x] <del> Implementar SAM (Spatial Attention Module) </del>
- [x] Active Learning
- [ ] Ampliar val-set
- [ ] Aplicar blur a las imagenes(Data Augmentation)
- [ ] Quantizar el modelo a INT8
- [ ] Compilarlo para Edge TPU
- [ ] Hacer version universal (Patentes de EU, BR, ...)
- [ ] Aumentar el training-set de patentes de motos

## Notas

* Este modelo deberia tener muy mala precisión en patentes **no** *Argentinas*
* Para obtener la mejor precisión es recomendable utilizar obtener las patentes recortadas con [YOLO v4/v4 tiny](https://github.com/ankandrew/LocalizadorPatentes)
* Los modelos fueron entrenados inicialmente con 1800 fotos solamente y validado en 596 imagenes (Se aumenta iterativamente con Active Learning)
* La proporcion de vehiculos y motos esta en desproporcion, las fotos de motos representan menos del 10% del training-set *(Por ahora)*
* DropBlock no dio buenos resultados
