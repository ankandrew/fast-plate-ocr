# cnn-ocr-lp

**OCR** implementado con solo Redes Convolucionales (**CNN**) de Patentes Argentinas. Los modelos son entrenados con patentes de 6 digitos (viejas) y patentes del Mercosur de 7 digitos (las nuevas). Tambien se incluyeron fotos de motos, el formato de estas es diferente al de los vehiculos.

Es común que se aplique una **ConvNet(CNN)** y una **Recurrent Neural Net. (LSTM/GRU)** para modelar este tipo de problema de secuencia de caracteres a partir de una imagen. En este caso se implementan solo ConvNets debido a:
* Se buscar deployear en **sistemas embebidos** como RaspBerry Pi + Accelerator, por ende tiene que ser ligero.
* No tenemos el problema de una **secuencia variable de longitud**. El máximo de caracteres posibles es 7 (para Argentina) por ende las patentes de 6 digitos se le asigna una caracter extra para indicar el faltante.

Todas las imagenes procesadas son:
1. Convertidas a **blanco y negro**
1. Ajustadas a un tamaño de **70x140**
1. Píxeles **normalizados** entre valores **[0, 1]**

### Caracteristicas

* **Label Smoothing**: le da un 10% notorio de aumento de `plate_acc`. Se suavizan los one-hot encoding y pasan de ser (por ejemplo) ```[0, 0, 0, 1]``` a [0.01, 0.01, 0.01, 0.90]
* **Regularización**: Se probo DropBlock, DropOut y l2 reg. a los filtros. Este ultimo dio los mejores resultas
* Data Augmentation: Se usa la augmentacion estandard de Keras y se aplica:
    * Cambios de brillo
    * Leve rotaciones
    * Shearing (tambien leve)
    * Zoom
    * Desplazamiento Vertical/Horizontal
* Imagen blanco & negro de **70x140** *(altura x ancho)*
    * Interpolacion **bilineal**

### Validación

Para validar la calidad de los modelos se utilizara una metrica personalizada `plate_acc`. Esta simplemente calcula el porcentaje de patentes bien categorizadas **en su totalidad**.

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

La fuente principal del set de validación proviene de estos de [video night drive](https://www.youtube.com/watch?v=75X9vSFCh14) y [video morning drive](https://www.youtube.com/watch?v=-TPJot7-HTs). Créditos a [J Utah](https://www.youtube.com/channel/UCBcVQr-07MH-p9e2kRTdB3A).

*Si desean colaborar para expandir el set de validación, es de ayuda.


### TODO

- [ ] Publicar modelo
- [ ] Ampliar val-set
- [ ] Implementar SAM (Spatial Attention Module)
- [x] Label Smoothing
