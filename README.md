# cnn-ocr-lp
OCR implementado con solo Redes Convolucionales (CNN) de Patentes Argentinas. Los modelos son entrenados con patentes de 6 digitos (viejas) y patentes del Mercosur de 7 digitos (las nuevas). Tambien se incluyeron fotos de motos, el formato de estas es diferente al de los vehiculos.

## Validación

Para validar la calidad de los modelos se utilizara una metrica personalizada `plate_acc`. Esta simplemente calcula el porcentaje de patentes bien categorizadas **en su totalidad**.

Ejemplo si se tiene 2 patentes: { `AB 123 DC`, `GKO 697` } y se predice { `AB 123 CC` y `GKO 697` } la precisión es de 50%, una patente correctamente reconocida y la otra no.
Métrica definida en Keras:
```python
from tensorflow.keras import backend as K

et = K.equal(K.argmax(y_true), K.argmax(y_pred))

return K.mean(
    K.cast(
      K.all(et, axis=-1, keepdims=False),
      dtype='float32'
    )
)
```
*Si desean colaborar para expandir el set de validación, es de ayuda.

## TODO

- [ ] Publicar modelo
- [ ] Ampliar val-set
- [ ] Implementar SAM (Spatial Attention Module)
- [x] Label Smoothing
