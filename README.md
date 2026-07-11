## Resumen del proyecto

Este proyecto implementa una solución en PyTorch para el concurso **Histopathologic Cancer Detection** de Kaggle. El objetivo es clasificar imágenes histopatológicas según la presencia o ausencia de tejido tumoral, generando como salida una probabilidad para cada imagen del conjunto de test.

El trabajo partió del análisis del código base proporcionado como referencia. A partir de ahí, se adaptó el flujo de entrenamiento y predicción a los requisitos del concurso, especialmente en la generación del archivo de `submission`, donde Kaggle espera probabilidades y no clases binarias `0/1`.

Durante el desarrollo se trabajó en varias fases:

1. Comprensión y adaptación del código inicial.
2. Preparación de `Dataset` y `DataLoader` para entrenamiento, validación y test.
3. Uso de `train_labels.csv` y `sample_submission.csv` para construir los datasets, evitando depender del orden de los archivos en disco.
4. Ajuste de la salida del modelo para generar probabilidades válidas para Kaggle.
5. Implementación de la métrica objetivo del concurso: `ROC-AUC`.
6. Registro de métricas y experimentos con **Weights & Biases**.
7. Aplicación de `hyperparameter tuning` mediante **W&B Sweeps**.
8. Guardado del mejor modelo de cada run según `val_auc_roc`.
9. Aplicación de normalización y `data augmentation` en entrenamiento.
10. Prueba de una arquitectura CNN alternativa con menor reducción espacial.
11. Generación y envío de submissions a Kaggle.

Aunque el resultado obtenido no alcanza las mejores puntuaciones históricas del leaderboard, el proyecto ha servido para construir un pipeline completo de entrenamiento, validación, optimización de hiperparámetros, guardado de modelos y generación de submissions.

Además, durante la revisión se corrigieron varias partes importantes del flujo original: construcción de datasets basada en CSV, separación de transformaciones, normalización, control de semillas aleatorias, logging más limpio de métricas en W&B y sustitución del patrón `LogSoftmax + NLLLoss` por una salida basada en logits junto con `CrossEntropyLoss`.

## Estructura general

El proyecto está dividido en módulos para separar responsabilidades:

* `src/train.py`: lógica principal de entrenamiento.
* `src/sweep.py`: configuración y ejecución de W&B Sweeps.
* `src/predict.py`: carga del modelo entrenado y generación de predicciones sobre test.
* `src/dataset.py`: definición de los datasets personalizados.
* `src/model.py`: arquitectura CNN inicial.
* `src/model_more_spatial.py`: arquitectura CNN alternativa con menor reducción espacial.
* `src/transforms.py`: transformaciones aplicadas a las imágenes.
* `src/config.py`: rutas y configuración general del proyecto.
* `src/helper_functions.py`: funciones auxiliares reutilizables.
* `scripts/`: scripts auxiliares para descarga de datos, análisis de modelos y otras tareas.

## Archivos ignorados

Por tamaño y reproducibilidad, algunos archivos no se incluyen en el repositorio:

* Datos originales del concurso.
* Modelos entrenados (`.pt`).
* Checkpoints.
* Métricas generadas localmente.
* Archivos temporales o resultados pesados.
* Submissions generadas durante la experimentación.

Estos archivos deben gestionarse localmente y están excluidos mediante `.gitignore`.

## Uso básico

Antes de ejecutar el proyecto, es necesario adaptar las rutas en `src/config.py` según la ubicación local del dataset y de los directorios de salida.

Flujo general:

```bash
# Entrenamiento normal
python src/train.py

# Lanzar sweep de W&B
python src/sweep.py

# Generar submission
python src/predict.py
```

Para revisar los mejores modelos guardados localmente:

```bash
bash scripts/top_10_models.sh
```

## Resultados

Se generaron varias submissions en Kaggle a partir de diferentes versiones del modelo:

| Submission | Modelo | Public Score | Private Score |
|---|---|---:|---:|
| `submission1.csv` | CNN base | 0.8898 | 0.8704 |
| `submission2.csv` | CNN base con sweep | 0.8886 | 0.8031 |
| `submission3.csv` | CNN MoreSpatial + normalización + data augmentation | 0.9091 | 0.8412 |

El modelo `CustomCNN_MoreSpatial` obtuvo el mejor resultado en `public score`, pero no mejoró el `private score` respecto a la primera submission.

Esto indica que las mejoras aplicadas al modelo y al pipeline sí mejoraron el rendimiento sobre la parte pública del test, pero no necesariamente la generalización sobre el conjunto privado de Kaggle.

El mejor checkpoint local del modelo `CustomCNN_MoreSpatial` se conserva en:

```text
outputs/models/custom-cnn-morespatial/best_model.pt
```

## Limitaciones

### Riesgo de data leakage

El dataset no proporciona identificadores de paciente, biopsia o slide original. Por este motivo, no es posible realizar una separación agrupada por paciente entre entrenamiento y validación.

La validación se realiza mediante un split aleatorio sobre las imágenes disponibles, lo que puede introducir cierto riesgo de `data leakage` si varios parches proceden de una misma muestra médica original y acaban repartidos entre train y validation.

En un entorno clínico real, lo adecuado sería validar agrupando por paciente, biopsia o slide, garantizando que todas las imágenes asociadas a una misma fuente permanezcan únicamente en uno de los subconjuntos.

### Diferencia entre validación local y Kaggle

El mejor modelo local fue seleccionado según `val_auc_roc`, pero esta métrica no siempre se correspondió directamente con el `private score` de Kaggle.

Esto puede deberse a varios factores:

* El split local puede no representar bien la distribución del test privado.
* Puede existir variabilidad entre la parte pública y privada del leaderboard.
* Las mejoras de arquitectura o `data augmentation` pueden aumentar la robustez en algunos subconjuntos, pero no necesariamente en todos.
* El dataset no permite controlar agrupaciones por paciente o biopsia.

## Próximas mejoras

Como trabajo futuro, la mejora principal sería reforzar la generalización del modelo. Algunas líneas razonables serían:

* Probar arquitecturas preentrenadas como `ResNet`, `EfficientNet` o `DenseNet`.
* Usar validación cruzada o varios splits estratificados.
* Comparar ensembles de varios modelos.
* Revisar estrategias de `data augmentation` específicas para imágenes histopatológicas.
* Ajustar el tamaño del `crop` y estudiar si se pierde información relevante.
* Mejorar el control experimental comparando los modelos con el mismo split y la misma configuración base.



# Histopathologic Image Classification Using Convolutiona Neural Network
---

# Descripción general
El cuaderno documenta un problema de clasificación binaria de computer vision. Específicamente, el conjunto de datos se compone de microscopías de 220.025x de 96x96 píxeles con las etiquetas correspondientes que indican si la región central de 32x32 contiene células cancerosas metastásicas. Se construyó una red neuronal convolucional de 4 capas convolucionales utilizando PyTorch y el entrenamiento se realiza en épocas de 10 veces con varios hiperparámetros.

Se utilizó ADAM (Estimación de momento adaptativo) para optimizar el entrenamiento junto con la probabilidad logarítmica negativa (NLL) como métrica de pérdida. La razón para utilizar ADAM y NLL es su versatilidad y capacidad para adaptarse a problemas de clasificación más generales más allá de un problema de clasificación binaria para futuras investigaciones.


# Acerca del conjunto de datos - PatchCamelyon (PCam)
El conjunto de datos se puede encontrar [aquí](https://www.kaggle.com/competitions/histopathologic-cancer-detection) proporcionado como parte de una competicion de Kaggle. Como se menciona en la descripción de este conjunto de datos, se adaptó del conjunto de datos PatchCamelyon (PCam) y se eliminaron los duplicados del conjunto de datos original.

Dado que el conjunto de datos se limpia (elimina duplicados), no es necesario realizar ninguna limpieza. Todas las etiquetas se organizaron limpiamente en un CSV con el nombre de archivo de imagen correspondiente.

# Análisis de datos exploratorios (EDA)
Debido a que el conjunto de datos son datos de imágenes en lugar de datos tabulares con dimensiones, no hay mucho que podamos hacer en términos de análisis de datos exploratorios sin tener que procesar la imagen utilizando algún tipo de técnica de visión por computadora.

En cambio, lo único que podemos hacer es analizar el equilibrio de las etiquetas (no cáncer vs cáncer). Descubrimos que hay aproximadamente un 46,9 % más de imágenes no cancerosas que cancerosas (consulte el gráfico de barras a continuación).

# Procesamiento de imágenes (transformación)
El conjunto de datos contiene imágenes de 96x96 píxeles; sin embargo, las etiquetas solo indican si la región central de 32x32 contiene al menos una célula cancerosa metastásica. Los píxeles adicionales ignorados en el etiquetado tienen el mismo efecto que agregar relleno alrededor de las imágenes de entrada para cada capa de convolución de CNN.

Debido a que el tamaño de este conjunto de datos modificado es relativamente grande (~7 GB) para la cantidad limitada de recursos de entrenamiento que tenemos a mano, durante la etapa de procesamiento (transformación) de la imagen, la imagen se recorta en el centro a un cuadrado de 46x46, lo que aún proporciona una 14 píxeles alrededor de la región de interés (32x32), pero elimina los píxeles adicionales que solo habrían aumentado el tiempo de lectura de E/S de nuestra canalización.

El procesamiento de imágenes se realiza mediante una tubería de transformadores. La canalización convierte la imagen en un tensor de PyTorch y normaliza los valores de píxeles de cada una de las tres capas RGB desde el rango 0-255 a 0-1 para obtener una superficie de error más suave que facilita una mejor convergencia del modelo.

En futuras mejoras del modelo, se pueden generar imágenes histológicas adicionales a partir de este conjunto de datos volteando/rotando las imágenes existentes. Dado que las imágenes histológicas no se limitan a una orientación específica y se pueden cortar de un órgano/organismo en cualquier orientación, nos gustaría asegurarnos de que el modelo esté entrenado en todas las permutaciones posibles.

También sería interesante y beneficioso si pudiéramos proporcionar contexto adicional al modelo, como la edad del sujeto, el órgano de origen.


# Arquitectura modelo
El modelo está inspirado en el modelo LeNet donde hay múltiplos de (capa convolucional + agrupación) como sección de extracción de características, seguidas de dos capas lineales con log-softmax como salida final utilizada como clasificador de perceptrón multicapa. log-softmax proporciona una fácil adaptación cuando queremos ajustar y entrenar la red para más de dos etiquetas de clasificación.

## Sección de extracción de funciones
Hay capas 4x (convolución + activación ReLU + maxpooling), cada una sin relleno y con núcleos de filtro 3x3. Debido a lo pequeña que es la región de interés (32x32), elegí un núcleo de filtro más pequeño y profundicé la profundidad del modelo, ya que con cada mayor profundidad, el campo de percepción aumenta, lo que permite que el modelo capture indirectamente una característica invariante más grande.

## Sección de clasificación
La sección de clasificación se compone de dos capas lineales con un log-softmax como salida final. El beneficio de log-softmax es su capacidad para adaptarse a cualquier número de clases, por lo que el modelo se puede adaptar fácilmente para clasificar más de 2 categorías.

# Training and Validation
La siguiente función es el bucle principal de entrenamiento/validación.
Se utiliza un diccionario de configuración para especificar los hiperparámetros del modelo junto con el número de épocas como argumentos de la función. Después de cada época, se guarda un punto de control localmente para recuperarlo más tarde para continuar con el entrenamiento o a prueba de fallas si ocurriera una interrupción inesperada.


# Ajuste de hiperparámetros
Las siguientes llamadas al bucle principal de entrenamiento/validación pasan por diferentes diccionarios de configuración para probar el rendimiento de cada combinación de hiperparámetros. Específicamente, probó diferentes tasas de aprendizaje y tasas de disminución de peso.

## Pensamientos sobre cada hiperparámetro (ADAM)

### Tasa de aprendizaje:
- La tasa de aprendizaje se compensaría con el tiempo de capacitación y el trabajo en conjunto con el impulso. Una tasa de aprendizaje muy pequeña necesitaría un impulso mayor en comparación con una tasa de aprendizaje mayor.

### Weight decay:
- La caída de peso puede interpretarse como una regularización L2, ya que penaliza los coeficientes grandes del modelo.
- Según la interpretación, prefiero que estos valores estén en el extremo más pequeño para que cada filtro sea más exclusivo entre sí en lugar de tener parámetros muy similares debido a la penalización.

### Beta 1 y Beta 2:
- Estos parámetros se pueden interpretar como el promedio móvil y el cuadrado del gradiente.
- Los valores predeterminados son 0,9 y 0,9, lo que proporciona mucho impulso al optimizador, por lo que es mejor combinarlo con una pequeña tasa de aprendizaje para evitar sobrepasar el parámetro óptimo.