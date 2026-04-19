# ============================================================
# SPRINT 2: MODELADO, ENTRENAMIENTO Y PRIMER PROTOTIPO
# PROYECTO: DIAGNÓSTICO DE ENFERMEDADES EN HOJAS DE TOMATE
# ============================================================

# ============================================================
# 1. ENTRENAMIENTO CON ACELERACIÓN POR HARDWARE (GPU)
# ============================================================

import tensorflow as tf
import numpy as np
import os

# Verificar GPU disponible
print("Dispositivos disponibles:", tf.config.list_physical_devices())
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

# ============================================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ============================================================

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import os
 
#tolerancia a imágenes truncadas
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = r"C:\Users\USER\Documents\Universidad de Cundinamarca\Semestre 11\RadicacionPGC\archive\train"
VAL_DIR = r"C:\Users\USER\Documents\Universidad de Cundinamarca\Semestre 11\RadicacionPGC\archive\val"

# ============================================================
# LIMPIEZA AUTOMÁTICA DEL DATASET
# ============================================================

from PIL import Image

def limpiar_dataset(dataset_dir):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verifica integridad
            except Exception:
                print("Imagen corrupta eliminada:", file_path)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print("No se pudo eliminar:", file_path, "Error:", e)

# Ejecutar limpieza en train y val
limpiar_dataset(TRAIN_DIR)
limpiar_dataset(VAL_DIR)


# ============================================================
# 2. PREPROCESAMIENTO Y AUGMENTACIÓN DE DATOS
# ============================================================

# Generador de entrenamiento con augmentación
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3]
)

# Generador de validación sin augmentación
val_datagen = ImageDataGenerator(rescale=1./255)

# Carga de datos desde carpetas
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ============================================================
# 3. BALANCEO DE CLASES
# ============================================================

labels = train_generator.classes

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))
print("Pesos de clase:", class_weights)

# ============================================================
# 4. MODELO BASE: MobileNetV2
# ============================================================

# Cargar modelo preentrenado sin la capa final
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Congelar capas base para no modificar pesos originales
for layer in base_model.layers:
    layer.trainable = False

# Añadir capas personalizadas para clasificación
x = base_model.output
x = GlobalAveragePooling2D()(x)   # Reduce dimensiones
x = Dense(128, activation='relu')(x)  # Capa densa
x = Dropout(0.5)(x)  # Evita overfitting
output = Dense(train_generator.num_classes, activation='softmax')(x)

# Crear modelo final
model = Model(inputs=base_model.input, outputs=output)

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 5. ENTRENAMIENTO DEL MODELO
# ============================================================

# Callbacks para mejorar entrenamiento
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.3, patience=3)
]

# Entrenamiento
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ============================================================
# 6. EVALUACIÓN DEL MODELO
# ============================================================

# Predicciones
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Etiquetas reales
y_true = val_generator.classes

# Reporte de métricas
print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred_classes))

# ============================================================
# 7. GUARDADO DEL MODELO (.h5)
# ============================================================

model.save("modelo_tomate.h5")
print("Modelo guardado en formato .h5")

# ============================================================
# 8. CONVERSIÓN A TENSORFLOW LITE
# ============================================================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimización
tflite_model = converter.convert()

with open("modelo_tomate.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo convertido a TensorFlow Lite")

# ============================================================
# 9. CONVERSIÓN A TENSORFLOW.JS
# ============================================================

# Instalar tensorflowjs si es necesario
try:
    import tensorflowjs
except:
    os.system("pip install tensorflowjs")

# Convertir modelo a formato web
os.system("tensorflowjs_converter --input_format=keras modelo_tomate.h5 tfjs_model/")

print("Modelo convertido a TensorFlow.js")

# ============================================================
# 10. GENERACIÓN DEL PROTOTIPO FRONTEND (HTML + JS)
# ============================================================

html_code = """
<!DOCTYPE html>
<html>
<head>
    <title>Diagnóstico de Enfermedades en Tomate</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
</head>
<body>

<h2>Captura de Imagen</h2>

<video id="video" width="224" height="224" autoplay></video><br>
<button onclick="capturar()">Capturar</button>
<button onclick="predecir()">Predecir</button><br>

<canvas id="canvas" width="224" height="224"></canvas>

<script>

let model;

// Cargar modelo
async function cargarModelo() {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log("Modelo cargado");
}
cargarModelo();

// Activar cámara
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    document.getElementById("video").srcObject = stream;
});

// Capturar imagen
function capturar() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 224, 224);
}

// Predecir
function predecir() {
    const canvas = document.getElementById("canvas");

    let tensor = tf.browser.fromPixels(canvas)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims();

    let pred = model.predict(tensor);
    pred.print();
}

</script>

</body>
</html>
"""

# Guardar archivo HTML
with open("prototipo.html", "w", encoding="utf-8") as f:
    f.write(html_code)

print("Prototipo web generado: prototipo.html")

# ============================================================
# FIN DEL SCRIPT
# ============================================================