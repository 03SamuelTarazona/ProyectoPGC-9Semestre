# PROYECTO: DETECCIÓN DE ENFERMEDADES EN HOJAS DE TOMATE
# SPRINT 1: PREPARACIÓN DE DATOS + MODELO BASE (MobileNetV2)

# ========================
# 1. LIBRERÍAS
# ========================
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile 
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
#import tensorflowjs as tfjs

# ========================
# 2. CONFIGURACIÓN
# ========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = r"C:\Users\USER\Documents\Universidad de Cundinamarca\Semestre 11\RadicacionPGC\archive\train"
VAL_DIR = r"C:\Users\USER\Documents\Universidad de Cundinamarca\Semestre 11\RadicacionPGC\archive\valid"

# ========================
# 3. DEPURACIÓN DEL DATASET
# ========================

# Permite cargar imágenes truncadas (evita crash)
ImageFile.LOAD_TRUNCATED_IMAGES = True

def clean_dataset(folder):
    print(f"Limpiando dataset en: {folder}")

    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)

            try:
                # Abrir imagen correctamente
                with Image.open(path) as img:
                    img = img.convert("RGB")  # fuerza formato válido

                # Leer con OpenCV
                img_cv = cv2.imread(path)

                if img_cv is None:
                    raise Exception("Imagen inválida")

                # Redimensionar
                img_cv = cv2.resize(img_cv, (224, 224))

                # Suavizado
                img_cv = cv2.GaussianBlur(img_cv, (5,5), 0)

                # Sobrescribir imagen limpia
                cv2.imwrite(path, img_cv)

            except Exception as e:
                print(f"Error con imagen: {path}")
                print(f"Motivo: {e}")

                try:
                    os.remove(path)
                    print("Imagen eliminada correctamente\n")
                except PermissionError:
                    print("⚠️ No se pudo eliminar (archivo en uso)\n")

# ========================
# 4. PREPROCESAMIENTO Y AUGMENTACIÓN
# ========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

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

# ========================
# 5. BALANCEO DE CLASES
# ========================

labels = train_generator.classes

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))
print("Pesos de clase:", class_weights)

# ========================
# 6. MODELO BASE: MobileNetV2
# ========================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Congelar capas base
for layer in base_model.layers:
    layer.trainable = False

# Capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compilar modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========================
# 7. ENTRENAMIENTO
# ========================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights
)

# ========================
# 8. GUARDADO DEL MODELO (.h5)
# ========================

model.save("modelo_tomate.h5")
print("Modelo guardado en formato .h5")

# ========================
# 9. CONVERSIÓN A TENSORFLOW LITE
# ========================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("modelo_tomate.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo convertido a TensorFlow Lite")

# ========================
# 10. CONVERSIÓN A TENSORFLOW.JS
# ========================
""""
# Exportar el modelo a formato TensorFlow.js
tfjs.converters.save_keras_model(model, "./modelo_js")

print("Modelo convertido a TensorFlow.js (carpeta 'modelo_js')")
"""