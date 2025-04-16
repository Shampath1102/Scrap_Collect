import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

# Dataset path
dataset_path = r"C:\Users\rupa1\OneDrive\Desktop\New LSM App\model_python\condition_dataset"

# Remove corrupted or unreadable images
for class_name in ['Good', 'Poor']:
    class_path = os.path.join(dataset_path, class_name)
    
    if not os.path.exists(class_path):
        print(f"[!] Folder not found: {class_path}")
        continue

    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        try:
            img = tf.keras.utils.load_img(file_path)
        except Exception as e:
            print(f"[!] Skipping invalid/missing file: {file_path}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"[✓] Removed corrupted file: {file_path}")
                except Exception as del_err:
                    print(f"[x] Failed to delete {file_path}: {del_err}")

# Image size and batch setup
img_size = (224, 224)
batch_size = 32

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

# Data generators
train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

print("Total training images:", train_gen.samples)
print("Class indices:", train_gen.class_indices)

# Safety wrapper to skip invalid batches
class SafeImageGenerator(Sequence):
    def __init__(self, generator):
        self.generator = generator

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, idx):
        while True:
            try:
                return self.generator[idx]
            except Exception as e:
                print(f"[!] Skipping batch {idx} due to error: {e}")
                idx = (idx + 1) % len(self.generator)

safe_train_gen = SafeImageGenerator(train_gen)
safe_val_gen = SafeImageGenerator(val_gen)

# Build the model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(safe_train_gen, epochs=10, validation_data=safe_val_gen)

# Save the model and class mapping
model.save("condition_model_binary.h5")
with open("condition_class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)
