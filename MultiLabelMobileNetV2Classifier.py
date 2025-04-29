import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import tensorflow as tf

# constants
IMAGE_SIZE = (224, 224)
IMG_SHAPE = IMAGE_SIZE + (3,) #full shape
BATCH_SIZE = 32
EPOCHS = 10

BASE_DIR = '/users/sarvin/Documents/python-vope/Experimental_Data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train_filtered')
VAL_DIR = os.path.join(BASE_DIR, 'validation_filtered')
TEST_DIR = os.path.join(BASE_DIR, 'test_filtered')



def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')# multi-label

    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

    return train_generator, val_generator, test_generator


train_gen, val_gen, test_gen = create_data_generators()


# model building
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax')  # Change output classes as needed
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# train model
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)


# model evaluation
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


# prediction
y_pred_prob = model.predict(test_gen)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_gen.classes

#confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_gen.class_indices.keys(),
            yticklabels=test_gen.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# plotting
history_dict = history.history
epochs = range(1, len(history_dict['accuracy']) + 1)

plt.figure(figsize=(12, 6))
plt.plot(epochs, history_dict['accuracy'], label='Training Accuracy')
plt.plot(epochs, history_dict['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
