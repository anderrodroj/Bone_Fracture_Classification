import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import KFold
from datetime import datetime
from PIL import ImageFile
import json

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define image size and batch size
image_size = (150, 150)
batch_size = 32

# Define paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Hyperparameter tuning using Keras Tuner
class MyHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential([
            tf.keras.Input(shape=(150, 150, 3)),
            Conv2D(hp.Int('conv_1_units', min_value=32, max_value=128, step=32), (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(hp.Int('conv_2_units', min_value=32, max_value=128, step=32), (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'),
            Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

hypermodel = MyHyperModel()

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='bone_fracture_classification'
)

# Run the hyperparameter search
tuner.search(train_generator, epochs=10, validation_data=val_generator)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Set up cross-validation
kf = KFold(n_splits=5)
all_train_indices = np.arange(train_generator.samples)
fold_accuracies = []

for train_index, val_index in kf.split(all_train_indices):
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    # Callbacks for model checkpointing and TensorBoard
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

    # Train the model
    history = best_model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[tensorboard_callback, checkpoint_callback]
    )

    # Save fold accuracy
    fold_accuracies.append(history.history['val_accuracy'][-1])

# Calculate mean accuracy from cross-validation
mean_accuracy = np.mean(fold_accuracies)
print(f'Mean cross-validation accuracy: {mean_accuracy}')

# Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_acc}')

# Generate predictions on the test data
predictions = best_model.predict(test_generator, steps=test_generator.samples // batch_size)
predicted_classes = np.round(predictions).astype(int).reshape(-1)
true_classes = test_generator.classes[:len(predicted_classes)]

# Calculate the number of correct and incorrect predictions
correct_predictions = np.sum(predicted_classes == true_classes)
incorrect_predictions = np.sum(predicted_classes != true_classes)

print(f"Number of correct predictions: {correct_predictions}")
print(f"Number of incorrect predictions: {incorrect_predictions}")

# Save training parameters and results
training_results = {
    'mean_cross_validation_accuracy': mean_accuracy,
    'test_accuracy': test_acc,
    'correct_predictions': correct_predictions,
    'incorrect_predictions': incorrect_predictions
}

with open('training_results.json', 'w') as f:
    json.dump(training_results, f)
