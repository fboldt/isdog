from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"), 
    layers.RandomRotation(0.1), 
    layers.RandomZoom(0.2)])

database_dir = "scraper_database"

def make_subset(subset="training", directory=database_dir):
    return image_dataset_from_directory(directory,
                                        image_size=(180, 180),
                                        validation_split=0.25,
                                        subset=subset,
                                        seed=42,
                                        batch_size=32)
    
train_dataset = make_subset("training")
validation_dataset = make_subset("validation")

def fineTuningLayers(x):
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dropout(0.5)(x)
    return x

def assessArchitecture(architecture, preprocess_input, checkpoint="architecture.dbg", debug=False):
    num_labels = len(train_dataset.class_names)
    
    conv_base = architecture(weights="imagenet", include_top=False)
    conv_base.trainable = False

    inputs = keras.Input(shape=(train_dataset.element_spec[0].shape[1], 
                                train_dataset.element_spec[0].shape[2], 
                                train_dataset.element_spec[0].shape[3]))
    x = data_augmentation(inputs)
    if preprocess_input:
        x = preprocess_input(x)
    x = conv_base(x)

    x = fineTuningLayers(x)
    outputs = layers.Dense(num_labels, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=checkpoint, 
                                                 save_best_only=True, 
                                                 monitor="val_loss"), 
                 keras.callbacks.EarlyStopping(monitor="val_loss", 
                                               patience=5) ]
    epochs = 1 if debug else 50
    history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, callbacks=callbacks)
    model = keras.models.load_model(checkpoint)
    return model.evaluate(validation_dataset), history
