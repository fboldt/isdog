from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"), 
    layers.RandomRotation(0.1), 
    layers.RandomZoom(0.2)])

database_dir = "Kaggle+API/teste" # KaggleScraper/Kaggle+Scraper Kaggle+API/teste kaggle_treino_dataset

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

'''
def f1_score(y_true, y_pred):
    
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
'''
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

    optimizer = keras.optimizers.RMSprop()

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    #Alterei o formato que o modelo está sendo salvo. Aparentemente o formato precisa ser ".h5"

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.05, patience=2, mode='max')
    
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=checkpoint+".keras", 
                                                 save_best_only=True, 
                                                 monitor="val_accuracy"), 
                 keras.callbacks.EarlyStopping(monitor="val_accuracy", 
                                               patience=5), reduce_lr]
    epochs = 1 if debug else 50    
    history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, callbacks=callbacks)
    model = keras.models.load_model(checkpoint+".keras") # custom object retirado junto com a função que calcula f1 score
    return model.evaluate(validation_dataset), history
