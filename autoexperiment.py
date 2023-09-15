import autokeras

training = autokeras.image_dataset_from_directory(
    "kaggle",
    batch_size=32,
    color_mode="rgb",
    image_size=(256, 256),
    interpolation="bilinear",
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="training",
)
validation = autokeras.image_dataset_from_directory(
    "kaggle",
    batch_size=32,
    color_mode="rgb",
    image_size=(256, 256),
    interpolation="bilinear",
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="validation",
)

# Initialize the ImageClassifier
clf = autokeras.ImageClassifier(overwrite=True, max_trials=10)

# Search for the best model architecture
clf.fit(training, epochs=10)

# Get the best model found during the search
best_model = clf.export_model()

# Evaluate the best model with validation data.
print(clf.evaluate(validation))