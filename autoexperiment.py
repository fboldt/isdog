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

# Initialize the image classifier.
clf = autokeras.ImageClassifier(overwrite=True, max_trials=1)
# Feed the image classifier with training data.
clf.fit(training, epochs=10)

# Predict with the best model.
predicted_y = clf.predict(validation)
print(predicted_y)


# Evaluate the best model with testing data.
print(clf.evaluate(validation))
