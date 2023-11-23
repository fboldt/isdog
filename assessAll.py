from assessArchitecture import assessArchitecture
from tensorflow import keras
import numpy as np

# https://keras.io/api/applications/
architectures = [
    ("Xception", keras.applications.xception.Xception, keras.applications.xception.preprocess_input),
    ("VGG16", keras.applications.vgg16.VGG16, keras.applications.vgg16.preprocess_input), 
    ("VGG19", keras.applications.vgg19.VGG19, keras.applications.vgg19.preprocess_input), 
    ]

for checkpoint, architecture, preprocess_input in architectures:
    scores, history = assessArchitecture(architecture, preprocess_input, checkpoint+".best")
    np.savetxt("scores/"+checkpoint+".csv", scores)
    print(scores)
