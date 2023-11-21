from assessArchitecture import assessArchitecture
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16

architectures = [VGG16, Xception]

for architecture in architectures:
    print(assessArchitecture(architecture))
