
from assessArchitecture import assessArchitecture
from tensorflow import keras
import numpy as np

# from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
# from convnets.models import ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge

# https://keras.io/api/applications/

# As outras arquiteturas estão comentadas, pois, estou utilizando apenas a Xception para rodar os experimentos para o TCC da Amanda.

architectures = [
    ("Xception", keras.applications.xception.Xception, keras.applications.xception.preprocess_input)
    # ("VGG16", keras.applications.vgg16.VGG16, keras.applications.vgg16.preprocess_input),
    # ("VGG19", keras.applications.vgg19.VGG19, keras.applications.vgg19.preprocess_input),
    # ("ResNet50", keras.applications.resnet50.ResNet50, keras.applications.resnet50.preprocess_input),
    # ("ResNet50V2", keras.applications.resnet_v2.ResNet50V2, keras.applications.resnet_v2.preprocess_input),
    # ("ResNet101", keras.applications.resnet.ResNet101, keras.applications.resnet.preprocess_input),
    # ("ResNet101V2", keras.applications.resnet_v2.ResNet101V2, keras.applications.resnet_v2.preprocess_input),
    # ("ResNet152", keras.applications.resnet.ResNet152, keras.applications.resnet.preprocess_input),
    # ("ResNet152V2", keras.applications.resnet_v2.ResNet152V2, keras.applications.resnet_v2.preprocess_input),
    # ("InceptionV3", keras.applications.inception_v3.InceptionV3, keras.applications.inception_v3.preprocess_input),
    # ("InceptionResNetV2", keras.applications.inception_resnet_v2.InceptionResNetV2, keras.applications.inception_resnet_v2.preprocess_input),
    # ("MobileNet", keras.applications.mobilenet.MobileNet, keras.applications.mobilenet.preprocess_input),
    # ("MobileNetV2", keras.applications.mobilenet_v2.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
    # ("DenseNet121", keras.applications.densenet.DenseNet121, keras.applications.densenet.preprocess_input),
    # ("DenseNet169", keras.applications.densenet.DenseNet169, keras.applications.densenet.preprocess_input),
    # ("DenseNet201", keras.applications.densenet.DenseNet201, keras.applications.densenet.preprocess_input),
    # ("NASNetMobile", keras.applications.nasnet.NASNetMobile, keras.applications.nasnet.preprocess_input),
    # ("NASNetLarge", keras.applications.nasnet.NASNetLarge, keras.applications.nasnet.preprocess_input),
    # ("EfficientNetB0", EfficientNetB0, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetB1", EfficientNetB1, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetB2", EfficientNetB2, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetB3", EfficientNetB3, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetB4", EfficientNetB4, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetB5", EfficientNetB5, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetB6", EfficientNetB6, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetB7", EfficientNetB7, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetV2B0", keras.applications.efficientnet.EfficientNetB0, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetV2B1", keras.applications.efficientnet.EfficientNetB1, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetV2B2", keras.applications.efficientnet.EfficientNetB2, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetV2B3", keras.applications.efficientnet.EfficientNetB3, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetV2S", keras.applications.efficientnet.EfficientNetS, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetV2M", keras.applications.efficientnet.EfficientNetM, keras.applications.efficientnet.preprocess_input),
    # ("EfficientNetV2L", keras.applications.efficientnet.EfficientNetL, keras.applications.efficientnet.preprocess_input),
    # ("ConvNeXtTiny", ConvNeXtTiny, keras.applications.efficientnet.preprocess_input),
    # ("ConvNeXtSmall", ConvNeXtSmall, keras.applications.efficientnet.preprocess_input),
    # ("ConvNeXtBase", ConvNeXtBase, keras.applications.efficientnet.preprocess_input),
    # ("ConvNeXtLarge", ConvNeXtLarge, keras.applications.efficientnet.preprocess_input),
    # ("ConvNeXtXLarge", ConvNeXtXLarge, keras.applications.efficientnet.preprocess_input),
]

for checkpoint, architecture, preprocess_input in architectures:
    scores, history = assessArchitecture(architecture, preprocess_input, checkpoint+"kaggleAPI_model_1604.best")
    np.savetxt("scores/"+checkpoint+".csv", scores)
    print(scores)
