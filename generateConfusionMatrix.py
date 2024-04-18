import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
import os

def make_subset(subset="testing", directory="kaggle_teste"):
    return image_dataset_from_directory(directory,
                                        image_size=(180, 180),
                                        seed=42,
                                        batch_size=8)

def generate_matrix(model, test_dataset, n_classes=120):
    cm = np.zeros((n_classes,n_classes), dtype = np.uint32)
    
    for x, y in test_dataset:
        predictions = model.predict(x)
        
        predictions = np.argmax(predictions, axis = -1)

        for pred, gt in zip(predictions, y):
            cm[gt][pred]+=1
            
    return cm

# Exemplo de uso
model_path = "models/kaggleAPI_model.keras"
# model_name = "kaggle"
test_directory = "test_dataset"
metrics = {'Accuracy':[], 'Recall':[], 'Precision':[], 'F1':[]}

test_dataset = make_subset(subset="testing", directory=test_directory)

# steps = int((120*4)/8) # total de imgs dividido pelo tamanho dos batchs

# metrics = predict_with_model(model_path, test_dataset)
# print(metrics)
model = keras.models.load_model(model_path)

arquivos = os.listdir(test_directory)

# Conta o número de diretórios no caminho
num_pastas = sum(os.path.isdir(os.path.join(test_directory, arquivo)) for arquivo in arquivos)
# Gerar a matriz de confusão
cm = generate_matrix(model, test_dataset, n_classes = num_pastas)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)  # Ajustar o tamanho da fonte
sns.heatmap(cm, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Salvar a figura como um arquivo PNG
