from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras import metrics
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from keras.preprocessing import image
import numpy as np
import tensorflow.keras.backend as K
import json


def make_subset(subset="testing", directory="kaggle_teste"):
    return image_dataset_from_directory(directory,
                                        image_size=(180, 180),
                                        seed=42,
                                        batch_size=1)

def predict_with_model(model_path, test_dataset):
    metrics = {'Accuracy':[], 'Recall':[], 'Precision':[], 'F1':[]}
    
    # Carregar o modelo treinado
    model = keras.models.load_model(model_path)
	
    for x, y in test_dataset:
        predictions = model.predict(x)
        
        predictions = np.argmax(predictions, axis = -1)

        print(y, predictions)
        
        
        results = calc_metrics(y, predictions)
        metrics['Accuracy'].append(results[0])
        metrics['Recall'].append(results[1])
        metrics['Precision'].append(results[2])
        metrics['F1'].append(results[3])
        
    for key, values in metrics.items():
        metrics[key] = np.mean(values)
    
    return metrics

def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((prec*recall)/(prec+recall+K.epsilon()))

def calc_metrics(test_dataset, predictions):

    accuracy = accuracy_score(test_dataset, predictions)
    #cm = keras.metrics.confusion_matrix(test_dataset, predictions)

    rc = recall_score(test_dataset, predictions, average='macro', zero_division = 1)

    prec = precision_score(test_dataset, predictions, average='macro', zero_division = 1)

    f1 = 2*((prec*rc)/(prec+rc+K.epsilon()))

	# print(accuracy, rc, prec, f1)
    return (accuracy, rc, prec, f1)


# Exemplo de uso
model_path = "XceptionkaggleScraper_manual.best.keras"
# model_name = "kaggle"
test_directory = "../../dogscraper/isdog/test_dataset"
metrics = {'Accuracy':0, 'Recall':0, 'Precision':0, 'F1':0}

test_dataset = make_subset(subset="testing", directory=test_directory)

# steps = int((120*4)/8) # total de imgs dividido pelo tamanho dos batchs

# metrics = predict_with_model(model_path, test_dataset)
# print(metrics)
model = keras.models.load_model(model_path)
#results = model.evaluate(test_dataset)

metrics = predict_with_model(model_path, test_dataset)

#metrics['Accuracy'] = results[0]
#metrics['Recall'] = results[1]
#metrics['Precision'] = results[2]
#metrics['F1'] = results[3]


print('saving final file!')

with open("metrics_scores/kaggleScraper_manual.json", 'w') as json_file:
    json.dump(metrics, json_file)
print("file saved with success!")


