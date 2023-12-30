import os, pathlib
import shutil
import pandas as pd

nome_diretorio = 'kaggle'

train_ori = pathlib.Path('dog-breed-identification/train')
test_ori = pathlib.Path('dog-breed-identification/test')

if not os.path.exists(nome_diretorio):
    os.makedirs(nome_diretorio)
diretorio_destino = pathlib.Path(nome_diretorio)

def move_images(image, label, dest_dir, source_dir):
    _ = os.path.join(dest_dir, label)
    if not os.path.exists(_):
        os.makedirs(_)
    source = os.path.join(source_dir, f'{image}.jpg')
    destination = os.path.join(dest_dir, label, f'{image}.jpg')
    if os.path.exists(source) and not os.path.exists(destination):
        shutil.copyfile(source, destination)

labels = pd.read_csv('./dog-breed-identification/labels.csv').to_numpy()
for image, label in labels:
    move_images(image, label, diretorio_destino, train_ori)
