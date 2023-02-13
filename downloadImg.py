

import json
import requests

import os
import sys

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

########################################

def baixarImgs(dados):
    lstError = {}
    for item in dados:
        for raca, links in item.items():
            lst = []
            
            dir = './racas/'+ raca
            
            if not(os.path.isdir(dir)):
                os.mkdir(dir)
            if (len(os.listdir(dir)) == 0):
                for link in links:
                    nomeArq = dir+"/" + link.split("/").pop()
                    r = requests.get(link, stream=True)
                    try:
                        with open(nomeArq, 'wb') as f:
                            f.write(r.content)
                    except:
                        lst.append(nomeArq)
            else: 
                for link in links:
                    nomeArq = dir+"/" + link.split("/").pop()
                    if not(os.path.isfile(nomeArq)):
                        r = requests.get(link, stream=True)
                        try:
                            with open(nomeArq, 'wb') as f:
                                f.write(r.content)
                        except:
                            lst.append(nomeArq)
            
            if (len(lst)!=0):
                lstError[raca] = lst             
                        
            print(f"Raca: {raca} \nImagens: {links}\nTotal de Imagens: {len(links)}\nImgs n baixadas: {lst}\n")
    
    return lstError


def main():

  with open('public/lista.json', encoding='utf-8') as meu_json:
    dados = json.load(meu_json)
    
  if not(os.path.isdir('racas')):
    os.mkdir('./racas')

  racasSemLink = dados.pop(0)
  imgsNBaixadas = baixarImgs(dados)
  
  print(racasSemLink)
  print(f'Total de racas: {len(dados)}')
  print(f'Não baixadas: {imgsNBaixadas}')
  meu_json.close()


if __name__ == '__main__':
    main()