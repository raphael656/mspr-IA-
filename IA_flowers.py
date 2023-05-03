#IMPORT
from keras.models import load_model
from PIL import Image
import numpy as np
import time
import os
from flask import Flask, request

# On definit les chemins d'acces au différentes hyper parametre


# Classe permettant de réaliser une prédiction sur une nouvelle donnée



 
app = Flask(__name__)

@app.route('/')
def index():
    return 'Server Works!'

@app.route('/IA_flowers/' , methods=['POST'])
def predict():
    """
    # Fonction qui permet de convertir une image en array, de charger le modele et de lui injecter notre image pour une prediction
    :param modelPath: chemin du modèle au format hdf5
    :param imagePath: chemin de l'image pour realiser une prediction
    :param imageSize: défini la taille de l'image. IMPORTANT : doit être de la même taille que celle des images
    du dataset d'entrainements
    :param label: nom de nos 5 classes de sortie
    """
    
    path = os.getcwd()
    image = request.files['image']
    modelPath = path + '/Model/moModel2.hdf5'
    #imagePath =  path + '/image/tulipe.jpg'
    imageSize = (50,50)
    label = ['dahlia', 'marguerite', 'pissenlit', 'rose', 'tournesol', 'tulipe']

    start = time.time()

    # Chargement du modele
    print("Chargement du modèle :\n")
    model = load_model(modelPath)
    print("\nModel chargé.")

    #Chargement de notre image et traitement
    data = []
    img = Image.open(image)
    img.load()
    img = img.resize(size=imageSize)
    img = np.asarray(img) / 255.
    data.append(img)
    data = np.asarray(data)

    #On reshape pour correspondre aux dimensions de notre modele
    # Arg1 : correspond au nombre d'image que on injecte
    # Arg2 : correspond a la largeur de l'image
    # Arg3 : correspond a la hauteur de l'image
    # Arg4 : correspond au nombre de canaux de l'image (1 grayscale, 3 couleurs)
    
    dimension = data[0].shape

    #Reshape pour passer de 3 à 4 dimension pour notre réseau
    data = data.astype(np.float32).reshape(data.shape[0], dimension[0], dimension[1], dimension[2])

    #On realise une prediction
    prediction = model.predict(data)


    #On recupere le numero de label qui a la plus haut prediction
    maxPredict = np.argmax(prediction)

    #On recupere le mot correspondant à l'indice precedent
    word = label[maxPredict]
    pred = prediction[0][maxPredict] * 100.
    end = time.time()

    text = ''
    #On affiche les prédictions
    if word == 'rose' :
        text = 'La rose est la fleur du rosier, arbuste du genre Rosa et de la famille des Rosaceae. La rose des jardins se caractérise avant tout par la multiplication de ses pétales imbriqués, qui lui donne sa forme caractéristique.'
    if word == 'marguerite' : 
        text = "Plante herbacée vivace de la famille des Asteraceae (astéracées), originaire d’Eurasie et dont l’inflorescence est un grand capitule composé d’une couronne de ligules blanches autour d'un disque jaune."
    if word == 'tournesol' :
        text = "Espèce d’hélianthe à très grandes fleurs, dont les graines oléagineuses sont utilisées dans l’alimentation et les fleurs comme ornement, de nom scientifique Helianthus annuus. Le tournesol est surtout cultivé pour ses graines oléagineuses ; une variété, le soleil uniflore ou soleil de Russie, est particulièrement intéressante, surtout la variété à graines blanches."
    if word == 'pissenlit' :
        text = "Plante à fleurs composées, qui croît dans les lieux herbeux et incultes et dont les feuilles, à peu près semblables à celles de la chicorée, se mangent en salade, quand elles sont jeunes et tendres."
    if word == 'dahlia' : 
        text = "Astéracée qui porte des fleurs simples ou doubles, dont les tiges naissent en touffe et dont les racines sont des tubercules."
    if word == 'tulipe' :
        text = "Plante de la famille des liliacées, à racine bulbeuse, à tige haute, qui porte une belle fleur et dont il existe un très grand nombre de variétés."
        
    
    fleur = 'à  ' + str(pred)+"%" + "un(e)  " + str(word)
    #print()
    #print('----------')
    #print(" Prediction :")
    #for i in range(0, len(label)):
    #   print('     ' + label[i] + ' : ' + "{0:.2f}%".format(prediction[0][i] * 100.))

    #print()
    #print('RESULTAT : ' + word + ' : ' + "{0:.2f}%".format(pred))
    #print('temps prediction : ' + "{0:.2f}secs".format(end-start))

    #print('----------')
  
    return f"prediction : {fleur} \n description :{text}"

if __name__ == '__main__':
      app.run(host='0.0.0.0', port='8888',debug=True)