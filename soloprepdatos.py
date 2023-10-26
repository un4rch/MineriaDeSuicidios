import numpy as np
import pandas as pd
import nltk
import emoji
import emot
import sys
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

#al hacer test --> algunas muestras del train, algunas muestras del test que se parezcan pero sean distintas, algunas muestras que no tengan ningún significado en distinto nivel
#Mostrar por cada una de las muestras a que cluster pertenecen, por los 5 puntos mas cercanos --> que id tiene, la distancia y la frase sin preprocesar.

dtPinnata = pd.read_csv(sys.argv[1])
print(dtPinnata.head())

def llenar_vacios(X):
    if X == '':
        X = np.nan
    return X

def convertirEmojis(texto):  # convierte un emoji en un conjunto de palabras en inglés que lo representa. Si switch es False, entonces se eliminan los emojis
        texto = emoji.demojize(texto)
        diccionario_emojis = emot.emo_unicode.EMOTICONS_EMO
        for emoticono, texto_emoji in diccionario_emojis.items():
            texto = texto.replace(emoticono, texto_emoji)
        return texto

def eliminarSignosPuntuacion(texto):  # dado un string, devuelve el mismo string eliminando todos los caracteres que no sean alfabéticos
        textoNuevo = ""
        for caracter in texto:  # por cada caracter en el texto
            if caracter == "_":  # si es una barra baja, entonces se traduce como espacio
                textoNuevo = textoNuevo + " "
            if caracter.isalpha() or caracter == " ":  # si pertenece al conjunto de letras del alfabeto, se engancha a "textoNuevo"
                textoNuevo = textoNuevo + caracter
        return(textoNuevo)

def eliminarStopWords(texto):  # dado un string, elimina las stopwords de ese string
        texto = word_tokenize(texto, language='english')
        textoNuevo = ""
        for palabra in texto:
            if palabra not in stopwords.words('english'):
                textoNuevo = textoNuevo + " " + palabra
        return(textoNuevo)

def normalizarTexto(texto):  # dado un string que contenga palabras, devuelve un string donde todas las letras sean minúsculas
        return(texto.lower())

def aux_lematizar(palabra):
    tag = nltk.pos_tag([palabra])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
    
def lematizar(texto):  # dado un string, lematiza las palabras de ese string
    texto = nltk.word_tokenize(texto)

    # Inicializar el lematizador
    lemmatizer = WordNetLemmatizer()

    # Lematizar cada palabra y agregarla a una lista
    palabras_lematizadas = []
    for palabra in texto:
        pos = aux_lematizar(palabra)
        palabra_l = lemmatizer.lemmatize(palabra, pos=pos)
        palabras_lematizadas.append(palabra_l)

    # Unir las palabras lematizadas en un solo string y devolverlo
    texto_lematizado = ' '.join(palabras_lematizadas)
    return texto_lematizado

def preprocesarLenguajeNatural(pColumna):  # realiza todo el preproceso de un string en el orden correcto
        linea = str(pColumna)
        linea = convertirEmojis(linea)
        linea = eliminarSignosPuntuacion(linea)
        linea = normalizarTexto(linea)
        linea = eliminarStopWords(linea)
        linea = lematizar(linea)
        return linea

x,y = np.asarray(dtPinnata["text"]),np.asarray(dtPinnata["class"])

label_map = {cat:index for index,cat in enumerate(np.unique(y))}
y_prep = np.asarray([label_map[l] for l in y])
print(y_prep)

dtPinnata['texto_limpio'] = dtPinnata['text'].apply(preprocesarLenguajeNatural)
dtPinnata['tokens'] = dtPinnata['texto_limpio'].apply(word_tokenize)

dtPinnata.to_csv('datos_limpios.csv', index=False)