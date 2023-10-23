import sys
import os
import csv

# Ruta de la carpeta que contiene los documentos de texto
carpeta = sys.argv[1]

# Inicializar una lista para almacenar todos los datos de los documentos de texto
texto = []
texto.append("text")

# Iterar a través de los archivos en la carpeta
for documento in os.listdir(carpeta):
    if documento.endswith('.txt'):
        filepath = os.path.join(carpeta, documento)
        with open(filepath, 'r', encoding='utf-8') as file:
            # Leer el contenido del archivo de texto y agregarlo a la lista
            contenido = file.read()
            texto.append(contenido[:-1])

archivo_csv = 'convertido.csv'

# Escribir los datos en un archivo CSV
with open(archivo_csv, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    for linea in texto:
        csvwriter.writerow([linea])
