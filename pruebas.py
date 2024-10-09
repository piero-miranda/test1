import os
import requests

# Verificar si el archivo existe, y si no, descargarlo
if not os.path.exists('SegNet_trained.h5'):
    url = 'https://github.com/piero-miranda/test1/raw/main/SegNet_trained.h5'  # Enlace directo al archivo
    r = requests.get(url, allow_redirects=True)

    # Guardar el archivo en el entorno
    with open('SegNet_trained.h5', 'wb') as f:
        f.write(r.content)

    print("Archivo SegNet_trained.h5 descargado correctamente.")
