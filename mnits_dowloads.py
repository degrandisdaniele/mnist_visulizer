import os
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. Definizione del nome della cartella di output
OUTPUT_FOLDER = "public"
NUM_IMAGES_TO_DOWNLOAD = 100

def create_and_download_mnist_images():
    """
    Crea una cartella 'public' e scarica 100 immagini casuali
    del dataset MNIST al suo interno.
    """
    
    # Crea la cartella di output se non esiste
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Cartella '{OUTPUT_FOLDER}' creata con successo.")
    else:
        print(f"La cartella '{OUTPUT_FOLDER}' esiste già.")

    # Carica il dataset MNIST
    # Viene caricato solo il set di training per selezionare le immagini
    print("Caricamento del dataset MNIST...")
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    print("Dataset MNIST caricato.")

    # Normalizza le immagini a 0-255 (se non lo fossero già o se fossero float)
    # MNIST le scarica già come uint8 da 0 a 255, quindi questa riga assicura il tipo corretto.
    train_images = train_images.astype(np.uint8)

    # Seleziona 100 indici casuali
    num_total_images = len(train_images)
    random_indices = np.random.choice(num_total_images, NUM_IMAGES_TO_DOWNLOAD, replace=False)

    print(f"Scaricamento di {NUM_IMAGES_TO_DOWNLOAD} immagini casuali...")
    for i, idx in enumerate(random_indices):
        image_data = train_images[idx]
        label = train_labels[idx]

        # Crea un oggetto immagine PIL
        # La modalità 'L' sta per immagini in scala di grigi (Luminosity)
        image = Image.fromarray(image_data, mode='L')

        # Costruisci il nome del file
        # Usiamo il formato f-string per rendere il nome chiaro e numerato
        filename = os.path.join(OUTPUT_FOLDER, f"mnist_image_{i+1:03d}_label_{label}.png")
        
        # Salva l'immagine
        image.save(filename)
        print(f"  Salvataggio: {filename}")

    print(f"Completato! {NUM_IMAGES_TO_DOWNLOAD} immagini salvate nella cartella '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    create_and_download_mnist_images()