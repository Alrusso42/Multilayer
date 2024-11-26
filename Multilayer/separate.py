import csv
import random
import os

def split_csv(input_csv, train_ratio=0.8, seed=42):
    # Dossier "dataset" où sont stockés les fichiers CSV
    dataset_dir = "dataset"

    # Assurer que le dossier dataset existe
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Création des chemins des fichiers de sortie
    train_csv = os.path.join(dataset_dir, "train.csv")
    valid_csv = os.path.join(dataset_dir, "valid.csv")

    random.seed(seed)

    # Lecture du fichier CSV d'entrée
    input_csv_path = os.path.join(dataset_dir, input_csv)
    with open(input_csv_path, 'r') as infile:
        data = list(csv.reader(infile))

    # Prétraitement : supprimer l'ID (colonne 1) et normaliser la colonne 'diagnosis' (colonne 2)
    for row in data:
        row.pop(0)  # Supprime l'ID
        row[0] = '1' if row[0] == 'M' else '0'  # M -> 1, B -> 0

    # Mélange aléatoire des données
    random.shuffle(data)

    # Division des données
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    valid_data = data[split_index:]

    # Écriture des données d'entraînement dans train.csv
    with open(train_csv, 'w', newline='') as trainfile:
        writer = csv.writer(trainfile)
        writer.writerows(train_data)

    # Écriture des données de validation dans valid.csv
    with open(valid_csv, 'w', newline='') as validfile:
        writer = csv.writer(validfile)
        writer.writerows(valid_data)

    # Message de confirmation
    print(f"Les fichiers {train_csv} et {valid_csv} ont été générés avec succès.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python split_csv.py <input_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    split_csv(input_csv)
