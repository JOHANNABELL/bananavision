from pathlib import Path
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def training_exportmodel():
    df = pd.read_csv('../../dataset/features_bananes.csv')

    colonnes_features = [
        "moyenne_R",
        "moyenne_V",
        "moyenne_B",
        "std_R",
        "std_V",
        "std_B",
        "pct_vert",
        "pct_jaune",
        "pct_sombre",
        "pct_blanc",
        "moyenne_H",
        "moyenne_S",
        "circularite",
        "aire_relative",
        "solidite"
    ]

    x_train = df[colonnes_features]
    y_train = df['classe']

    modele_gnb = GaussianNB()
    modele_gnb.fit(x_train, y_train)

    fichier_modele = 'modele_classification_nb.pkl'
    joblib.dump(modele_gnb, fichier_modele)
    print(f"\nmodèle entrainé et sauvegardé avec succès sous : {fichier_modele}")

def evaluation_model(dataframe, model_prediction):
    y_true = dataframe['classe']

    classes = modele_charge.classes_
    matrice = confusion_matrix(y_true, model_prediction, labels=classes)

    print("\n--- Matrice de Confusion (Texte) ---")
    print(f"Ordre des classes : {classes}")
    print(matrice)

    print("\nGénération du graphique de la matrice de confusion pour l'evaluation de la classification Bayes naïf...")
    affichage = ConfusionMatrixDisplay(confusion_matrix=matrice, display_labels=classes)
    affichage.plot(cmap=plt.cm.Blues)

    plt.title("Matrice de Confusion - Classification des Bananes")

    plt.show()

if __name__ == '__main__':
    """
    training_exportmodel()

    """
    script = Path(__file__).parent
    modele_save = script / 'modele_classification_nb.pkl'
    modele_charge = joblib.load(modele_save)

    chemin_test = script / '../../dataset/features_bananes_validation.csv'
    df_test = pd.read_csv(chemin_test)

    colonnes_prises = [
        "moyenne_R",
        "moyenne_V",
        "moyenne_B",
        "std_R",
        "std_V",
        "std_B",
        "pct_vert",
        "pct_jaune",
        "pct_sombre",
        "pct_blanc",
        "moyenne_H",
        "moyenne_S",
        "circularite",
        "aire_relative",
        "solidite"
    ]

    nouvelles_donnees = df_test[colonnes_prises]
    predictions_test = modele_charge.predict(nouvelles_donnees)
    
    evaluation_model(df_test, predictions_test)