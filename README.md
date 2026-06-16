# Exemple d'une representation graphique frontend pour la partie dashboard a appliquer:
### Le frontend (par exemple une page HTML avec Chart.js ou une application Streamlit) peut :

Appeler GET /dashboard toutes les 5 secondes (ou après chaque upload).

Mettre à jour :

    Un camembert avec action_distribution. 
    Un graphique en barres avec filiere_distribution.
    Un graphique linéaire avec time_series (abscisse = timestamp, ordonnée = nombre).
    Une jauge pour la confiance moyenne et le temps moyen.
    Un tableau listant dernieres_analyses.
    Un histogramme des temps d’inférence avec temps_histogramme.

# Note bien bien vouloir changer dans le fichier pipeline le lien vers les differents modeles et fichier pour la compilation MERCI LIGNE[16-20] et mettre dans ce repertoire le code data_processing aussi c'est dans la branche A dossier dataset !!

## Installation packages necessaires
    Installer les packages avec pip install -r requirements.txt