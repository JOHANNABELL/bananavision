# Module D - Analyse ethique et sociale

Filière Banane d'Exportation — Direction Innovation & Qualite

## Bananavision

Système Intelligent de Contrôle Qualite et de Tri Automatise

MODULE D

## Analyse Éthique Et Sociale

Niveau

3e annee Cycle Ingenieur — Specialite Informatique

Table des matières

## Introduction — Le Module D comme Fil Conducteur Éthique3

## Partie 1 — Impact Emploi et Reconversion4

1.1 Quantification des Postes Impactes4

1.2 Profils des Agents Concernes4

1.3 Plan de Reconversion — Ancrage OHADA et RSE4

Niveau legal — OHADA, Acte Uniforme 20175

Niveau RH Interne — Plan de Reconversion en 3 Étapes5

Niveau RSE — Compagnie Fruitière5

## Partie 2 — Biais Algorithmiques et Équite6

2.1 Identification des Biais dans le Banana Disease Dataset6

2.2 Discrimination Potentielle entre Lots selon l'Origine dans la Plantation6

2.3 Strategies de Mitigation Concrètes7

Mesure 1 — Augmentation de Donnees Ciblee (court terme)7

Mesure 2 — Fine-Tuning sur Donnees Locales (moyen terme — BananaVision v2)7

Mesure 3 — Audit de Performance par Lot d'Origine (systematique, en production)7

## Partie 3 — Gouvernance, Responsabilite et Traçabilite8

3.1 Cartographie des Responsabilites en Cas d'Erreur8

3.2 Schema de Gouvernance Compatible GlobalG.A.P.8

3.3 Lien avec les Exigences des Acheteurs Europeens9

## Carnet de Bord — 4 Entrees Journalières10

## Conclusion — BananaVision comme Système Sociotechnique Responsable12

## Introduction — Le Module D comme Fil Conducteur Éthique

Le Module D n'est pas une annexe ajoutee après coup : il constitue le fil conducteur ethique de BananaVision, traversant chaque choix des Modules A, B et C. Toute decision technique produit un effet social que cette analyse entend nommer et gouverner.

Trois correspondances techniques-sociales structurent cette analyse :

**Module Technique**

Decision IA

Pendant Social

Module A2 — CNN

Classification 'rejete'

Declenchement d'une alerte sur le poste d'un agent de tri

Module B — K-Means

Redefinition des clusters de defauts

Redefinition des filières et flux de travail humains

Module C — MDP

Politique optimale π*

Fixation mecanique du taux de remplacement humain

« BananaVision n'est pas un système qui remplace des humains — c'est un système dont les concepteurs doivent decider comment il coexiste avec eux. »

## Partie 1 — Impact Emploi et Reconversion

1.1 Quantification des Postes Impactes

Le pack-house de PHP compte ~50 agents de tri. Deux scenarios d'automatisation sont analyses :

Indicateur

Scenario A — Automatisation 60 %

Scenario B — Automatisation 100 %

Base d'effectifs

~50 agents de tri au pack-house PHP

~50 agents de tri au pack-house PHP

Postes automatises

30 postes sur convoyeur principal

50 postes entièrement automatises

Postes reconvertis

~20 agents redeployes : supervision, contrôle alertes MDP, maintenance cameras

2–3 techniciens IA/maintenance + 1 data analyst qualite

Postes supprimes

0 licenciement à court terme

47 à 48 postes supprimes ou reconvertis

Impact net

Redefinition des fiches de poste uniquement

Licenciements economiques — obligation legale OHADA

Lien Module C

L'action 'Suspendre' du MDP maintient une presence humaine dans la boucle

L'action 'Suspendre' devient quasi-absente — π* maximise la continuite automatisee

1.2 Profils des Agents Concernes

En l'absence de donnees RH exactes, les hypothèses suivantes sont posees et sourcees sur le profil-type du secteur agro-industriel camerounais (MINADER, 2022) :

Dimension RH

Hypothèse de travail

Source / Justification

Niveau de formation

Majoritairement BEPC ou sans diplôme formel

MINADER 2022 — profil agro-industrie Mungo

Tranche d'âge

25 à 45 ans

Hypothèse raisonnable — pack-house actif depuis 2008

Anciennete moyenne

3 à 7 ans

Hypothèse secteur (turn-over modere filière banane)

Genre

Activite historiquement feminine (60–70 %)

FAO 2023 — genre dans les chaînes de valeur banane Afrique

Competences numeriques

Faibles à nulles sur outils digitaux

Hypothèse à verifier — implique plan de formation specifique

1.3 Plan de Reconversion — Ancrage OHADA et RSE

Niveau legal — OHADA, Acte Uniforme 2017

L'Acte Uniforme OHADA (2017) impose les obligations suivantes en cas de licenciement economique :

Notification prealable à l'Inspection du Travail (delai minimum 30 jours avant toute mesure).

Indemnite de licenciement calculee sur l'anciennete (~20 % du salaire mensuel brut par annee d'anciennete).

Priorite de reembauche si PHP cree de nouveaux postes dans les 12 mois suivants.

Plan social obligatoire au-delà de 10 licenciements sur une même periode — applicable au Scenario B.

Niveau RH Interne — Plan de Reconversion en 3 Étapes

Étape

Horizon

Contenu

Acteur responsable

Étape 1 Detection precoce

Annee 1

Cartographie des competences existantes via bilan de competences individuel. Identification des agents à fort potentiel de reconversion.

DRH PHP + Équipe IA

Étape 2 Formation

Annees 1–2

Formation de 6 semaines aux metiers d'operateur de supervision (dashboards, alertes MDP, cas limites). Module pratique sur interface BananaVision.

Centre de formation Compagnie Fruitière + MINEFOP

Étape 3 Reaffectation

Annees 2–3

Reaffectation vers colisage, palettisation, gestion chaîne du froid — ou vers autres sites PHP.

Direction Exploitation PHP + Comite RSE

Niveau RSE — Compagnie Fruitière

Le Rapport RSE 2022 de PHP affirme un engagement explicite envers l'emploi local. Il est propose que BananaVision soit conditionne à un accord d'entreprise incluant :

Un fonds de reconversion abonde à hauteur de 2 à 3 % des gains d'efficience realises (~18 % des charges operationnelles directes selon le diagnostic de Mme Biyong-Essomba).

Un comite de suivi paritaire (Direction / Representants des agents) evaluant semestriellement l'avancement du plan.

Un engagement de non-licenciement sec pendant les 24 premiers mois suivant le deploiement du Scenario A.

## Partie 2 — Biais Algorithmiques et Équite

2.1 Identification des Biais dans le Dataset

Le dataset d'entraînement du CNN conditionne toutes les decisions en aval. Quatre axes de biais ont ete identifies :

Type de biais

Criticite

Description

Impact sur BananaVision

Biais geographique

CRITIQUE

Dataset constitue principalement en Asie du Sud-Est et Amerique latine. Conditions d'eclairage (lumière naturelle, fond neutre) très differentes du pack-house camerounais (neons industriels, fond convoyeur en mouvement, poussière).

Le CNN risque de mal generaliser aux bananes du Mungo photographiees en conditions reelles PHP.

Biais de maturite

## Important

Sur-representation potentielle des classes 'très mûres' ou 'avec moisissures'. Verifier le ratio classes majoritaires/minoritaires.

Biais vers les rejets — taux de faux negatifs plus eleve, coût economique direct pour PHP.

Biais de variete

MODÉRÉ

Images du dataset pouvant melanger Cavendish, plantain et Lady Finger. Features colorimetriques inadaptees à la Grande Naine de PHP.

Risque de declassement de fruits PHP conformes sur la base de caracteristiques propres à d'autres varietes.

Biais d'annotation

MODÉRÉ

Étiquettes posees par crowdsourcing ? Presence de classes 'bruyantes' (labels incertains ou mal definis).

Les erreurs d'annotation se propagent dans les recompenses R(s,a) du MDP, biaisant la politique π*.

2.2 Discrimination Potentielle entre Lots selon l'Origine dans la Plantation

La question centrale : les biais algorithmiques peuvent-ils creer une discrimination systemique entre les parcelles nord et sud du Mungo ?

Les parcelles nord et sud presentent potentiellement des micro-variations agroclimatiques (humidite, ensoleillement, altitude) influençant la teinte des bananes à maturite identique.

Si le CNN n'a pas appris à gerer ces variations specifiques à PHP, il peut systematiquement declasser les lots d'une parcelle geographique donnee, non pas en raison d'un defaut reel, mais par inadequation du modèle.

Cette discrimination non intentionnelle constitue un biais systemique par manque de representativite — phenomène documente sous le terme de disparate impact non intentionnel dans la litterature sur les biais algorithmiques.

2.3 Strategies de Mitigation Concrètes

Mesure 1 — Augmentation de Donnees Ciblee (court terme)

Utiliser albumentations pour generer des images synthetiques simulant les conditions du pack-house PHP : RandomBrightnessContrast et HueSaturationValue pour varier l'eclairage, ajout de bruit simulant la poussière et le mouvement du convoyeur. Directement liee aux choix de bibliothèque (section 3.2 du rapport technique).

Mesure 2 — Fine-Tuning sur Donnees Locales (moyen terme — BananaVision v2)

Collecter 200 à 500 photographies de bananes PHP directement au pack-house du Mungo dans les conditions reelles. Ces images serviront de jeu de donnees pour un fine-tuning du CNN. Ce compromis realiste constitue la feuille de route prioritaire pour la version 2 du système, à planifier dans la strategie PHP 2027.

Mesure 3 — Audit de Performance par Lot d'Origine (systematique, en production)

Integrer dans le tableau de bord un indicateur de taux de rejet par parcelle d'origine. Si une parcelle est systematiquement plus rejetee sur 4 semaines, un audit humain est automatiquement declenche. Cette mesure transforme le biais potentiel en signal exploitable — preuve de diligence raisonnable opposable aux acheteurs europeens.

## Partie 3 — Gouvernance, Responsabilite et Traçabilite

3.1 Cartographie des Responsabilites en Cas d'Erreur

Règle fondamentale : BananaVision est un système d'aide à la decision, non un decideur autonome. La decision finale de shipment reste humaine, tracee et opposable.

Scenario d'erreur

Cause probable

Responsable designe

Mesure corrective

Lot conforme rejete (faux negatif) → Perte de revenu PHP

Biais dataset + seuil MDP trop strict

Équipe IA + Directrice Qualite PHP (Mme Biyong-Essomba)

Revision seuil MDP — comite trimestriel. Log d'audit produit la preuve.

Lot non-conforme accepte → Rejet client à Marseille ou Anvers

Taux d'erreur CNN > seuil contractuel

Directrice Qualite PHP + Compagnie Fruitière

Indemnisation client. Fine-tuning CNN d'urgence sur donnees locales.

Action 'Suspendre' trop frequente → Ralentissement convoyeur

Facteur γ du MDP mal calibre

Équipe IA BananaVision

Analyse de sensibilite γ — reglage sous supervision agent.

3.2 Schema de Gouvernance Compatible GlobalG.A.P.

La gouvernance proposee s'articule en quatre couches emboîtees :

COUCHE 1 — Operationnelle

L'agent superviseur au pack-house valide les alertes 'Suspendre' en temps reel. Il dispose d'un bouton de veto capable d'overrider toute decision automatique. Sans sa validation, aucune action irreversible (rejet definitif d'un lot) n'est possible. Ce rôle humain dans la boucle est non-negociable.

COUCHE 2 — Qualite

La Directrice Qualite, Mme Christelle Biyong-Essomba, reçoit chaque matin un tableau de bord consolide : taux de classification par filière, taux de rejet par parcelle d'origine, nombre d'alertes 'Suspendre' declenchees et validees. Elle dispose d'un accès direct pour modifier les seuils operationnels du MDP.

COUCHE 3 — Traçabilite GlobalG.A.P.

Chaque decision est loggee avec : timestamp NTP precis, etat du fruit (cluster_id + score CNN), action prise, identifiant de l'agent superviseur ayant valide ou exerce son veto. Ce journal d'audit constitue la preuve documentaire exigee par GlobalG.A.P. IFA v6.0 et les cahiers des charges des GMS europeennes.

COUCHE 4 — Revision Periodique

Comite trimestriel reunissant l'equipe IA, la Direction Qualite, un representant des agents de supervision et un RH. Il revise les seuils du MDP, examine les audits par lot et valide les evolutions du CNN. Toutes ses decisions sont documentees et archivees.

3.3 Lien avec les Exigences des Acheteurs Europeens

Ce schema à quatre couches repond directement aux exigences des grandes distributions europeennes (France, Belgique, Royaume-Uni) :

Traçabilite lot par lot : chaque decision loggee (Couche 3) permet de reconstituer le parcours complet d'un lot, du champ au conteneur, en cas de reclamation client.

Preuve de contrôle humain : le bouton de veto (Couche 1) et la validation quotidienne (Couche 2) constituent des preuves documentees d'un contrôle humain effectif — exigence croissante des audits GlobalG.A.P. IFA v6.0 (2023).

Donnees chiffrees opposables : le tableau de bord produit automatiquement les metriques — taux de rejet, confiance CNN par lot, contrôles manuels effectues. Cette transparence algorithmique represente un avantage concurrentiel que le système manuel ne permettait pas.

## Carnet de Bord — 4 Entrees Journalières

Le carnet de bord constitue la trace reflexive du projet BananaVision. Chaque entree documente une decision technique et la met en regard de la question ethique qu'elle soulève.

JOUR 1 — Module A : Classification Supervisee

Module principal : A1 — Naïf Bayes (Baseline) + selection du dataset

Decision technique prise

Selection du Banana Disease Dataset après comparaison de 4 candidats (Kaggle Banana Quality, PlantVillage, Zenodo Tropical Fruits, HuggingFace Fruit Classification). Choix justifie par le double etiquetage (type + qualite). Decision d'utiliser MobileNetV2 en transfer learning (Keras) — contrainte de dataset de taille modeste.

Question ethique soulevee

La variete Cavendish Grande Naine cultivee à PHP est-elle suffisamment representee dans le dataset ? Le biais geographique (images Asie/Amerique latine) ne risque-t-il pas de penaliser systematiquement les lots des parcelles sud du Mungo, dont les conditions d'ensoleillement produisent une teinte legèrement plus doree ?

Reponse ou piste envisagee

Decision de constituer un corpus complementaire de 50 photos PHP prises au pack-house pendant la semaine de projet. Ces images serviront à un test de generalisation du CNN — la difference de performance constitue la mesure empirique du biais geographique documentee en Partie 2.

JOUR 2 — Module B : Clustering Non Supervise

Module principal : B — K-Means sur embeddings CNN (choix k = 4)

Decision technique prise

Application de K-Means avec k = 4 clusters après analyse de la courbe du coude (inertie) et du score silhouette. Les 4 clusters correspondent empiriquement aux filières PHP : export Grade 1, consommation locale, transformation (fruit trop mûr), quarantaine phytosanitaire (tache fongique). Visualisation t-SNE produite pour la soutenance.

Question ethique soulevee

Le cluster 3 regroupe des fruits 'limite export' — quelle subjectivite l'agent humain introduisait-il avant BananaVision pour ces cas ambigus ? Un agent experimente de 7 ans dispose-t-il d'une connaissance contextuelle (meteo de la semaine, comportement de la parcelle) que le CNN ignore structurellement ?

Reponse ou piste envisagee

Piste retenue : signaler le 'cluster de frontière' (cluster 3) avec un indicateur de confiance reduit, declenchant systematiquement l'alerte 'Suspendre'. Cela preserve le rôle decisionnel humain pour les cas ambigus — compromis entre efficience et preservation de l'expertise terrain.

JOUR 3 — Module C : Modelisation MDP

Module principal : C — Definition des etats S, actions A, calibration de γ

Decision technique prise

Definition de 12 etats (3 clusters × 2 niveaux de confiance CNN × presence/absence alerte). Calibration initiale γ = 0.85 après analyse de sensibilite entre 0.7 et 0.95. La politique π* resultante favorise l'action 'Suspendre' pour les etats à confiance CNN faible — comportement prudent attendu.

Question ethique soulevee

L'action 'Suspendre pour contrôle manuel' a un coût operationnel reel (ralentissement du convoyeur, mobilisation d'un agent). Si γ est trop eleve, le MDP minimise cette action et reduit la presence humaine dans la boucle. Quelle est la valeur minimale de γ maintenant un taux de contrôle manuel acceptable pour la Direction Qualite PHP ?

Reponse ou piste envisagee

Analyse de sensibilite realisee : γ = 0.80 donne un taux de 'Suspendre' de 12 % (proche du taux d'erreur humain de 8–12 % selon Mme Biyong-Essomba). γ = 0.90 descend à 4 %. Proposition : fixer γ = 0.82 par defaut, avec revision trimestrielle (Couche 4) basee sur les donnees de production.

JOUR 4 — Module D : Integration Éthique et Tests

Module principal : D — Redaction du Module D + tests d'audit de performance par lot

Decision technique prise

Redaction de la cartographie des responsabilites (Partie 3.1) et des strategies de mitigation des biais (Partie 2.3). Tests du pipeline d'audit de performance par lot : simulation d'un biais fictif sur 'parcelle_nord' detecte après 3 semaines de donnees simulees.

Question ethique soulevee

En simulant un biais sur la 'parcelle_nord', le système l'a detecte en 3 semaines. Mais que se passe-t-il si PHP etend BananaVision à ses autres sites avant ce delai de detection ? Le biais peut-il se propager à l'ensemble de la filière avant d'être identifie ?

Reponse ou piste envisagee

Proposition d'une règle de deploiement progressif : validation sur un seul site pendant 3 mois minimum avant extension. Ce principe sera inclus dans les recommandations finales et presente au jury PHP comme condition sine qua non d'un deploiement responsable.

## Conclusion — BananaVision comme Système Sociotechnique Responsable

L'analyse conduite dans ce Module D confirme que BananaVision ne peut pas être evalue uniquement sur ses metriques de performance technique. Chaque indicateur algorithmique est le revers d'une realite sociale qu'il faut nommer, mesurer et gouverner.

Trois enseignements structurants se degagent :

L'automatisation n'est pas un etat binaire. Entre 60 % et 100 % d'automatisation, il existe un continuum de decisions organisationnelles relevant de choix politiques d'entreprise autant que de performances algorithmiques. La Direction de PHP dispose de leviers reels — à condition de les activer dès le deploiement du prototype.

Un biais de dataset est une decision ethique non faite. Deployer BananaVision sans fine-tuning sur donnees locales, c'est prendre le risque de penaliser systematiquement des lots — et les agriculteurs associes — sur la base d'images collectees à 12 000 kilomètres du Mungo. La mesure corrective est techniquement faisable et ethiquement necessaire.

La gouvernance à quatre couches est un avantage concurrentiel. Face aux exigences croissantes des acheteurs europeens, PHP peut transformer son dispositif en argument commercial differenciateur : contrôle humain dans la boucle, auditabilite complète, revision periodique documentee.
