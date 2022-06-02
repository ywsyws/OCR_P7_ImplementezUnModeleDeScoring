# OpenClassrooms Project 7 - Implémentez un modèle de scoring

## 1. Introduction
Ce projet a été réalisé pour le [parcours Data Scientist de OpenClassrooms](https://openclassrooms.com/fr/paths/164-data-scientist). <br>
Il s'agit du sixième projet du parcours : [Implémentez un modèle de scoring](https://openclassrooms.com/fr/paths/164/projects/632/assignment).
Ce repos est fait our la partie de modelisation. Il y a un autre pour la partie de l'API et du dashboard.

## 2. Description du Projet
Développer un algorithme de classification pour classifier la demande en crédit accordé ou refusé avec des données déséquilibres. Puis, développer un dashboard interactif pour expliquer de façon la plus transparente possible les décisions d’octroi de crédit.

## 3. Compétences Exigées
- Utiliser un logiciel de version de code pour assurer l’intégration du modèle
- Déployer un modèle via une API dans le Web
- Réaliser un dashboard pour présenter son travail de modélisation
- Rédiger une note méthodologique afin de communiquer sa démarche de modélisation
- Présenter son travail de modélisation à l'oral

## 4. Contenu du Repository
- helper.py : comporte tous les fonctions d'outils pour le nettoyage et analyses de données.
- notebooks/7_01_exploration_données : comporte les nettoyages de données, les visualistions des données mergées ainsi que l'analyse de résultats.
- notebooks/P7_02_modèles : comporte les préparations et les nettoyages de données avant la modelisation, l'analysis du modèle de référence et tous les modèles entraînés.
- notebooks/P7_03_local_features_importance : comporte les visualistions des features importances locales et d'autres graphes pour le dashboard.
- configs : comporte ous les fichiers configuration pour l'entraînement.
- train.py : entraîner les modèles
- optimizer.py : optimiser les hyperparametres des modèles.
- metrics.py : calculer les métriques d'évaluations.

## 5. Données
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)


## 6. Retours du Jury de Soutenance
**Livrable**<br><br>
<ins>Points forts</ins> :
- Tous les attendus sont déposés, complets, et répondent aux attentes du projet
- Le dashbord est réalisé, et un parcours utilisateur simple permettant de répondre aux besoins des utilisateurs, les graphiques réalisés respectent les règles de lisibilités et de clarté 
- L'API est déployée sur le web via un outil gratuit et disponible plusieurs mois 
- L’étudiant fournit des explications claires, avec assurance.
<br><br>

**Soutenance**<br><br>
<ins>Remarques</ins> :
<br>
Bonne présentation, et explication des étapes et démarches. 
