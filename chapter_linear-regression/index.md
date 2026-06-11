# Réseaux de neurones linéaires pour la régression
:label:`chap_regression`

Avant de nous soucier de rendre nos réseaux de neurones profonds,
il sera utile d'en implémenter quelques-uns peu profonds,
pour lesquels les entrées se connectent directement aux sorties.
Cela s'avérera important pour plusieurs raisons.
Premièrement, plutôt que de se laisser distraire par des architectures compliquées,
nous pouvons nous concentrer sur les bases de l'entraînement des réseaux de neurones,
y compris la paramétrisation de la couche de sortie, la manipulation des données,
la spécification d'une fonction de perte et l'entraînement du modèle.
Deuxièmement, cette classe de réseaux peu profonds se trouve être composée de l'ensemble des modèles linéaires,
ce qui englobe de nombreuses méthodes classiques de prédiction statistique,
notamment la régression linéaire et la régression softmax.
Comprendre ces outils classiques est essentiel car ils sont largement utilisés dans de nombreux contextes
et nous devrons souvent les utiliser comme références (baselines) pour justifier l'utilisation d'architectures plus sophistiquées.
Ce chapitre se concentrera précisément sur la régression linéaire
et le suivant étendra notre répertoire de modélisation en développant des réseaux de neurones linéaires pour la classification.

```toc
:maxdepth: 2

linear-regression
oo-design
synthetic-regression-data
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```
