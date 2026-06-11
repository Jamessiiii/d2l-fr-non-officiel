# Perceptrons multicouches
:label:`chap_perceptrons`

Dans ce chapitre, nous allons introduire votre premier réseau véritablement *profond*.
Les réseaux profonds les plus simples sont appelés *perceptrons multicouches*,
et ils consistent en plusieurs couches de neurones,
chacune étant entièrement connectée à celles de la couche inférieure
(dont elles reçoivent les entrées)
et à celles de la couche supérieure (qu'elles influencent à leur tour).
Bien que la différenciation automatique
simplifie considérablement la mise en œuvre des algorithmes d'apprentissage profond,
nous plongerons dans la manière dont ces gradients
sont calculés dans les réseaux profonds.
Ensuite, nous
serons prêts à
discuter des questions relatives à la stabilité numérique et à l'initialisation des paramètres,
qui sont essentielles pour entraîner avec succès des réseaux profonds.
Lorsque nous entraînons de tels modèles à haute capacité, nous courons le risque de surapprentissage. Ainsi, nous
reviendrons sur la régularisation et la généralisation
pour les réseaux profonds.
Tout au long de ce chapitre, nous visons
à vous donner une solide compréhension non seulement des concepts mais aussi de la pratique de l'utilisation des réseaux profonds.
À la fin de ce chapitre, nous appliquerons ce que nous avons introduit jusqu'à présent à un cas réel : la prédiction
des prix des maisons. Nous renvoyons les questions relatives aux performances de calcul, à l'extensibilité et à l'efficacité
de nos modèles aux chapitres suivants.

```toc
:maxdepth: 2

mlp
mlp-implementation
backprop
numerical-stability-and-init
generalization-deep
dropout
kaggle-house-price
```
