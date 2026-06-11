# Généralisation en apprentissage profond


Dans les :numref:`chap_regression` et :numref:`chap_classification`,
nous avons abordé des problèmes de régression et de classification
en ajustant des modèles linéaires aux données d'entraînement.
Dans les deux cas, nous avons fourni des algorithmes pratiques
pour trouver les paramètres qui maximisaient
la vraisemblance des étiquettes d'entraînement observées.
Ensuite, vers la fin de chaque chapitre,
nous avons rappelé que l'ajustement aux données d'entraînement
n'était qu'un objectif intermédiaire.
Notre véritable quête a toujours été de découvrir des *schémas généraux*
sur la base desquels nous pouvons faire des prédictions précises
même sur de nouveaux exemples issus de la même population sous-jacente.
Les chercheurs en apprentissage automatique sont des *consommateurs* d'algorithmes d'optimisation.
Parfois, nous devons même développer de nouveaux algorithmes d'optimisation.
Mais en fin de compte, l'optimisation n'est qu'un moyen pour parvenir à une fin.
À la base, l'apprentissage automatique est une discipline statistique
et nous souhaitons optimiser la perte d'entraînement uniquement dans la mesure
où un principe statistique (connu ou inconnu)
amène les modèles résultants à généraliser au-delà de l'ensemble d'entraînement.


Du côté positif, il s'avère que les réseaux de neurones profonds
entraînés par descente de gradient stochastique généralisent remarquablement bien
à travers une myriade de problèmes de prédiction, couvrant la vision par ordinateur ;
le traitement du langage naturel ; les données de séries temporelles ; les systèmes de recommandation ;
les dossiers de santé électroniques ; le repliement des protéines ;
l'approximation de fonctions de valeur dans les jeux vidéo
et les jeux de société ; et de nombreux autres domaines.
Du côté négatif, si vous cherchiez
un compte rendu simple
soit de l'histoire de l'optimisation
(pourquoi nous pouvons les ajuster aux données d'entraînement)
soit de l'histoire de la généralisation
(pourquoi les modèles résultants généralisent à des exemples non vus),
alors vous pourriez avoir envie de vous servir un verre.
Alors que nos procédures d'optimisation des modèles linéaires
et les propriétés statistiques des solutions
sont toutes deux bien décrites par un ensemble complet de théories,
notre compréhension de l'apprentissage profond
ressemble encore au Far West sur les deux fronts.

La théorie et la pratique de l'apprentissage profond
évoluent rapidement,
les théoriciens adoptant de nouvelles stratégies
pour expliquer ce qui se passe,
même si les praticiens continuent
d'innover à un rythme effréné,
en construisant des arsenaux d'heuristiques pour l'entraînement des réseaux profonds
et un ensemble d'intuitions et de connaissances populaires
qui fournissent des conseils pour décider
quelles techniques appliquer dans quelles situations.

Le résumé du moment présent est que la théorie de l'apprentissage profond
a produit des pistes d'attaque prometteuses et des résultats fascinants épars,
mais semble encore loin d'un compte rendu complet
de (i) pourquoi nous sommes capables d'optimiser les réseaux de neurones
et (ii) comment les modèles appris par descente de gradient
parviennent à si bien généraliser, même sur des tâches à haute dimension.
Cependant, en pratique, (i) est rarement un problème
(nous pouvons toujours trouver des paramètres qui s'ajusteront à toutes nos données d'entraînement)
et ainsi, comprendre la généralisation est de loin le plus gros problème.
D'un autre côté, même en l'absence du confort d'une théorie scientifique cohérente,
les praticiens ont développé une large collection de techniques
qui peuvent vous aider à produire des modèles qui généralisent bien en pratique.
Bien qu'aucun résumé succinct ne puisse rendre justice
au vaste sujet de la généralisation en apprentissage profond,
et alors que l'état général de la recherche est loin d'être résolu,
nous espérons, dans cette section, présenter un large aperçu
de l'état de la recherche et de la pratique.


## Retour sur le surapprentissage et la régularisation

Selon le théorème du « pas de repas gratuit » (no free lunch) de :citet:`wolpert1995no`,
tout algorithme d'apprentissage généralise mieux sur des données avec certaines distributions, et moins bien avec d'autres distributions.
Ainsi, étant donné un ensemble d'entraînement fini,
un modèle repose sur certaines hypothèses : 
pour atteindre des performances de niveau humain,
il peut être utile d'identifier des *biais inductifs* 
qui reflètent la façon dont les humains pensent le monde.
De tels biais inductifs montrent des préférences 
pour des solutions ayant certaines propriétés.
Par exemple,
un MLP profond a un biais inductif
vers la construction d'une fonction compliquée par la composition de fonctions plus simples.

Avec des modèles d'apprentissage automatique encodant des biais inductifs,
notre approche de leur entraînement
consiste généralement en deux phases : (i) ajuster les données d'entraînement ;
et (ii) estimer l' *erreur de généralisation*
(l'erreur réelle sur la population sous-jacente)
en évaluant le modèle sur des données de réserve (holdout).
La différence entre notre ajustement sur les données d'entraînement
et notre ajustement sur les données de test est appelée l' *écart de généralisation* et quand celui-ci est important,
nous disons que nos modèles font du *surapprentissage* (overfit) sur les données d'entraînement.
Dans des cas extrêmes de surapprentissage,
nous pourrions ajuster exactement les données d'entraînement,
même lorsque l'erreur de test reste significative.
Et dans la vision classique,
l'interprétation est que nos modèles sont trop complexes,
nécessitant que nous réduisions soit le nombre de caractéristiques,
le nombre de paramètres non nuls appris,
soit la taille des paramètres telle que quantifiée.
Rappelez-vous le graphique de la complexité du modèle par rapport à la perte
(:numref:`fig_capacity_vs_error`)
de la :numref:`sec_generalization_basics`.


Cependant, l'apprentissage profond complique ce tableau de manière contre-intuitive.
Premièrement, pour les problèmes de classification,
nos modèles sont généralement assez expressifs
pour s'ajuster parfaitement à chaque exemple d'entraînement,
même dans des ensembles de données comprenant des millions d'exemples
:cite:`zhang2021understanding`.
Dans le schéma classique, nous pourrions penser
que ce réglage se situe à l'extrême droite
de l'axe de complexité du modèle,
et que toute amélioration de l'erreur de généralisation
doit passer par la régularisation,
soit en réduisant la complexité de la classe de modèles,
soit en appliquant une pénalité, contraignant sévèrement
l'ensemble des valeurs que nos paramètres pourraient prendre.
Mais c'est là que les choses commencent à devenir étranges.

Étrangement, pour de nombreuses tâches d'apprentissage profond
(par exemple, la reconnaissance d'images et la classification de textes),
nous choisissons généralement parmi des architectures de modèles
qui peuvent toutes atteindre une perte d'entraînement arbitrairement basse
(et une erreur d'entraînement nulle).
Parce que tous les modèles considérés atteignent une erreur d'entraînement nulle,
*la seule voie pour de nouveaux gains est de réduire le surapprentissage*.
Plus étrange encore, il arrive souvent que
malgré un ajustement parfait des données d'entraînement,
nous puissions en fait *réduire l'erreur de généralisation*
davantage en rendant le modèle *encore plus expressif*,
par exemple en ajoutant des couches, des nœuds, ou en s'entraînant
pendant un plus grand nombre d'époques.
Plus étrange encore, le schéma reliant l'écart de généralisation
à la *complexité* du modèle (telle que capturée, par exemple, par la profondeur ou la largeur des réseaux)
peut être non-monotone,
une plus grande complexité nuisant au début
mais aidant par la suite dans un schéma dit de « double descente » (double-descent)
:cite:`nakkiran2021deep`.
Ainsi, le praticien de l'apprentissage profond possède un sac d'astuces,
dont certaines semblent restreindre le modèle d'une certaine manière
et d'autres qui semblent le rendre encore plus expressif,
et toutes sont, d'une certaine manière, appliquées pour atténuer le surapprentissage.

Compliquant encore plus les choses,
alors que les garanties fournies par la théorie classique de l'apprentissage
peuvent être conservatrices même pour les modèles classiques,
elles semblent impuissantes à expliquer pourquoi
les réseaux de neurones profonds généralisent en premier lieu.
Parce que les réseaux de neurones profonds sont capables de s'ajuster
à des étiquettes arbitraires même pour de grands ensembles de données,
et malgré l'utilisation de méthodes familières telles que la régularisation $\ell_2$,
les bornes de généralisation traditionnelles basées sur la complexité,
par exemple celles basées sur la dimension VC
ou la complexité de Rademacher d'une classe d'hypothèses,
ne peuvent expliquer pourquoi les réseaux de neurones généralisent.

## S'inspirer du non-paramétrique

En abordant l'apprentissage profond pour la première fois,
il est tentant de les considérer comme des modèles paramétriques.
Après tout, les modèles *ont* des millions de paramètres.
Lorsque nous mettons à jour les modèles, nous mettons à jour leurs paramètres.
Lorsque nous sauvegardons les modèles, nous écrivons leurs paramètres sur le disque.
Cependant, les mathématiques et l'informatique sont truffées
de changements de perspective contre-intuitifs,
et d'isomorphismes surprenants entre des problèmes apparemment différents.
Alors que les réseaux de neurones *ont* clairement des paramètres,
il peut être, à certains égards, plus fructueux
de les considérer comme se comportant comme des modèles non-paramétriques.
Alors, qu'est-ce qui rend précisément un modèle non-paramétrique ?
Bien que le nom couvre un ensemble diversifié d'approches,
un thème commun est que les méthodes non-paramétriques
tendent à avoir un niveau de complexité qui croît
à mesure que la quantité de données disponibles augmente.

L'exemple le plus simple de modèle non-paramétrique est peut-être
l'algorithme des $k$ plus proches voisins (nous couvrirons plus de modèles non-paramétriques plus tard, par exemple dans la :numref:`sec_attention-pooling`).
Ici, au moment de l'entraînement,
l'apprenant mémorise simplement l'ensemble de données.
Ensuite, au moment de la prédiction,
confronté à un nouveau point $\mathbf{x}$,
l'apprenant recherche les $k$ plus proches voisins
(les $k$ points $\mathbf{x}_i'$ qui minimisent
une certaine distance $d(\mathbf{x}, \mathbf{x}_i')$).
Lorsque $k=1$, cet algorithme est appelé 1 plus proche voisin ($1$-nearest neighbors),
et l'algorithme atteindra toujours une erreur d'entraînement de zéro.
Cela ne signifie pas pour autant que l'algorithme ne généralisera pas.
En fait, il s'avère que sous certaines conditions légères,
l'algorithme du 1 plus proche voisin est consistant
(convergeant finalement vers le prédicteur optimal).


Notez que le 1 plus proche voisin nécessite que nous spécifiions
une certaine fonction de distance $d$, ou de manière équivalente,
que nous spécifiions une certaine fonction de base à valeur vectorielle $\phi(\mathbf{x})$
pour vectoriser nos données.
Pour tout choix de la métrique de distance,
nous atteindrons une erreur d'entraînement nulle
et finirons par atteindre un prédicteur optimal,
mais différentes métriques de distance $d$
encodent différents biais inductifs
et, avec une quantité finie de données disponibles,
produiront différents prédicteurs.
Différents choix de la métrique de distance $d$
représentent différentes hypothèses sur les schémas sous-jacents
et la performance des différents prédicteurs
dépendra de la compatibilité des hypothèses
avec les données observées.

En un sens, parce que les réseaux de neurones sont sur-paramétrés,
possédant beaucoup plus de paramètres que nécessaire pour s'ajuster aux données d'entraînement,
ils ont tendance à *interpoler* les données d'entraînement (en s'y ajustant parfaitement)
et se comportent ainsi, à certains égards, plus comme des modèles non-paramétriques.
Des recherches théoriques plus récentes ont établi
un lien profond entre les grands réseaux de neurones
et les méthodes non-paramétriques, notamment les méthodes à noyaux.
En particulier, :citet:`Jacot.Grabriel.Hongler.2018`
ont démontré qu'à la limite, alors que les perceptrons multicouches
avec des poids initialisés de manière aléatoire deviennent infiniment larges,
ils deviennent équivalents aux méthodes à noyaux (non-paramétriques)
pour un choix spécifique de la fonction de noyau
(essentiellement, une fonction de distance),
qu'ils appellent le noyau tangent neuronal (neural tangent kernel).
Bien que les modèles actuels de noyau tangent neuronal ne puissent pas expliquer pleinement
le comportement des réseaux profonds modernes,
leur succès en tant qu'outil analytique
souligne l'utilité de la modélisation non-paramétrique
pour comprendre le comportement des réseaux profonds sur-paramétrés.


## Arrêt précoce (Early Stopping)

Bien que les réseaux de neurones profonds soient capables de s'ajuster à des étiquettes arbitraires,
même lorsque les étiquettes sont attribuées de manière incorrecte ou aléatoire
:cite:`zhang2021understanding`,
cette capacité n'émerge qu'après de nombreuses itérations d'entraînement.
Une nouvelle ligne de travail :cite:`Rolnick.Veit.Belongie.Shavit.2017`
a révélé que dans le cadre du bruit des étiquettes,
les réseaux de neurones ont tendance à s'ajuster d'abord aux données proprement étiquetées
et seulement par la suite à interpoler les données mal étiquetées.
De plus, il a été établi que ce phénomène
se traduit directement par une garantie sur la généralisation :
chaque fois qu'un modèle s'est ajusté aux données proprement étiquetées
mais pas aux exemples étiquetés de manière aléatoire inclus dans l'ensemble d'entraînement,
il a en fait généralisé :cite:`Garg.Balakrishnan.Kolter.Lipton.2021`.

Ensemble, ces résultats aident à motiver l' *arrêt précoce* (early stopping),
une technique classique pour régulariser les réseaux de neurones profonds.
Ici, plutôt que de contraindre directement les valeurs des poids,
on contraint le nombre d'époques d'entraînement.
La manière la plus courante de déterminer le critère d'arrêt
est de surveiller l'erreur de validation tout au long de l'entraînement
(généralement en vérifiant une fois après chaque époque)
et d'interrompre l'entraînement lorsque l'erreur de validation
n'a pas diminué de plus d'une certaine petite quantité $\epsilon$
pendant un certain nombre d'époques.
C'est ce qu'on appelle parfois un *critère de patience*.
En plus du potentiel de mener à une meilleure généralisation
dans le cadre d'étiquettes bruitées,
un autre avantage de l'arrêt précoce est le gain de temps.
Une fois le critère de patience satisfait, on peut terminer l'entraînement.
Pour les grands modèles qui peuvent nécessiter des jours d'entraînement
simultanément sur huit GPU ou plus,
un arrêt précoce bien ajusté peut faire gagner des jours aux chercheurs
et peut faire économiser à leurs employeurs plusieurs milliers de dollars.

Notamment, lorsqu'il n'y a pas de bruit d'étiquette et que les ensembles de données sont *réalisables*
(les classes sont véritablement séparables, par exemple en distinguant les chats des chiens),
l'arrêt précoce a tendance à ne pas conduire à des améliorations significatives de la généralisation.
D'un autre côté, lorsqu'il y a du bruit d'étiquette,
ou une variabilité intrinsèque de l'étiquette
(par exemple, prédire la mortalité parmi les patients),
l'arrêt précoce est crucial.
Entraîner des modèles jusqu'à ce qu'ils interpolent des données bruitées est généralement une mauvaise idée.


## Méthodes de régularisation classiques pour les réseaux profonds

Dans le :numref:`chap_regression`, nous avons décrit
plusieurs techniques de régularisation classiques
pour contraindre la complexité de nos modèles.
En particulier, la :numref:`sec_weight_decay`
a introduit une méthode appelée décomposition du poids (weight decay),
qui consiste à ajouter un terme de régularisation à la fonction de perte
afin de pénaliser les grandes valeurs des poids.
Selon la norme de poids qui est pénalisée,
cette technique est connue soit sous le nom de régularisation ridge (pour une pénalité $\ell_2$)
soit sous le nom de régularisation lasso (pour une pénalité $\ell_1$).
Dans l'analyse classique de ces régulariseurs,
ils sont considérés comme suffisamment restrictifs sur les valeurs
que les poids peuvent prendre pour empêcher le modèle de s'ajuster à des étiquettes arbitraires.

Dans les implémentations d'apprentissage profond,
la décomposition du poids reste un outil populaire.
Cependant, les chercheurs ont noté
que les forces typiques de régularisation $\ell_2$
sont insuffisantes pour empêcher les réseaux
d'interpoler les données :cite:`zhang2021understanding` et ainsi les avantages, s'ils sont interprétés
comme une régularisation, ne pourraient avoir de sens
qu'en combinaison avec le critère d'arrêt précoce.
En l'absence d'arrêt précoce, il est possible
que tout comme le nombre de couches
ou le nombre de nœuds (en apprentissage profond)
ou la métrique de distance (en 1 plus proche voisin),
ces méthodes puissent conduire à une meilleure généralisation
non pas parce qu'elles contraignent de manière significative
la puissance du réseau de neurones
mais plutôt parce qu'elles encodent d'une manière ou d'une autre des biais inductifs
qui sont mieux compatibles avec les schémas
trouvés dans les ensembles de données d'intérêt.
Ainsi, les régulariseurs classiques restent populaires
dans les implémentations d'apprentissage profond,
même si la justification théorique
de leur efficacité peut être radicalement différente.

Notamment, les chercheurs en apprentissage profond se sont également appuyés
sur des techniques d'abord popularisées
dans des contextes de régularisation classiques,
comme l'ajout de bruit aux entrées du modèle.
Dans la section suivante, nous introduirons
la célèbre technique du dropout
(inventée par :citet:`Srivastava.Hinton.Krizhevsky.ea.2014`),
qui est devenue un pilier de l'apprentissage profond,
même si la base théorique de son efficacité
reste tout aussi mystérieuse.


## Résumé

Contrairement aux modèles linéaires classiques,
qui ont tendance à avoir moins de paramètres que d'exemples,
les réseaux profonds ont tendance à être sur-paramétrés,
et pour la plupart des tâches sont capables
de s'ajuster parfaitement à l'ensemble d'entraînement.
Ce *régime d'interpolation* remet en question
de nombreuses intuitions fermement établies.
Fonctionnellement, les réseaux de neurones ressemblent à des modèles paramétriques.
Mais les considérer comme des modèles non-paramétriques
peut parfois être une source d'intuition plus fiable.
Parce qu'il arrive souvent que tous les réseaux profonds considérés
soient capables de s'ajuster à toutes les étiquettes d'entraînement,
presque tous les gains doivent provenir de l'atténuation du surapprentissage
(en réduisant l' *écart de généralisation*).
Paradoxalement, les interventions
qui réduisent l'écart de généralisation
semblent parfois augmenter la complexité du modèle
et d'autres fois semblent diminuer la complexité.
Cependant, ces méthodes diminuent rarement suffisamment la complexité
pour que la théorie classique
explique la généralisation des réseaux profonds,
et *pourquoi certains choix conduisent à une meilleure généralisation*
reste pour l'essentiel une question ouverte massive
malgré les efforts concertés de nombreux chercheurs brillants.


## Exercices

1. Dans quel sens les mesures traditionnelles basées sur la complexité ne parviennent-elles pas à rendre compte de la généralisation des réseaux de neurones profonds ?
1. Pourquoi l' *arrêt précoce* pourrait-il être considéré comme une technique de régularisation ?
1. Comment les chercheurs déterminent-ils généralement le critère d'arrêt ?
1. Quel facteur important semble différencier les cas où l'arrêt précoce conduit à de grandes améliorations de la généralisation ?
1. Au-delà de la généralisation, décrivez un autre avantage de l'arrêt précoce.

[Discussions](https://discuss.d2l.ai/t/7473)
