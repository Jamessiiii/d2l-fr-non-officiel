# Généralisation
:label:`sec_generalization_basics`

Considérez deux étudiants universitaires se préparant
assidûment pour leur examen final.
Couramment, cette préparation consistera
à s'entraîner et à tester leurs capacités
en passant des examens administrés les années précédentes.
Néanmoins, réussir les examens passés ne garantit pas
qu'ils excelleront le moment venu.
Par exemple, imaginez une étudiante, Ellie l'Extraordinaire,
dont la préparation consistait entièrement
à mémoriser les réponses
aux questions des examens des années précédentes.
Même si Ellie était dotée
d'une mémoire extraordinaire,
et pouvait ainsi se rappeler parfaitement la réponse
à toute question *déjà vue*,
elle pourrait néanmoins se figer
face à une nouvelle question (*jamais vue auparavant*).
En comparaison, imaginez une autre étudiante,
Irene l'Inductive, avec des capacités de mémorisation
comparativement médiocres,
mais un don pour déceler des motifs.
Notez que si l'examen consistait réellement en
des questions recyclées d'une année précédente,
Ellie surpasserait Irene haut la main.
Même si les motifs inférés par Irene
produisaient des prédictions précises à 90 %,
ils ne pourraient jamais rivaliser avec
le rappel à 100 % d'Ellie.
Cependant, même si l'examen consistait
entièrement en de nouvelles questions,
Irene pourrait maintenir sa moyenne de 90 %.

En tant que scientifiques de l'apprentissage automatique,
notre objectif est de découvrir des *motifs*.
Mais comment pouvons-nous être sûrs que nous avons
véritablement découvert un motif *général*
et non simplement mémorisé nos données ?
La plupart du temps, nos prédictions ne sont utiles
que si notre modèle découvre un tel motif.
Nous ne voulons pas prédire les cours de la bourse d'hier, mais ceux de demain.
Nous n'avons pas besoin de reconnaître
des maladies déjà diagnostiquées
pour des patients déjà vus,
mais plutôt des affections non diagnostiquées auparavant
chez des patients jamais vus.
Ce problème --- comment découvrir des motifs qui se *généralisent* --- est
le problème fondamental de l'apprentissage automatique,
et sans doute de toutes les statistiques.
Nous pourrions considérer ce problème comme n'étant qu'une partie
 d'une question bien plus vaste
qui englobe toute la science :
quand sommes-nous jamais justifiés
à faire le saut d'observations particulières
à des énoncés plus généraux ?


Dans la vie réelle, nous devons ajuster nos modèles
en utilisant une collection finie de données.
Les échelles typiques de ces données
varient énormément selon les domaines.
Pour de nombreux problèmes médicaux importants,
nous ne pouvons accéder qu'à quelques milliers de points de données.
Lors de l'étude de maladies rares,
nous pourrions avoir de la chance d'en accéder à des centaines.
En revanche, les plus grands jeux de données publics
composés de photographies étiquetées,
par exemple ImageNet :cite:`Deng.Dong.Socher.ea.2009`,
contiennent des millions d'images.
Et certaines collections d'images non étiquetées
telles que le jeu de données Flickr YFC100M
peuvent être encore plus vastes, contenant
plus de 100 millions d'images :cite:`thomee2016yfcc100m`.
Cependant, même à cette échelle extrême,
le nombre de points de données disponibles
reste infinitésimal
par rapport à l'espace de toutes les images possibles
à une résolution d'un mégapixel.
Chaque fois que nous travaillons avec des échantillons finis,
nous devons garder à l'esprit le risque
que nous puissions ajuster nos données d'entraînement,
pour découvrir finalement que nous avons échoué
à découvrir un motif généralisable.

Le phénomène d'ajustement plus proche de nos données d'entraînement
que de la distribution sous-jacente est appelé *surapprentissage* (overfitting),
et les techniques pour lutter contre le surapprentissage
sont souvent appelées méthodes de *régularisation*.
Bien que cela ne remplace pas une véritable introduction
à la théorie de l'apprentissage statistique (voir :citet:`Vapnik98,boucheron2005theory`),
nous vous donnerons juste assez d'intuition pour commencer.
Nous reviendrons sur la généralisation dans de nombreux chapitres
tout au long du livre,
en explorant à la fois ce qui est connu sur
les principes sous-jacents à la généralisation
dans divers modèles,
ainsi que les techniques heuristiques
qui ont été trouvées (empiriquement)
pour produire une meilleure généralisation
sur des tâches d'intérêt pratique.



## Erreur d'entraînement et erreur de généralisation


Dans le cadre standard de l'apprentissage supervisé,
nous supposons que les données d'entraînement et les données de test
sont tirées de manière *indépendante* à partir de distributions *identiques*.
C'est ce qu'on appelle communément l' *hypothèse IID*.
Bien que cette hypothèse soit forte,
il convient de noter que, sans une telle hypothèse,
nous serions dans l'impasse.
Pourquoi devrions-nous croire que des données d'entraînement
échantillonnées à partir de la distribution $P(X,Y)$
devraient nous dire comment faire des prédictions sur
des données de test générées par une *distribution différente* $Q(X,Y)$ ?
Faire de tels sauts s'avère nécessiter
des hypothèses fortes sur la manière dont $P$ et $Q$ sont liées.
Plus tard, nous discuterons de certaines hypothèses
qui permettent des décalages de distribution,
mais nous devons d'abord comprendre le cas IID,
où $P(\cdot) = Q(\cdot)$.

Pour commencer, nous devons différencier l' *erreur d'entraînement* $R_\textrm{emp}$,
qui est une *statistique*
calculée sur le jeu de données d'entraînement,
de l' *erreur de généralisation* $R$,
qui est une *espérance* prise
par rapport à la distribution sous-jacente.
Vous pouvez considérer l'erreur de généralisation comme
ce que vous verriez si vous appliquiez votre modèle
à un flux infini d'exemples de données supplémentaires
tirés de la même distribution de données sous-jacente.
Formellement, l'erreur d'entraînement est exprimée sous forme d'une *somme* (avec la même notation que dans la :numref:`sec_linear_regression`) :

$$R_\textrm{emp}[\mathbf{X}, \mathbf{y}, f] = \frac{1}{n} \sum_{i=1}^n l(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)})),$$


tandis que l'erreur de généralisation est exprimée sous forme d'une intégrale :

$$R[p, f] = E_{(\mathbf{x}, y) \sim P} [l(\mathbf{x}, y, f(\mathbf{x}))] =
\int \int l(\mathbf{x}, y, f(\mathbf{x})) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Problématiquement, we can never calculate
exactement l'erreur de généralisation $R$.
Personne ne nous donne jamais la forme précise
de la fonction de densité $p(\mathbf{x}, y)$.
De plus, nous ne pouvons pas échantillonner un flux infini de points de données.
Ainsi, en pratique, nous devons *estimer* l'erreur de généralisation
en appliquant notre modèle à un jeu de test indépendant
constitué d'une sélection aléatoire d'exemples
$\mathbf{X}'$ et d'étiquettes $\mathbf{y}'$
qui ont été écartés de notre jeu d'entraînement.
Cela consiste à appliquer la même formule
que celle utilisée pour calculer l'erreur d'entraînement empirique,
mais à un jeu de test $\mathbf{X}', \mathbf{y}'$.


Crucialement, lorsque nous évaluons notre classifieur sur le jeu de test,
nous travaillons avec un classifieur *fixe*
(il ne dépend pas de l'échantillon du jeu de test),
et ainsi l'estimation de son erreur
est simplement le problème de l'estimation de la moyenne.
Cependant, on ne peut pas en dire autant
pour le jeu d'entraînement.
Notez que le modèle avec lequel nous finissons
dépend explicitement de la sélection du jeu d'entraînement,
et ainsi l'erreur d'entraînement sera en général
une estimation biaisée de l'erreur réelle
sur la population sous-jacente.
La question centrale de la généralisation
est alors de savoir quand nous devrions nous attendre à ce que notre erreur d'entraînement
soit proche de l'erreur de la population
(et donc de l'erreur de généralisation).

### Complexité du modèle

Dans la théorie classique, lorsque nous avons
des modèles simples et des données abondantes,
les erreurs d'entraînement et de généralisation ont tendance à être proches.
Cependant, lorsque nous travaillons avec
des modèles plus complexes et/ou moins d'exemples,
nous nous attendons à ce que l'erreur d'entraînement diminue
mais que l'écart de généralisation augmente.
Cela ne devrait pas être surprenant.
Imaginez une classe de modèles si expressive que
pour n'importe quel jeu de données de $n$ exemples,
nous puissions trouver un ensemble de paramètres
capables d'ajuster parfaitement des étiquettes arbitraires,
même assignées de manière aléatoire.
Dans ce cas, même si nous ajustons parfaitement nos données d'entraînement,
comment pouvons-nous conclure quoi que ce soit sur l'erreur de généralisation ?
Pour tout ce que nous en savons, notre erreur de généralisation
pourrait n'être pas meilleure qu'une supposition aléatoire.

En général, en l'absence de toute restriction sur notre classe de modèles,
nous ne pouvons pas conclure, sur la seule base de l'ajustement des données d'entraînement,
que notre modèle a découvert un motif généralisable :cite:`vapnik1994measuring`.
D'un autre côté, si notre classe de modèles
n'était pas capable d'ajuster des étiquettes arbitraires,
alors elle doit avoir découvert un motif.
Les idées de la théorie de l'apprentissage sur la complexité des modèles
se sont inspirées en partie des idées
de Karl Popper, un philosophe des sciences influent,
qui a formalisé le critère de réfutation (falsifiabilité).
Selon Popper, une théorie
qui peut expliquer n'importe quelle observation
n'est pas du tout une théorie scientifique !
Après tout, que nous a-t-elle dit sur le monde
si elle n'a exclu aucune possibilité ?
En résumé, ce que nous voulons, c'est une hypothèse
qui *ne pourrait pas* expliquer n'importe quelle observation
que nous pourrions concevoir
et qui, pourtant, se trouve être compatible
avec les observations que nous faisons *en fait*.

Maintenant, ce qui constitue précisément une notion appropriée
de complexité de modèle est une question complexe.
Souvent, les modèles ayant plus de paramètres
sont capables d'ajuster un plus grand nombre
d'étiquettes assignées arbitrairement.
Cependant, ce n'est pas nécessairement vrai.
Par exemple, les méthodes à noyau opèrent dans des espaces
avec un nombre infini de paramètres,
pourtant leur complexité est contrôlée
par d'autres moyens :cite:`Scholkopf.Smola.2002`.
Une notion de complexité qui s'avère souvent utile
est la plage de valeurs que les paramètres peuvent prendre.
Ici, un modèle dont les paramètres sont autorisés
à prendre des valeurs arbitraires
serait plus complexe.
Nous reviendrons sur cette idée dans la section suivante,
lorsque nous introduirons la *décroissance des poids* (weight decay),
votre première technique pratique de régularisation.
Notamment, il peut être difficile de comparer
la complexité entre les membres de classes de modèles sensiblement différentes
(par exemple, les arbres de décision par rapport aux réseaux de neurones).


À ce stade, nous devons souligner un autre point important
sur lequel nous reviendrons lors de l'introduction des réseaux de neurones profonds.
Lorsqu'un modèle est capable d'ajuster des étiquettes arbitraires,
une faible erreur d'entraînement n'implique pas nécessairement
une faible erreur de généralisation.
*Cependant, elle n'implique pas nécessairement non plus une erreur de généralisation élevée !*
Tout ce que nous pouvons dire avec certitude est qu'une
faible erreur d'entraînement seule ne suffit pas
à certifier une faible erreur de généralisation.
Les réseaux de neurones profonds s'avèrent être justement de tels modèles :
bien qu'ils se généralisent bien en pratique,
ils sont trop puissants pour nous permettre de conclure
grand-chose sur la seule base de l'erreur d'entraînement.
Dans ces cas, nous devons nous appuyer plus lourdement
sur nos données de réserve pour certifier la généralisation
après coup.
L'erreur sur les données de réserve, c'est-à-dire le jeu de validation,
est appelée l' *erreur de validation*.

## Sous-apprentissage ou surapprentissage ?

Lorsque nous comparons les erreurs d'entraînement et de validation,
nous voulons être attentifs à deux situations courantes.
Premièrement, nous voulons surveiller les cas
où notre erreur d'entraînement et notre erreur de validation sont toutes deux substantielles,
mais avec un faible écart entre elles.
Si le modèle est incapable de réduire l'erreur d'entraînement,
cela pourrait signifier que notre modèle est trop simple
(c'est-à-dire insuffisamment expressif)
pour capturer le motif que nous essayons de modéliser.
De plus, puisque l' *écart de généralisation* ($R_\textrm{emp} - R$)
entre nos erreurs d'entraînement et de généralisation est faible,
nous avons des raisons de croire que nous pourrions nous en sortir avec un modèle plus complexe.
Ce phénomène est connu sous le nom de *sous-apprentissage* (underfitting).

D'un autre côté, comme nous l'avons discuté plus haut,
nous voulons surveiller les cas
où notre erreur d'entraînement est nettement inférieure
à notre erreur de validation, ce qui indique un *surapprentissage* (overfitting) sévère.
Notez que le surapprentissage n'est pas toujours une mauvaise chose.
Dans l'apprentissage profond en particulier,
les meilleurs modèles prédictifs sont souvent
bien plus performants sur les données d'entraînement que sur les données de réserve.
En fin de compte, nous nous soucions généralement
de réduire l'erreur de généralisation,
et nous ne nous soucions de l'écart que dans la mesure
où il devient un obstacle à cette fin.
Notez que si l'erreur d'entraînement est nulle,
alors l'écart de généralisation est précisément égal à l'erreur de généralisation
et nous ne pouvons progresser qu'en réduisant l'écart.

### Ajustement de courbe polynomiale
:label:`subsec_polynomial-curve-fitting`

Pour illustrer une certaine intuition classique
sur le surapprentissage et la complexité des modèles,
considérez ce qui suit :
étant donné des données d'entraînement composées d'une seule caractéristique $x$
et d'une étiquette à valeur réelle correspondante $y$,
nous essayons de trouver le polynôme de degré $d$

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

pour estimer l'étiquette $y$.
Il s'agit simplement d'un problème de régression linéaire
où nos caractéristiques sont données par les puissances de $x$,
les poids du modèle sont donnés par $w_i$,
et le biais est donné par $w_0$ puisque $x^0 = 1$ pour tout $x$.
Puisqu'il s'agit simplement d'un problème de régression linéaire,
nous pouvons utiliser l'erreur quadratique comme fonction de perte.


Une fonction polynomiale d'ordre supérieur est plus complexe
qu'une fonction polynomiale d'ordre inférieur,
car le polynôme d'ordre supérieur a plus de paramètres
et la plage de sélection de la fonction du modèle est plus large.
En fixant le jeu de données d'entraînement,
les fonctions polynomiales d'ordre supérieur devraient toujours
atteindre une erreur d'entraînement inférieure (au pire, égale)
par rapport aux polynômes de degré inférieur.
En fait, chaque fois que chaque exemple de données
a une valeur distincte de $x$,
une fonction polynomiale de degré
égal au nombre d'exemples de données
peut ajuster parfaitement le jeu d'entraînement.
Nous comparons la relation entre le degré polynomial (complexité du modèle)
et à la fois le sous-apprentissage et le surapprentissage dans la :numref:`fig_capacity_vs_error`.

![Influence de la complexité du modèle sur le sous-apprentissage et le surapprentissage.](../img/capacity-vs-error.svg)
:label:`fig_capacity_vs_error`


### Taille du jeu de données

Comme l'indique déjà la borne ci-dessus,
une autre considération importante
à garder à l'esprit est la taille du jeu de données.
En fixant notre modèle, moins nous avons d'échantillons
dans le jeu de données d'entraînement,
plus nous sommes susceptibles (et plus sévèrement)
de rencontrer du surapprentissage.
À mesure que nous augmentons la quantité de données d'entraînement,
l'erreur de généralisation diminue généralement.
De plus, en général, plus de données ne font jamais de mal.
Pour une tâche et une distribution de données fixées,
la complexité du modèle ne devrait pas augmenter
plus rapidement que la quantité de données.
Avec plus de données, nous pourrions essayer
d'ajuster un modèle plus complexe.
En l'absence de données suffisantes, les modèles plus simples
peuvent être plus difficiles à battre.
Pour de nombreuses tâches, l'apprentissage profond
ne surpasse les modèles linéaires
que lorsque plusieurs milliers d'exemples d'entraînement sont disponibles.
En partie, le succès actuel de l'apprentissage profond
doit beaucoup à l'abondance de jeux de données massifs
provenant des entreprises Internet, du stockage bon marché,
des appareils connectés et de la numérisation généralisée de l'économie.

## Sélection de modèle
:label:`subsec_generalization-model-selection`

Typiquement, nous sélectionnons notre modèle final
seulement après avoir évalué plusieurs modèles
qui diffèrent de diverses manières
(différentes architectures, objectifs d'entraînement,
caractéristiques sélectionnées, prétraitement des données,
taux d'apprentissage, etc.).
Choisir parmi de nombreux modèles est appelé à juste titre
la *sélection de modèle*.

En principe, we should not touch our test set
avant d'avoir choisi tous nos hyperparamètres.
Si nous devions utiliser les données de test dans le processus de sélection de modèle,
il y aurait un risque que nous fassions du surapprentissage sur les données de test.
Nous serions alors dans de sérieux ennuis.
Si nous faisons du surapprentissage sur nos données d'entraînement,
il y a toujours l'évaluation sur les données de test pour nous garder honêtes.
Mais si nous faisons du surapprentissage sur les données de test, comment le saurions-nous jamais ?
Voir :citet:`ong2005learning` pour un exemple de la manière dont
cela peut conduire à des résultats absurdes même pour des modèles dont la complexité
peut être étroitement contrôlée.

Ainsi, nous ne devrions jamais nous fier aux données de test pour la sélection de modèle.
Et pourtant, nous ne pouvons pas non plus nous fier uniquement aux données d'entraînement
pour la sélection de modèle, car
nous ne pouvons pas estimer l'erreur de généralisation
sur les données mêmes que nous utilisons pour entraîner le modèle.


Dans les applications pratiques, le tableau devient plus trouble.
Bien qu'idéalement nous ne devrions toucher aux données de test qu'une seule fois,
pour évaluer le tout meilleur modèle ou pour comparer
un petit nombre de modèles entre eux,
les données de test du monde réel sont rarement jetées après une seule utilisation.
Nous pouvons rarement nous permettre un nouveau jeu de test pour chaque série d'expériences.
En fait, le recyclage des données de référence (benchmark) pendant des décennies
peut avoir un impact significatif sur le
développement des algorithmes,
par exemple pour la [classification d'images](https://paperswithcode.com/sota/image-classification-on-imagenet)
et la [reconnaissance optique de caractères](https://paperswithcode.com/sota/image-classification-on-mnist).

La pratique courante pour aborder le problème de l' *entraînement sur le jeu de test*
consiste à diviser nos données en trois parties,
en incorporant un *jeu de validation*
en plus des jeux de données d'entraînement et de test.
Le résultat est une affaire trouble où les frontières
entre les données de validation et de test sont d'une ambiguïté inquiétante.
Sauf indication contraire explicite, dans les expériences de ce livre,
nous travaillons réellement avec ce qui devrait être appelé à juste titre
des données d'entraînement et des données de validation, sans véritable jeu de test.
Par conséquent, la précision rapportée dans chaque expérience du livre est réellement
la précision de validation et non une véritable précision sur jeu de test.

### Validation croisée

Lorsque les données d'entraînement sont rares,
nous pourrions ne même pas être en mesure de nous permettre de mettre de côté
suffisamment de données pour constituer un jeu de validation approprié.
Une solution populaire à ce problème consiste à employer
la *validation croisée à $K$ blocs* ($K$-fold cross-validation).
Ici, les données d'entraînement originales sont divisées en $K$ sous-ensembles non chevauchants.
Ensuite, l'entraînement et la validation du modèle sont exécutés $K$ fois,
chaque fois en s'entraînant sur $K-1$ sous-ensembles et en validant
sur un sous-ensemble différent (celui qui n'a pas été utilisé pour l'entraînement lors de ce tour).
Enfin, les erreurs d'entraînement et de validation sont estimées
en faisant la moyenne des résultats des $K$ expériences.



## Résumé

Cette section a exploré certains des fondements
de la généralisation dans l'apprentissage automatique.
Certaines de ces idées deviennent compliquées
et contre-intuitives lorsque nous passons à des modèles plus profonds ; ici, les modèles sont capables de faire un surapprentissage sévère sur les données,
et les notions de complexité pertinentes
peuvent être à la fois implicites et contre-intuitives
(par exemple, des architectures plus grandes avec plus de paramètres
se généralisant mieux).
Nous vous laissons avec quelques règles empiriques :

1. Utilisez des jeux de validation (ou la *validation croisée à $K$ blocs*) pour la sélection de modèle ;
1. Les modèles plus complexes nécessitent souvent plus de données ;
1. Les notions de complexité pertinentes incluent à la fois le nombre de paramètres et la plage de valeurs qu'ils sont autorisés à prendre ;
1. Toutes choses étant égales par ailleurs, plus de données conduisent presque toujours à une meilleure généralisation ;
1. Tout ce discours sur la généralisation est fondé sur l'hypothèse IID. Si nous assouplissons cette hypothèse, en permettant aux distributions de se décaler entre les périodes d'entraînement et de test, alors nous ne pouvons rien dire sur la généralisation sans une hypothèse supplémentaire (peut-être plus légère).


## Exercices

1. Quand pouvez-vous résoudre exactement le problème de la régression polynomiale ?
1. Donnez au moins cinq exemples où des variables aléatoires dépendantes font qu'il est déconseillé de traiter le problème comme des données IID.
1. Pouvez-vous espérer voir un jour une erreur d'entraînement nulle ? Dans quelles circonstances verriez-vous une erreur de généralisation nulle ?
1. Pourquoi la validation croisée à $K$ blocs est-elle très coûteuse à calculer ?
1. Pourquoi l'estimation de l'erreur par validation croisée à $K$ blocs est-elle biaisée ?
1. La dimension VC est définie comme le nombre maximum de points pouvant être classés avec des étiquettes arbitraires $\{\pm 1\}$ par une fonction d'une classe de fonctions. Pourquoi cela pourrait-il ne pas être une bonne idée pour mesurer la complexité de la classe de fonctions ? Indice : considérez l'amplitude des fonctions.
1. Votre responsable vous donne un jeu de données difficile sur lequel votre algorithme actuel ne fonctionne pas très bien. Comment lui justifieriez-vous que vous avez besoin de plus de données ? Indice : vous ne pouvez pas augmenter les données, mais vous pouvez les réduire.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17978)
:end_tab:
