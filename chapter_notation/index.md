# Notation
:label:`chap_notation`

Tout au long de ce livre, nous adhérons 
aux conventions de notation suivantes.
Notez que certains de ces symboles sont des espaces réservés,
tandis que d'autres font référence à des objets spécifiques.
En règle générale, 
l'article indéfini « un » indique souvent
que le symbole est un espace réservé
et que des symboles formatés de manière similaire
peuvent désigner d'autres objets du même type.
Par exemple, « $x$ : un scalaire » signifie 
que les lettres minuscules représentent généralement
des valeurs scalaires,
mais « $\mathbb{Z}$ : l'ensemble des entiers »
fait spécifiquement référence au symbole $\mathbb{Z}$.



## Objets numériques

* $x$ : un scalaire
* $\mathbf{x}$ : un vecteur
* $\mathbf{X}$ : une matrice
* $\mathsf{X}$ : un tenseur général
* $\mathbf{I}$ : la matrice identité (d'une dimension donnée), c'est-à-dire une matrice carrée avec des $1$ sur toutes les entrées de la diagonale et des $0$ partout ailleurs
* $x_i$, $[\mathbf{x}]_i$ : le $i$-ème élément du vecteur $\mathbf{x}$
* $x_{ij}$, $x_{i,j}$, $[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$ : l'élément de la matrice $\mathbf{X}$ à la ligne $i$ et à la colonne $j$.



## Théorie des ensembles


* $\mathcal{X}$ : un ensemble
* $\mathbb{Z}$ : l'ensemble des entiers
* $\mathbb{Z}^+$ : l'ensemble des entiers positifs
* $\mathbb{R}$ : l'ensemble des nombres réels
* $\mathbb{R}^n$ : l'ensemble des vecteurs de nombres réels de dimension $n$
* $\mathbb{R}^{a\times b}$ : l'ensemble des matrices de nombres réels avec $a$ lignes et $b$ colonnes
* $|\mathcal{X}|$ : cardinalité (nombre d'éléments) de l'ensemble $\mathcal{X}$
* $\mathcal{A}\cup\mathcal{B}$ : union des ensembles $\mathcal{A}$ et $\mathcal{B}$
* $\mathcal{A}\cap\mathcal{B}$ : intersection des ensembles $\mathcal{A}$ et $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$ : soustraction d'ensembles de $\mathcal{B}$ de $\mathcal{A}$ (contient uniquement les éléments de $\mathcal{A}$ qui n'appartiennent pas à $\mathcal{B}$)



## Fonctions et opérateurs


* $f(\cdot)$ : une fonction
* $\log(\cdot)$ : le logarithme naturel (base $e$)
* $\log_2(\cdot)$ : logarithme en base $2$
* $\exp(\cdot)$ : la fonction exponentielle
* $\mathbf{1}(\cdot)$ : la fonction indicatrice ; vaut $1$ si l'argument booléen est vrai, et $0$ sinon
* $\mathbf{1}_{\mathcal{X}}(z)$ : la fonction indicatrice d'appartenance à un ensemble ; vaut $1$ si l'élément $z$ appartient à l'ensemble $\mathcal{X}$ et $0$ sinon
* $\mathbf{(\cdot)}^\top$ : transposée d'un vecteur ou d'une matrice
* $\mathbf{X}^{-1}$ : inverse de la matrice $\mathbf{X}$
* $\odot$ : produit de Hadamard (élément par élément)
* $[\cdot, \cdot]$ : concaténation
* $\|\cdot\|_p$ : norme $\ell_p$
* $\|\cdot\|$ : norme $\ell_2$
* $\langle \mathbf{x}, \mathbf{y} \rangle$ : produit scalaire des vecteurs $\mathbf{x}$ et $\mathbf{y}$
* $\sum$ : sommation sur une collection d'éléments
* $\prod$ : produit sur une collection d'éléments
* $\stackrel{\textrm{def}}{=}$ : une égalité affirmée comme une définition du symbole sur le côté gauche



## Analyse

* $\frac{dy}{dx}$ : dérivée de $y$ par rapport à $x$
* $\frac{\partial y}{\partial x}$ : dérivée partielle de $y$ par rapport à $x$
* $\nabla_{\mathbf{x}} y$ : gradient de $y$ par rapport à $\mathbf{x}$
* $\int_a^b f(x) \;dx$ : intégrale définie de $f$ de $a$ à $b$ par rapport à $x$
* $\int f(x) \;dx$ : intégrale indéfinie de $f$ par rapport à $x$



## Probabilités et théorie de l'information

* $X$ : une variable aléatoire
* $P$ : une distribution de probabilité
* $X \sim P$ : la variable aléatoire $X$ suit la distribution $P$
* $P(X=x)$ : la probabilité assignée à l'événement où la variable aléatoire $X$ prend la valeur $x$
* $P(X \mid Y)$ : la distribution de probabilité conditionnelle de $X$ sachant $Y$
* $p(\cdot)$ : une fonction de densité de probabilité (PDF) associée à la distribution $P$
* ${E}[X]$ : espérance d'une variable aléatoire $X$
* $X \perp Y$ : les variables aléatoires $X$ et $Y$ sont indépendantes
* $X \perp Y \mid Z$ : les variables aléatoires $X$ et $Y$ sont conditionnellement indépendantes sachant $Z$
* $\sigma_X$ : écart-type de la variable aléatoire $X$
* $\textrm{Var}(X)$ : variance de la variable aléatoire $X$, égale à $\sigma^2_X$
* $\textrm{Cov}(X, Y)$ : covariance des variables aléatoires $X$ et $Y$
* $\rho(X, Y)$ : le coefficient de corrélation de Pearson entre $X$ and $Y$, égal à $\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $H(X)$ : entropie de la variable aléatoire $X$
* $D_{\textrm{KL}}(P\|Q)$ : la divergence de Kullback-Leibler (KL) (ou entropie relative) de la distribution $Q$ vers la distribution $P$



[Discussions](https://discuss.d2l.ai/t/25)
