```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Algèbre linéaire
:label:`sec_linear-algebra`

À ce stade, nous savons charger des jeux de données dans des tenseurs
et manipuler ces tenseurs
avec des opérations mathématiques de base.
Pour commencer à construire des modèles sophistiqués,
nous aurons également besoin de quelques outils de l'algèbre linéaire.
Cette section propose une introduction douce
aux concepts les plus essentiels,
en partant de l'arithmétique scalaire
jusqu'à la multiplication de matrices.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from jax import numpy as jnp
```

## Scalaires


La plupart des mathématiques quotidiennes
consistent à manipuler
des nombres un par un.
Formellement, nous appelons ces valeurs des *scalaires*.
Par exemple, la température à Palo Alto
est de $72$ degrés Fahrenheit.
Si vous vouliez convertir la température en Celsius,
vous évalueriez l'expression
$c = \frac{5}{9}(f - 32)$, en fixant $f$ à $72$.
Dans cette équation, les valeurs
$5$, $9$ et $32$ sont des scalaires constants.
Les variables $c$ et $f$
représentent en général des scalaires inconnus.

Nous désignons les scalaires
par des lettres minuscules ordinaires
(par exemple, $x$, $y$ et $z$)
et l'espace de tous les scalaires (continus)
*à valeurs réelles* par $\mathbb{R}$.
Par souci de concision, nous ferons l'impasse sur
les définitions rigoureuses d'*espaces* :
retenez simplement que l'expression $x \in \mathbb{R}$
est une manière formelle de dire que $x$ est un scalaire à valeur réelle.
Le symbole $\in$ (prononcé "appartient à")
désigne l'appartenance à un ensemble.
Par exemple, $x, y \in \{0, 1\}$
indique que $x$ et $y$ sont des variables
qui ne peuvent prendre que les valeurs $0$ ou $1$.

(**Les scalaires sont implémentés comme des tenseurs
qui ne contiennent qu'un seul élément.**)
Ci-dessous, nous assignons deux scalaires
et effectuons les opérations familières d'addition, de multiplication,
de division et d'exponentiation.

```{.python .input}
%%tab mxnet
x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab jax
x = jnp.array(3.0)
y = jnp.array(2.0)

x + y, x * y, x / y, x**y
```

## Vecteurs

Pour les besoins actuels, [**vous pouvez considérer un vecteur comme un tableau de scalaires de longueur fixe.**]
Comme pour leurs équivalents en code,
nous appelons ces scalaires les *éléments* du vecteur
(les synonymes incluent *entrées* et *composantes*).
Lorsque les vecteurs représentent des exemples issus de jeux de données réels,
leurs valeurs ont une signification concrète.
Par exemple, si nous entraînions un modèle pour prédire
le risque de défaut de paiement d'un prêt,
nous pourrions associer chaque demandeur à un vecteur
dont les composantes correspondent à des quantités
telles que leur revenu, la durée de leur emploi,
ou le nombre de défauts de paiement précédents.
Si nous étudiions le risque de crise cardiaque,
chaque vecteur pourrait représenter un patient
et ses composantes pourraient correspondre à
ses signes vitaux les plus récents, son taux de cholestérol,
ses minutes d'exercice par jour, etc.
Nous désignons les vecteurs par des lettres minuscules en gras,
(par exemple, $\mathbf{x}$, $\mathbf{y}$ et $\mathbf{z}$).

Les vecteurs sont implémentés comme des tenseurs d'ordre $1$.
En général, de tels tenseurs peuvent avoir des longueurs arbitraires,
sous réserve des limitations de mémoire. Attention : en Python, comme dans la plupart des langages de programmation, les indices de vecteurs commencent à $0$, ce que l'on appelle également l'*indexation à partir de zéro*, alors qu'en algèbre linéaire, les indices commencent à $1$ (indexation à partir de un).

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(3)
x
```

Nous pouvons nous référer à un élément d'un vecteur en utilisant un indice.
Par exemple, $x_2$ désigne le deuxième élément de $\mathbf{x}$.
Comme $x_2$ est un scalaire, nous ne le mettons pas en gras.
Par défaut, nous visualisons les vecteurs
en empilant leurs éléments verticalement :

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$
:eqlabel:`eq_vec_def`

Ici $x_1, \ldots, x_n$ sont les éléments du vecteur.
Plus tard, nous distinguerons ces *vecteurs colonnes*
des *vecteurs lignes* dont les éléments sont empilés horizontalement.
Rappelez-vous que [**nous accédons aux éléments d'un tenseur via l'indexation.**]

```{.python .input}
%%tab all
x[2]
```

Pour indiquer qu'un vecteur contient $n$ éléments,
nous écrivons $\mathbf{x} \in \mathbb{R}^n$.
Formellement, nous appelons $n$ la *dimensionnalité* du vecteur.
[**En code, cela correspond à la longueur du tenseur**],
accessible via la fonction intégrée `len` de Python.

```{.python .input}
%%tab all
len(x)
```

Nous pouvons également accéder à la longueur via l'attribut `shape`.
La forme (*shape*) est un tuple qui indique la longueur d'un tenseur le long de chaque axe.
(**Les tenseurs avec un seul axe ont des formes avec un seul élément.**)

```{.python .input}
%%tab all
x.shape
```

Souvent, le mot "dimension" est surchargé
pour signifier à la fois le nombre d'axes
et la longueur le long d'un axe particulier.
Pour éviter cette confusion,
nous utilisons *ordre* pour désigner le nombre d'axes
et *dimensionnalité* exclusivement pour désigner
le nombre de composantes.


## Matrices

Tout comme les scalaires sont des tenseurs d'ordre $0$
et les vecteurs sont des tenseurs d'ordre $1$,
les matrices sont des tenseurs d'ordre $2$.
Nous désignons les matrices par des lettres majuscules en gras
(par exemple, $\mathbf{X}$, $\mathbf{Y}$ et $\mathbf{Z}$),
et nous les représentons en code par des tenseurs à deux axes.
L'expression $\mathbf{A} \in \mathbb{R}^{m \times n}$
indique qu'une matrice $\mathbf{A}$
contient $m \times n$ scalaires à valeurs réelles,
disposés en $m$ lignes et $n$ colonnes.
Lorsque $m = n$, nous disons qu'une matrice est *carrée*.
Visuellement, nous pouvons illustrer n'importe quelle matrice sous forme de tableau.
Pour désigner un élément individuel,
nous mettons en indice les indices de ligne et de colonne, par exemple,
$a_{ij}$ est la valeur qui appartient à la
$i^{\textrm{ème}}$ ligne et à la $j^{\textrm{ème}}$ colonne de $\mathbf{A}$ :

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


En code, nous représentons une matrice $\mathbf{A} \in \mathbb{R}^{m \times n}$
par un tenseur d'ordre $2$ avec la forme ($m$, $n$).
[**Nous pouvons convertir n'importe quel tenseur de taille appropriée $m \times n$
en une matrice $m \times n$**]
en passant la forme souhaitée à `reshape` :

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

```{.python .input}
%%tab jax
A = jnp.arange(6).reshape(3, 2)
A
```

Parfois, nous voulons inverser les axes.
Lorsque nous échangeons les lignes et les colonnes d'une matrice,
le résultat est appelé sa *transposée*.
Formellement, nous désignons la transposée d'une matrice $\mathbf{A}$
par $\mathbf{A}^\top$ et si $\mathbf{B} = \mathbf{A}^\top$,
alors $b_{ij} = a_{ji}$ pour tous $i$ et $j$.
Ainsi, la transposée d'une matrice $m \times n$
est une matrice $n \times m$ :

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

En code, nous pouvons accéder à la (**transposée de n'importe quelle matrice**) comme suit :

```{.python .input}
%%tab mxnet, pytorch, jax
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**Les matrices symétriques sont le sous-ensemble des matrices carrées
qui sont égales à leurs propres transposées :
$\mathbf{A} = \mathbf{A}^\top$.**]
La matrice suivante est symétrique :

```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

```{.python .input}
%%tab jax
A = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

Les matrices sont utiles pour représenter des jeux de données.
Généralement, les lignes correspondent à des enregistrements individuels
et les colonnes correspondent à des attributs distincts.



## Tenseurs

Bien que vous puissiez aller loin dans votre parcours d'apprentissage automatique
avec seulement des scalaires, des vecteurs et des matrices,
vous pourriez éventuellement avoir besoin de travailler avec des
[**tenseurs**] d'ordre supérieur.
Les tenseurs (**nous offrent une manière générique de décrire
des extensions aux tableaux d'ordre $n$.**)
Nous appelons les objets logiciels de la *classe tenseur* "tenseurs"
précisément parce qu'ils peuvent également avoir un nombre arbitraire d'axes.
Bien qu'il puisse être déroutant d'utiliser le mot
*tenseur* à la fois pour l'objet mathématique
et sa réalisation en code,
notre sens devrait généralement être clair d'après le contexte.
Nous désignons les tenseurs généraux par des lettres majuscules
avec une police de caractères spéciale
(par exemple, $\mathsf{X}$, $\mathsf{Y}$ et $\mathsf{Z}$)
et leur mécanisme d'indexation
(par exemple, $x_{ijk}$ et $[\mathsf{X}]_{1, 2i-1, 3}$)
découle naturellement de celui des matrices.

Les tenseurs deviendront plus importants
lorsque nous commencerons à travailler avec des images.
Chaque image arrive comme un tenseur d'ordre $3$
avec des axes correspondant à la hauteur, à la largeur et au *canal*.
À chaque emplacement spatial, les intensités
de chaque couleur (rouge, vert et bleu)
sont empilées le long du canal.
De plus, une collection d'images est représentée
en code par un tenseur d'ordre $4$,
où les images distinctes sont indexées
le long du premier axe.
Les tenseurs d'ordre supérieur sont construits, comme l'étaient les vecteurs et les matrices,
en augmentant le nombre de composantes de la forme.

```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.arange(24).reshape(2, 3, 4)
```

## Propriétés fondamentales de l'arithmétique des tenseurs

Les scalaires, les vecteurs, les matrices
et les tenseurs d'ordre supérieur
ont tous des propriétés pratiques.
Par exemple, les opérations élément par élément
produisent des sorties qui ont la
même forme que leurs opérandes.

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab jax
A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
B = A
A, A + B
```

Le [**produit élément par élément de deux matrices
est appelé leur *produit de Hadamard***] (noté $\odot$).
Nous pouvons détailler les entrées
du produit de Hadamard de deux matrices
$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$ :



$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
%%tab all
A * B
```

[**L'addition ou la multiplication d'un scalaire et d'un tenseur**] produit un résultat
avec la même forme que le tenseur original.
Ici, chaque élément du tenseur est ajouté au (ou multiplié par le) scalaire.

```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
%%tab jax
a = 2
X = jnp.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

## Réduction
:label:`subsec_lin-alg-reduction`

Souvent, nous souhaitons calculer [**la somme des éléments d'un tenseur.**]
Pour exprimer la somme des éléments d'un vecteur $\mathbf{x}$ de longueur $n$,
nous écrivons $\sum_{i=1}^n x_i$. Il existe une fonction simple pour cela :

```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
%%tab jax
x = jnp.arange(3, dtype=jnp.float32)
x, x.sum()
```

Pour exprimer des [**sommes sur les éléments de tenseurs de forme arbitraire**],
nous faisons simplement la somme sur tous ses axes.
Par exemple, la somme des éléments
d'une matrice $m \times n$ $\mathbf{A}$
pourrait s'écrire $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

Par défaut, l'invocation de la fonction somme
*réduit* un tenseur le long de tous ses axes,
produisant finalement un scalaire.
Nos bibliothèques nous permettent également de [**spécifier les axes
le long desquels le tenseur doit être réduit.**]
Pour faire la somme de tous les éléments le long des lignes (axe 0),
nous spécifions `axis=0` dans `sum`.
Comme la matrice d'entrée se réduit le long de l'axe 0
pour générer le vecteur de sortie,
cet axe est absent de la forme de la sortie.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

Spécifier `axis=1` réduira la dimension des colonnes (axe 1) en additionnant les éléments de toutes les colonnes.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

Réduire une matrice à la fois selon les lignes et les colonnes via une sommation
équivaut à additionner tous les éléments de la matrice.

```{.python .input}
%%tab mxnet, pytorch, jax
A.sum(axis=[0, 1]) == A.sum()  # Same as A.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # Same as tf.reduce_sum(A)
```

[**Une quantité liée est la *moyenne*.**]
Nous calculons la moyenne en divisant la somme
par le nombre total d'éléments.
Parce que le calcul de la moyenne est si courant,
il dispose d'une fonction de bibliothèque dédiée
qui fonctionne de manière analogue à `sum`.

```{.python .input}
%%tab mxnet, jax
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

De même, la fonction permettant de calculer la moyenne
peut également réduire un tenseur le long d'axes spécifiques.

```{.python .input}
%%tab mxnet, pytorch, jax
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## Somme sans réduction
:label:`subsec_lin-alg-non-reduction`

Parfois, il peut être utile de [**garder le nombre d'axes inchangé**]
lors de l'invocation de la fonction pour calculer la somme ou la moyenne.
Ceci est important lorsque nous voulons utiliser le mécanisme de diffusion (*broadcasting*).

```{.python .input}
%%tab mxnet, pytorch, jax
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

Par exemple, puisque `sum_A` conserve ses deux axes après avoir sommé chaque ligne,
nous pouvons (**diviser `A` par `sum_A` par diffusion**)
pour créer une matrice où chaque ligne totalise $1$.

```{.python .input}
%%tab all
A / sum_A
```

Si nous voulons calculer [**la somme cumulative des éléments de `A` le long d'un certain axe**],
disons `axis=0` (ligne par ligne), nous pouvons appeler la fonction `cumsum`.
Par conception, cette fonction ne réduit pas le tenseur d'entrée le long d'aucun axe.

```{.python .input}
%%tab mxnet, pytorch, jax
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## Produits scalaires

Jusqu'à présent, nous n'avons effectué que des opérations élément par élément, des sommes et des moyennes.
Et si c'était tout ce que nous pouvions faire, l'algèbre linéaire
ne mériterait pas sa propre section.
Heureusement, c'est ici que les choses deviennent plus intéressantes.
L'une des opérations les plus fondamentales est le produit scalaire.
Étant donné deux vecteurs $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$,
leur *produit scalaire* $\mathbf{x}^\top \mathbf{y}$ (également connu sous le nom de *produit intérieur*, $\langle \mathbf{x}, \mathbf{y}  \rangle$)
est une somme des produits des éléments à la même position :
$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~Le *produit scalaire* de deux vecteurs est une somme sur les produits des éléments à la même position~~]

```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
%%tab jax
y = jnp.ones(3, dtype = jnp.float32)
x, y, jnp.dot(x, y)
```

De manière équivalente, (**nous pouvons calculer le produit scalaire de deux vecteurs
en effectuant une multiplication élément par élément suivie d'une somme :**)

```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
%%tab jax
jnp.sum(x * y)
```

Les produits scalaires sont utiles dans un large éventail de contextes.
Par exemple, étant donné un certain ensemble de valeurs,
noté par un vecteur $\mathbf{x}  \in \mathbb{R}^n$,
et un ensemble de poids, noté par $\mathbf{w} \in \mathbb{R}^n$,
la somme pondérée des valeurs de $\mathbf{x}$
selon les poids $\mathbf{w}$
pourrait être exprimée par le produit scalaire $\mathbf{x}^\top \mathbf{w}$.
Lorsque les poids sont non négatifs
et que leur somme est égale à $1$, c'est-à-dire $\left(\sum_{i=1}^{n} {w_i} = 1\right)$,
le produit scalaire exprime une *moyenne pondérée*.
Après avoir normalisé deux vecteurs pour qu'ils aient une longueur unitaire,
les produits scalaires expriment le cosinus de l'angle entre eux.
Plus tard dans cette section, nous introduirons formellement cette notion de *longueur*.


## Produits matrice--vecteur

Maintenant que nous savons calculer les produits scalaires,
nous pouvons commencer à comprendre le *produit*
entre une matrice $m \times n$ $\mathbf{A}$
et un vecteur de dimension $n$ $\mathbf{x}$.
Pour commencer, nous visualisons notre matrice
en termes de ses vecteurs lignes

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

où chaque $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
est un vecteur ligne représentant la $i^\textrm{ème}$ ligne
de la matrice $\mathbf{A}$.

[**Le produit matrice--vecteur $\mathbf{A}\mathbf{x}$
est simplement un vecteur colonne de longueur $m$,
dont le $i^\textrm{ème}$ élément est le produit scalaire
$\mathbf{a}^\top_i \mathbf{x}$ :**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

Nous pouvons considérer la multiplication par une matrice
$\mathbf{A}\in \mathbb{R}^{m \times n}$
comme une transformation qui projette des vecteurs
de $\mathbb{R}^{n}$ vers $\mathbb{R}^{m}$.
Ces transformations sont remarquablement utiles.
Par exemple, nous pouvons représenter les rotations
comme des multiplications par certaines matrices carrées.
Les produits matrice--vecteur décrivent également
le calcul clé impliqué dans le calcul
des sorties de chaque couche d'un réseau de neurones
étant donné les sorties de la couche précédente.

:begin_tab:`mxnet`
Pour exprimer un produit matrice--vecteur en code,
nous utilisons la même fonction `dot`.
L'opération est inférée
en fonction du type des arguments.
Notez que la dimension des colonnes de `A`
(sa longueur le long de l'axe 1)
doit être la même que la dimension de `x` (sa longueur).
:end_tab:

:begin_tab:`pytorch`
Pour exprimer un produit matrice--vecteur en code,
nous utilisons la fonction `mv`.
Notez que la dimension des colonnes de `A`
(sa longueur le long de l'axe 1)
doit être la même que la dimension de `x` (sa longueur).
Python dispose d'un opérateur pratique `@`
qui peut exécuter à la fois des produits matrice--vecteur
et matrice--matrice
(selon ses arguments).
Ainsi, nous pouvons écrire `A@x`.
:end_tab:

:begin_tab:`tensorflow`
Pour exprimer un produit matrice--vecteur en code,
nous utilisons la fonction `matvec`.
Notez que la dimension des colonnes de `A`
(sa longueur le long de l'axe 1)
doit être la même que la dimension de `x` (sa longueur).
:end_tab:

```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
%%tab jax
A.shape, x.shape, jnp.matmul(A, x)
```

## Multiplication matrice--matrice

Une fois que vous avez compris les produits scalaires et les produits matrice--vecteur,
alors la *multiplication matrice--matrice* devrait être simple.

Supposons que nous ayons deux matrices
$\mathbf{A} \in \mathbb{R}^{n \times k}$
et $\mathbf{B} \in \mathbb{R}^{k \times m}$ :

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


Soit $\mathbf{a}^\top_{i} \in \mathbb{R}^k$
le vecteur ligne représentant la $i^\textrm{ème}$ ligne
de la matrice $\mathbf{A}$
et soit $\mathbf{b}_{j} \in \mathbb{R}^k$
le vecteur colonne de la $j^\textrm{ème}$ colonne
de la matrice $\mathbf{B}$ :

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


Pour former le produit matriciel $\mathbf{C} \in \mathbb{R}^{n \times m}$,
nous calculons simplement chaque élément $c_{ij}$
comme le produit scalaire entre
la $i^{\textrm{ème}}$ ligne de $\mathbf{A}$
et la $j^{\textrm{ème}}$ colonne de $\mathbf{B}$,
c'est-à-dire $\mathbf{a}^\top_i \mathbf{b}_j$ :

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**Nous pouvons considérer la multiplication matrice--matrice $\mathbf{AB}$
comme l'exécution de $m$ produits matrice--vecteur
ou de $m \times n$ produits scalaires
et l'assemblage des résultats
pour former une matrice $n \times m$.**]
Dans l'extrait suivant,
nous effectuons une multiplication matricielle sur `A` et `B`.
Ici, `A` est une matrice avec deux lignes et trois colonnes,
et `B` est une matrice avec trois lignes et quatre colonnes.
Après multiplication, nous obtenons une matrice avec deux lignes et quatre colonnes.

```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
%%tab jax
B = jnp.ones((3, 4))
jnp.matmul(A, B)
```

Le terme *multiplication matrice--matrice* est
souvent simplifié en *multiplication matricielle*,
et ne doit pas être confondu avec le produit de Hadamard.


## Normes
:label:`subsec_lin-algebra-norms`

Certains des opérateurs les plus utiles en algèbre linéaire sont les *normes*.
Informellement, la norme d'un vecteur nous dit à quel point il est *grand*.
Par exemple, la norme $\ell_2$ mesure
la longueur (euclidienne) d'un vecteur.
Ici, nous utilisons une notion de *taille* qui concerne l'ampleur des composantes d'un vecteur
(et non sa dimensionnalité).

Une norme est une fonction $\| \cdot \|$ qui projette un vecteur
vers un scalaire et satisfait les trois propriétés suivantes :

1. Étant donné n'importe quel vecteur $\mathbf{x}$, si nous redimensionnons (tous les éléments du) vecteur
   par un scalaire $\alpha \in \mathbb{R}$, sa norme change en conséquence :
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. Pour n'importe quels vecteurs $\mathbf{x}$ et $\mathbf{y}$ :
   les normes satisfont l'inégalité triangulaire :
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. La norme d'un vecteur est non négative et elle ne s'annule que si le vecteur est nul :
   $$\|\mathbf{x}\| > 0 \textrm{ pour tout } \mathbf{x} \neq 0.$$

De nombreuses fonctions sont des normes valides et différentes normes
encodent différentes notions de taille.
La norme euclidienne que nous avons tous apprise en géométrie à l'école primaire
lors du calcul de l'hypoténuse d'un triangle rectangle
est la racine carrée de la somme des carrés des éléments d'un vecteur.
Formellement, on l'appelle [**la *norme* $\ell_2$**] et elle s'exprime comme

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**)

La méthode `norm` calcule la norme $\ell_2$.

```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

```{.python .input}
%%tab jax
u = jnp.array([3.0, -4.0])
jnp.linalg.norm(u)
```

[**La norme $\ell_1$**] est également courante
et la mesure associée est appelée distance de Manhattan.
Par définition, la norme $\ell_1$ additionne
les valeurs absolues des éléments d'un vecteur :

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

Par rapport à la norme $\ell_2$, elle est moins sensible aux valeurs aberrantes.
Pour calculer la norme $\ell_1$,
nous composons la valeur absolue
avec l'opération de somme.

```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(u, ord=1) # same as jnp.abs(u).sum()
```

Les normes $\ell_2$ et $\ell_1$ sont toutes deux des cas particuliers
des *normes* $\ell_p$ plus générales :

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Dans le cas des matrices, les choses sont plus compliquées.
Après tout, les matrices peuvent être considérées à la fois comme des collections d'entrées individuelles
*et* comme des objets qui agissent sur des vecteurs et les transforment en d'autres vecteurs.
Par exemple, nous pouvons nous demander de combien plus long
le produit matrice--vecteur $\mathbf{X} \mathbf{v}$
pourrait être par rapport à $\mathbf{v}$.
Cette ligne de pensée mène à ce qu'on appelle la norme *spectrale*.
Pour l'instant, nous introduisons [**la *norme de Frobenius*,
qui est beaucoup plus facile à calculer**] et définie comme
la racine carrée de la somme des carrés
des éléments d'une matrice :

[**$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

La norme de Frobenius se comporte comme s'il s'agissait d'une
norme $\ell_2$ d'un vecteur ayant la forme d'une matrice.
L'invocation de la fonction suivante calculera
la norme de Frobenius d'une matrice.

```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(jnp.ones((4, 9)))
```

Bien que nous ne voulions pas trop anticiper sur la suite,
nous pouvons déjà donner une idée de la raison pour laquelle ces concepts sont utiles.
En apprentissage profond, nous essayons souvent de résoudre des problèmes d'optimisation :
*maximiser* la probabilité assignée aux données observées ;
*maximiser* les revenus associés à un modèle de recommandation ;
*minimiser* la distance entre les prédictions
et les observations de vérité terrain ;
*minimiser* la distance entre les représentations
de photos de la même personne
tout en *maximisant* la distance entre les représentations
de photos de personnes différentes.
Ces distances, qui constituent
les objectifs des algorithmes d'apprentissage profond,
sont souvent exprimées sous forme de normes.


## Discussion

Dans cette section, nous avons passé en revue toute l'algèbre linéaire
dont vous aurez besoin pour comprendre
une part importante de l'apprentissage profond moderne.
Il y a bien plus encore dans l'algèbre linéaire, cependant,
et une grande partie est utile pour l'apprentissage automatique.
Par exemple, les matrices peuvent être décomposées en facteurs,
et ces décompositions peuvent révéler
une structure de faible dimension dans des jeux de données réels.
Il existe des sous-domaines entiers de l'apprentissage automatique
qui se concentrent sur l'utilisation de décompositions matricielles
et de leurs généralisations à des tenseurs d'ordre élevé
pour découvrir une structure dans les jeux de données
et résoudre des problèmes de prédiction.
Mais ce livre se concentre sur l'apprentissage profond.
Et nous pensons que vous serez plus enclin
à apprendre davantage de mathématiques
une fois que vous vous serez sali les mains
en appliquant l'apprentissage automatique à de vrais jeux de données.
Ainsi, bien que nous nous réservions le droit
d'introduire plus de mathématiques plus tard,
nous concluons cette section ici.

Si vous avez hâte d'en apprendre davantage sur l'algèbre linéaire,
il existe de nombreux livres et ressources en ligne excellents.
Pour un cours accéléré plus avancé, envisagez de consulter
:citet:`Strang.1993`, :citet:`Kolter.2008` et :citet:`Petersen.Pedersen.ea.2008`.

Pour récapituler :

* Les scalaires, les vecteurs, les matrices et les tenseurs sont
  les objets mathématiques de base utilisés en algèbre linéaire
  et possèdent respectivement zéro, un, deux et un nombre arbitraire d'axes.
* Les tenseurs peuvent être découpés ou réduits le long d'axes spécifiés
  via l'indexation, ou des opérations telles que `sum` et `mean`, respectivement.
* Les produits élément par élément sont appelés produits de Hadamard.
  En revanche, les produits scalaires, les produits matrice--vecteur et les produits matrice--matrice
  ne sont pas des opérations élément par élément et renvoient en général des objets
  ayant des formes différentes de celles des opérandes.
* Par rapport aux produits de Hadamard, les produits matrice--matrice
  prennent beaucoup plus de temps à calculer (temps cubique plutôt que quadratique).
* Les normes capturent diverses notions de l'ampleur d'un vecteur (ou d'une matrice),
  et sont couramment appliquées à la différence de deux vecteurs
  pour mesurer leur distance.
* Les normes de vecteur courantes incluent les normes $\ell_1$ et $\ell_2$,
   et les normes de matrice courantes incluent les normes *spectrale* et de *Frobenius*.


## Exercices

1. Prouvez que la transposée de la transposée d'une matrice est la matrice elle-même : $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Étant donné deux matrices $\mathbf{A}$ et $\mathbf{B}$, montrez que la somme et la transposition commutent : $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Étant donné n'importe quelle matrice carrée $\mathbf{A}$, est-ce que $\mathbf{A} + \mathbf{A}^\top$ est toujours symétrique ? Pouvez-vous prouver le résultat en utilisant uniquement les résultats des deux exercices précédents ?
1. Nous avons défini le tenseur `X` de forme (2, 3, 4) dans cette section. Quel est le résultat de `len(X)` ? Écrivez votre réponse sans implémenter de code, puis vérifiez votre réponse à l'aide de code.
1. Pour un tenseur `X` de forme arbitraire, est-ce que `len(X)` correspond toujours à la longueur d'un certain axe de `X` ? Quel est cet axe ?
1. Exécutez `A / A.sum(axis=1)` et voyez ce qui se passe. Pouvez-vous analyser les résultats ?
1. Lors d'un déplacement entre deux points du centre de Manhattan, quelle est la distance que vous devez parcourir en fonction des coordonnées, c'est-à-dire en fonction des avenues et des rues ? Pouvez-vous voyager en diagonale ?
1. Considérons un tenseur de forme (2, 3, 4). Quelles sont les formes des sorties de sommation selon les axes 0, 1 et 2 ?
1. Fournissez un tenseur avec trois axes ou plus à la fonction `linalg.norm` et observez sa sortie. Que calcule cette fonction pour des tenseurs de forme arbitraire ?
1. Considérons trois grandes matrices, disons $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ et $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$, initialisées avec des variables aléatoires gaussiennes. Vous souhaitez calculer le produit $\mathbf{A} \mathbf{B} \mathbf{C}$. Y a-t-il une différence en termes d'empreinte mémoire et de vitesse, selon que vous calculiez $(\mathbf{A} \mathbf{B}) \mathbf{C}$ ou $\mathbf{A} (\mathbf{B} \mathbf{C})$ ? Pourquoi ?
1. Considérons trois grandes matrices, disons $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ et $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$. Y a-t-il une différence de vitesse selon que vous calculez $\mathbf{A} \mathbf{B}$ ou $\mathbf{A} \mathbf{C}^\top$ ? Pourquoi ? Qu'est-ce qui change si vous initialisez $\mathbf{C} = \mathbf{B}^\top$ sans cloner la mémoire ? Pourquoi ?
1. Considérons trois matrices, disons $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$. Construisez un tenseur à trois axes en empilant $[\mathbf{A}, \mathbf{B}, \mathbf{C}]$. Quelle est la dimensionnalité ? Extrayez la deuxième coordonnée du troisième axe pour récupérer $\mathbf{B}$. Vérifiez que votre réponse est correcte.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17968)
:end_tab:
