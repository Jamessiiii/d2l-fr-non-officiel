```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Manipulation de données
:label:`sec_ndarray`

Pour pouvoir faire quoi que ce soit, 
nous avons besoin d'un moyen de stocker et de manipuler des données.
En général, il y a deux choses importantes 
que nous devons faire avec les données : 
(i) les acquérir ; 
et (ii) les traiter une fois qu'elles sont dans l'ordinateur. 
Il est inutile d'acquérir des données 
sans avoir un moyen de les stocker, 
alors pour commencer, mettons les mains à la pâte 
avec les tableaux à $n$ dimensions, 
que nous appelons aussi *tenseurs*.
Si vous connaissez déjà le package de calcul scientifique 
NumPy, ce sera un jeu d'enfant.
Pour tous les frameworks d'apprentissage profond modernes,
la *classe de tenseur* (`ndarray` dans MXNet, 
`Tensor` dans PyTorch et TensorFlow) 
ressemble au `ndarray` de NumPy,
avec quelques fonctionnalités redoutables en plus.
Premièrement, la classe de tenseur
prend en charge la différenciation automatique.
Deuxièmement, elle tire parti des GPU
pour accélérer le calcul numérique,
alors que NumPy ne fonctionne que sur les CPU.
Ces propriétés rendent les réseaux de neurones
à la fois faciles à coder et rapides à exécuter.



## Premiers pas

:begin_tab:`mxnet`
Pour commencer, nous importons les modules `np` (`numpy`) et
`npx` (`numpy_extension`) de MXNet.
Ici, le module `np` comprend 
des fonctions prises en charge par NumPy,
tandis que le module `npx` contient un ensemble d'extensions
développées pour renforcer l'apprentissage profond 
dans un environnement de type NumPy.
Lorsque nous utilisons des tenseurs, nous invoquons 
presque toujours la fonction `set_np` :
c'est pour la compatibilité du traitement des tenseurs 
par d'autres composants de MXNet.
:end_tab:

:begin_tab:`pytorch`
(**Pour commencer, nous importons la bibliothèque PyTorch.
Notez que le nom du package est `torch`.**)
:end_tab:

:begin_tab:`tensorflow`
Pour commencer, nous importons `tensorflow`. 
Par souci de brièveté, les praticiens 
attribuent souvent l'alias `tf`.
:end_tab:

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
import jax
from jax import numpy as jnp
```

[**Un tenseur représente un tableau (éventuellement multidimensionnel) de valeurs numériques.**]
Dans le cas unidimensionnel, c'est-à-dire quand un seul axe est nécessaire pour les données,
un tenseur est appelé un *vecteur*.
Avec deux axes, un tenseur est appelé une *matrice*.
Avec $k > 2$ axes, nous abandonnons les noms spécialisés
et nous nous référons simplement à l'objet comme un *tenseur d'ordre* $k$.

:begin_tab:`mxnet`
MXNet fournit une variété de fonctions 
pour créer de nouveaux tenseurs 
pré-remplis de valeurs. 
Par exemple, en invoquant `arange(n)`,
nous pouvons créer un vecteur de valeurs régulièrement espacées,
commençant à 0 (inclus) 
et se terminant à `n` (exclu).
Par défaut, la taille de l'intervalle est de $1$.
Sauf indication contraire, 
les nouveaux tenseurs sont stockés dans la mémoire principale 
et destinés au calcul basé sur le CPU.
:end_tab:

:begin_tab:`pytorch`
PyTorch fournit une variété de fonctions 
pour créer de nouveaux tenseurs 
pré-remplis de valeurs. 
Par exemple, en invoquant `arange(n)`,
nous pouvons créer un vecteur de valeurs régulièrement espacées,
commençant à 0 (inclus) 
et se terminant à `n` (exclu).
Par défaut, la taille de l'intervalle est de $1$.
Sauf indication contraire, 
les nouveaux tenseurs sont stockés dans la mémoire principale 
et destinés au calcul basé sur le CPU.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow fournit une variété de fonctions 
pour créer de nouveaux tenseurs 
pré-remplis de valeurs. 
Par exemple, en invoquant `range(n)`,
nous pouvons créer un vecteur de valeurs régulièrement espacées,
commençant à 0 (inclus) 
et se terminant à `n` (exclu).
Par défaut, la taille de l'intervalle est de $1$.
Sauf indication contraire, 
les nouveaux tenseurs sont stockés dans la mémoire principale 
et destinés au calcul basé sur le CPU.
:end_tab:

```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(12)
x
```

:begin_tab:`mxnet`
Chacune de ces valeurs est appelée
un *élément* du tenseur.
Le tenseur `x` contient 12 éléments.
Nous pouvons inspecter le nombre total d'éléments 
dans un tenseur via son attribut `size`.
:end_tab:

:begin_tab:`pytorch`
Chacune de ces valeurs est appelée
un *élément* du tenseur.
Le tenseur `x` contient 12 éléments.
Nous pouvons inspecter le nombre total d'éléments 
dans un tenseur via sa méthode `numel`.
:end_tab:

:begin_tab:`tensorflow`
Chacune de ces valeurs est appelée
un *élément* du tenseur.
Le tenseur `x` contient 12 éléments.
Nous pouvons inspecter le nombre total d'éléments 
dans un tenseur via la fonction `size`.
:end_tab:

```{.python .input}
%%tab mxnet, jax
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

(**Nous pouvons accéder à la *forme* (*shape*) d'un tenseur**) 
(la longueur le long de chaque axe)
en inspectant son attribut `shape`.
Comme nous traitons d'un vecteur ici,
la `shape` ne contient qu'un seul élément
et est identique à la taille.

```{.python .input}
%%tab all
x.shape
```

Nous pouvons [**changer la forme d'un tenseur
sans altérer sa taille ou ses valeurs**],
en invoquant `reshape`.
Par exemple, nous pouvons transformer 
notre vecteur `x` dont la forme est (12,) 
en une matrice `X` de forme (3, 4).
Ce nouveau tenseur conserve tous les éléments
mais les reconfigure en une matrice.
Notez que les éléments de notre vecteur
sont disposés ligne par ligne et ainsi
`x[3] == X[0, 3]`.

```{.python .input}
%%tab mxnet, pytorch, jax
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

Notez que spécifier chaque composant de la forme à `reshape` est redondant.
Puisque nous connaissons déjà la taille de notre tenseur,
nous pouvons déduire un composant de la forme à partir des autres.
Par exemple, étant donné un tenseur de taille $n$
et une forme cible ($h$, $w$),
nous savons que $w = n/h$.
Pour inférer automatiquement un composant de la forme,
nous pouvons placer un `-1` pour le composant de la forme
qui doit être inféré automatiquement.
Dans notre cas, au lieu d'appeler `x.reshape(3, 4)`,
nous aurions pu appeler de manière équivalente `x.reshape(-1, 4)` ou `x.reshape(3, -1)`.

Les praticiens ont souvent besoin de travailler avec des tenseurs
initialisés pour contenir uniquement des 0 ou des 1.
[**Nous pouvons construire un tenseur avec tous les éléments fixés à 0**] (~~ou 1~~)
et une forme de (2, 3, 4) via la fonction `zeros`.

```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.zeros((2, 3, 4))
```

De même, nous pouvons créer un tenseur 
avec tous les 1 en invoquant `ones`.

```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.ones((2, 3, 4))
```

Nous souhaitons souvent 
[**échantillonner chaque élément de manière aléatoire (et indépendante)**] 
à partir d'une distribution de probabilité donnée.
Par exemple, les paramètres des réseaux de neurones
sont souvent initialisés de manière aléatoire.
L'extrait suivant crée un tenseur 
avec des éléments tirés d'une 
distribution gaussienne (normale) standard
avec une moyenne de 0 et un écart-type de 1.

```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
%%tab jax
# Any call of a random function in JAX requires a key to be
# specified, feeding the same key to a random function will
# always result in the same sample being generated
jax.random.normal(jax.random.PRNGKey(0), (3, 4))
```

Enfin, nous pouvons construire des tenseurs en
[**fournissant les valeurs exactes de chaque élément**] 
par des listes Python (éventuellement imbriquées) 
contenant des littéraux numériques.
Ici, nous construisons une matrice avec une liste de listes,
où la liste la plus externe correspond à l'axe 0,
et la liste interne correspond à l'axe 1.

```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab jax
jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Indexation et découpage

Comme avec les listes Python,
nous pouvons accéder aux éléments d'un tenseur 
par indexation (en commençant par 0).
Pour accéder à un élément en fonction de sa position
par rapport à la fin de la liste,
nous pouvons utiliser l'indexation négative.
Enfin, nous pouvons accéder à des plages entières d'indices 
via le découpage (*slicing*, par exemple `X[start:stop]`), 
où la valeur retournée inclut 
le premier indice (`start`) *mais pas le dernier* (`stop`).
Enfin, lorsqu'un seul indice (ou découpe)
est spécifié pour un tenseur d'ordre $k$,
il est appliqué le long de l'axe 0.
Ainsi, dans le code suivant,
[**`[-1]` sélectionne la dernière ligne et `[1:3]`
sélectionne les deuxième et troisième lignes**].

```{.python .input}
%%tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
Au-delà de leur lecture, (**nous pouvons également *écrire* les éléments d'une matrice en spécifiant des indices.**)
:end_tab:

:begin_tab:`tensorflow`
Les `Tensors` dans TensorFlow sont immuables et ne peuvent pas faire l'objet d'une affectation.
Les `Variables` dans TensorFlow sont des conteneurs d'état mutables qui prennent en charge
les affectations. Gardez à l'esprit que les gradients dans TensorFlow ne remontent pas
à travers les affectations de `Variable`.

Au-delà de l'affectation d'une valeur à l'ensemble de la `Variable`, nous pouvons écrire des éléments d'une
`Variable` en spécifiant des indices.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

```{.python .input}
%%tab jax
# JAX arrays are immutable. jax.numpy.ndarray.at index
# update operators create a new array with the corresponding
# modifications made
X_new_1 = X.at[1, 2].set(17)
X_new_1
```

Si nous voulons [**affecter la même valeur à plusieurs éléments,
nous appliquons l'indexation sur le côté gauche 
de l'opération d'affectation.**]
Par exemple, `[:2, :]` accède 
aux première et deuxième lignes,
où `:` prend tous les éléments le long de l'axe 1 (colonne).
Bien que nous ayons discuté de l'indexation pour les matrices,
cela fonctionne également pour les vecteurs
et pour les tenseurs de plus de deux dimensions.

```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

```{.python .input}
%%tab jax
X_new_2 = X_new_1.at[:2, :].set(12)
X_new_2
```

## Opérations

Maintenant que nous savons comment construire des tenseurs
et comment lire et écrire dans leurs éléments,
nous pouvons commencer à les manipuler
avec diverses opérations mathématiques.
Parmi les plus utiles de celles-ci 
figurent les opérations *élément par élément* (*elementwise*).
Celles-ci appliquent une opération scalaire standard
à chaque élément d'un tenseur.
Pour les fonctions qui prennent deux tenseurs en entrée,
les opérations élément par élément appliquent un opérateur binaire standard
sur chaque paire d'éléments correspondants.
Nous pouvons créer une fonction élément par élément 
à partir de n'importe quelle fonction qui fait correspondre 
un scalaire à un scalaire.

En notation mathématique, nous désignons de tels
opérateurs scalaires *unaires* (prenant une seule entrée)
par la signature 
$f: \mathbb{R} \rightarrow \mathbb{R}$.
Cela signifie simplement que la fonction fait correspondre
n'importe quel nombre réel à un autre nombre réel.
La plupart des opérateurs standards, y compris les opérateurs unaires comme $e^x$, peuvent être appliqués élément par élément.

```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

```{.python .input}
%%tab jax
jnp.exp(x)
```

De même, nous désignons les opérateurs scalaires *binaires*,
qui font correspondre des paires de nombres réels
à un (seul) nombre réel
via la signature 
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Étant donné deux vecteurs $\mathbf{u}$ 
et $\mathbf{v}$ *de même forme*,
et un opérateur binaire $f$, nous pouvons produire un vecteur
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$
en fixant $c_i \gets f(u_i, v_i)$ pour tout $i$,
où $c_i, u_i$ et $v_i$ sont les $i$-èmes éléments
des vecteurs $\mathbf{c}, \mathbf{u}$ et $\mathbf{v}$.
Ici, nous avons produit le vecteur
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
en *élevant* (*lifting*) la fonction scalaire
à une opération de vecteur élément par élément.
Les opérateurs arithmétiques standards communs
pour l'addition (`+`), la soustraction (`-`), 
la multiplication (`*`), la division (`/`) 
et l'exponentiation (`**`)
ont tous été élevés à des opérations élément par élément
pour des tenseurs de forme identique de forme arbitraire.

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab jax
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

En plus des calculs élément par élément,
nous pouvons également effectuer des opérations d'algèbre linéaire,
telles que les produits scalaires et les multiplications de matrices.
Nous détaillerons ces points
dans :numref:`sec_linear-algebra`.

Nous pouvons également [***concaténer* plusieurs tenseurs,**]
en les empilant bout à bout pour en former un plus grand.
Il suffit de fournir une liste de tenseurs
et de dire au système le long de quel axe concaténer.
L'exemple ci-dessous montre ce qui se passe lorsque nous concaténons
deux matrices le long des lignes (axe 0)
au lieu des colonnes (axe 1).
On peut voir que la longueur de l'axe 0 de la première sortie ($6$)
est la somme des longueurs de l'axe 0 des deux tenseurs d'entrée ($3 + 3$) ;
alors que la longueur de l'axe 1 de la deuxième sortie ($8$)
est la somme des longueurs de l'axe 1 des deux tenseurs d'entrée ($4 + 4$).

```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

```{.python .input}
%%tab jax
X = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
Y = jnp.array([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
jnp.concatenate((X, Y), axis=0), jnp.concatenate((X, Y), axis=1)
```

Parfois, nous voulons 
[**construire un tenseur binaire via des *énoncés logiques*.**]
Prenons `X == Y` comme exemple.
Pour chaque position `i, j`, si `X[i, j]` et `Y[i, j]` sont égaux, 
alors l'entrée correspondante dans le résultat prend la valeur `1`,
sinon elle prend la valeur `0`.

```{.python .input}
%%tab all
X == Y
```

[**Faire la somme de tous les éléments du tenseur**] produit un tenseur avec un seul élément.

```{.python .input}
%%tab mxnet, pytorch, jax
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## Diffusion (Broadcasting)
:label:`subsec_broadcasting`

À présent, vous savez comment effectuer 
des opérations binaires élément par élément
sur deux tenseurs de même forme. 
Sous certaines conditions,
même lorsque les formes diffèrent, 
nous pouvons toujours [**effectuer des opérations binaires élément par élément
en invoquant le *mécanisme de diffusion* (*broadcasting*).**]
La diffusion fonctionne selon 
la procédure en deux étapes suivante :
(i) étendre l'un des tableaux ou les deux
en copiant les éléments le long des axes de longueur 1
de sorte qu'après cette transformation,
les deux tenseurs aient la même forme ;
(ii) effectuer une opération élément par élément
sur les tableaux résultants.

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
%%tab jax
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
a, b
```

Comme `a` et `b` sont respectivement des matrices $3\times1$ 
et $1\times2$, leurs formes ne correspondent pas.
La diffusion produit une matrice plus grande $3\times2$ 
en répliquant la matrice `a` le long des colonnes
et la matrice `b` le long des lignes
avant de les additionner élément par élément.

```{.python .input}
%%tab all
a + b
```

## Économiser de la mémoire

[**L'exécution d'opérations peut entraîner l'allocation d'une nouvelle mémoire 
pour héberger les résultats.**]
Par exemple, si nous écrivons `Y = X + Y`,
nous déréférençons le tenseur vers lequel `Y` pointait auparavant
et pointons plutôt `Y` vers la mémoire nouvellement allouée.
Nous pouvons démontrer ce problème avec la fonction `id()` de Python,
qui nous donne l'adresse exacte 
de l'objet référencé en mémoire.
Notez qu'après avoir exécuté `Y = Y + X`,
`id(Y)` pointe vers un emplacement différent.
C'est parce que Python évalue d'abord `Y + X`,
allouant de la nouvelle mémoire pour le résultat 
puis pointe `Y` vers ce nouvel emplacement en mémoire.

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

Cela peut être indésirable pour deux raisons.
Premièrement, nous ne voulons pas passer notre temps
à allouer de la mémoire inutilement tout le temps.
En apprentissage automatique, nous avons souvent
des centaines de mégaoctets de paramètres
et nous les mettons tous à jour plusieurs fois par seconde.
Autant que possible, nous voulons effectuer ces mises à jour *sur place* (*in place*).
Deuxièmement, nous pourrions pointer vers les 
mêmes paramètres à partir de plusieurs variables.
Si nous ne mettons pas à jour sur place, 
nous devons faire attention à mettre à jour toutes ces références,
de peur de provoquer une fuite de mémoire 
ou de se référer par inadvertance à des paramètres obsolètes.

:begin_tab:`mxnet, pytorch`
Heureusement, (**effectuer des opérations sur place**) est facile.
Nous pouvons affecter le résultat d'une opération
à un tableau `Y` précédemment alloué
en utilisant la notation de découpage : `Y[:] = <expression>`.
Pour illustrer ce concept, 
nous écrasons les valeurs du tenseur `Z`,
après l'avoir initialisé à l'aide de `zeros_like`,
pour avoir la même forme que `Y`.
:end_tab:

:begin_tab:`tensorflow`
Les `Variables` sont des conteneurs d'état mutables dans TensorFlow. Elles fournissent
un moyen de stocker les paramètres de votre modèle.
Nous pouvons affecter le résultat d'une opération
à une `Variable` avec `assign`.
Pour illustrer ce concept, 
nous écrasons les valeurs de la `Variable` `Z`
après l'avoir initialisée, à l'aide de `zeros_like`,
pour avoir la même forme que `Y`.
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
%%tab jax
# JAX arrays do not allow in-place operations
```

:begin_tab:`mxnet, pytorch`
[**Si la valeur de `X` n'est pas réutilisée dans les calculs ultérieurs,
nous pouvons également utiliser `X[:] = X + Y` ou `X += Y`
pour réduire la charge mémoire de l'opération.**]
:end_tab:

:begin_tab:`tensorflow`
Même une fois que vous stockez l'état de manière persistante dans une `Variable`, 
vous voudrez peut-être réduire davantage votre utilisation de la mémoire en évitant les
allocations excessives pour les tenseurs qui ne sont pas les paramètres de votre modèle.
Comme les `Tensors` TensorFlow sont immuables 
et que les gradients ne remontent pas via les affectations de `Variable`, 
TensorFlow ne fournit pas de moyen explicite d'exécuter
une opération individuelle sur place.

Cependant, TensorFlow fournit le décorateur `tf.function` 
pour envelopper le calcul à l'intérieur d'un graphe TensorFlow 
qui est compilé et optimisé avant l'exécution.
Cela permet à TensorFlow d'élaguer les valeurs inutilisées 
et de réutiliser les allocations antérieures qui ne sont plus nécessaires. 
Cela minimise la charge mémoire des calculs TensorFlow.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be reused when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## Conversion vers d'autres objets Python

:begin_tab:`mxnet, tensorflow`
[**La conversion en tenseur NumPy (`ndarray`)**], ou vice versa, est facile.
Le résultat converti ne partage pas la mémoire.
Ce petit inconvénient est en fait assez important :
lorsque vous effectuez des opérations sur le CPU ou sur des GPU,
vous ne voulez pas arrêter le calcul, en attendant de voir
si le package NumPy de Python 
pourrait vouloir faire autre chose
avec le même morceau de mémoire.
:end_tab:

:begin_tab:`pytorch`
[**La conversion en tenseur NumPy (`ndarray`)**], ou vice versa, est facile.
Le tenseur Torch et le tableau NumPy 
partageront leur mémoire sous-jacente, 
et la modification de l'un via une opération sur place 
modifiera également l'autre.
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
%%tab jax
A = jax.device_get(X)
B = jax.device_put(A)
type(A), type(B)
```

Pour (**convertir un tenseur de taille 1 en scalaire Python**),
nous pouvons invoquer la fonction `item` ou les fonctions intégrées de Python.

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab jax
a = jnp.array([3.5])
a, a.item(), float(a), int(a)
```

## Résumé

La classe de tenseur est l'interface principale pour stocker et manipuler des données dans les bibliothèques d'apprentissage profond.
Les tenseurs offrent une variété de fonctionnalités, notamment des routines de construction, l'indexation et le découpage, des opérations mathématiques de base, la diffusion, l'affectation économe en mémoire et la conversion vers et depuis d'autres objets Python.


## Exercices

1. Exécutez le code de cette section. Remplacez l'énoncé conditionnel `X == Y` par `X < Y` ou `X > Y`, puis voyez quel type de tenseur vous obtenez.
1. Remplacez les deux tenseurs qui opèrent par élément dans le mécanisme de diffusion par d'autres formes, par exemple des tenseurs à 3 dimensions. Le résultat est-il le même que celui attendu ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17966)
:end_tab:
