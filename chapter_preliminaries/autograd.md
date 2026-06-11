```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Différentiation automatique
:label:`sec_autograd`

Rappelez-vous de :numref:`sec_calculus` 
que le calcul des dérivées est l'étape cruciale
dans tous les algorithmes d'optimisation
que nous utiliserons pour entraîner des réseaux profonds.
Bien que les calculs soient simples,
les effectuer à la main peut être fastidieux et sujet à erreur, 
et ces problèmes ne font que croître
à mesure que nos modèles deviennent plus complexes.

Heureusement, tous les frameworks de deep learning modernes
nous déchargent de ce travail
en proposant la *différentiation automatique*
(souvent abrégée en *autograd*). 
À mesure que nous passons les données à travers chaque fonction successive,
le framework construit un *graphe de calcul* 
qui suit comment chaque valeur dépend des autres.
Pour calculer les dérivées, 
la différentiation automatique 
travaille à rebours à travers ce graphe
en appliquant la règle de la chaîne. 
L'algorithme de calcul pour appliquer la règle de la chaîne
de cette manière est appelé *rétropropagation* (backpropagation).

Bien que les bibliothèques d'autograd soient devenues
une préoccupation majeure au cours de la dernière décennie,
elles ont une longue histoire. 
En fait, les premières références à l'autograd
remontent à plus d'un demi-siècle :cite:`Wengert.1964`.
Les idées fondamentales derrière la rétropropagation moderne
datent d'une thèse de doctorat de 1980 :cite:`Speelpenning.1980`
et ont été développées davantage à la fin des années 1980 :cite:`Griewank.1989`.
Bien que la rétropropagation soit devenue la méthode par défaut 
pour calculer les gradients, ce n'est pas la seule option. 
Pour instance, le langage de programmation Julia utilise 
la propagation avant (forward propagation) :cite:`Revels.Lubin.Papamarkou.2016`. 
Avant d'explorer les méthodes, 
maîtrisons d'abord le package autograd.

```{.python .input}
%%tab mxnet
from mxnet import autograd, np, npx
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

## Une fonction simple

Supposons que nous soyons intéressés par
(**la différenciation de la fonction
$y = 2\mathbf{x}^{\top}\mathbf{x}$
par rapport au vecteur colonne $\mathbf{x}$.**)
Pour commencer, nous assignons à `x` une valeur initiale.

```{.python .input  n=1}
%%tab mxnet
x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(4.0)
x
```

:begin_tab:`mxnet, pytorch, tensorflow`
[**Avant de calculer le gradient
de $y$ par rapport à $\mathbf{x}$,
nous avons besoin d'un endroit pour le stocker.**]
En général, nous évitons d'allouer de la nouvelle mémoire
chaque fois que nous calculons une dérivée 
car le deep learning nécessite 
le calcul successif de dérivées
par rapport aux mêmes paramètres
un très grand nombre de fois,
et nous risquerions de manquer de mémoire.
Notez que le gradient d'une fonction à valeur scalaire
par rapport à un vecteur $\mathbf{x}$
est à valeur vectorielle avec 
la même forme que $\mathbf{x}$.
:end_tab:

```{.python .input  n=8}
%%tab mxnet
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input  n=9}
%%tab pytorch
# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # The gradient is None by default
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

(**Nous calculons maintenant notre fonction de `x` et assignons le résultat à `y`.**)

```{.python .input  n=10}
%%tab mxnet
# Our code is inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
%%tab jax
y = lambda x: 2 * jnp.dot(x, x)
y(x)
```

:begin_tab:`mxnet`
[**Nous pouvons maintenant prendre le gradient de `y`
par rapport à `x`**] en appelant 
sa méthode `backward`.
Ensuite, nous pouvons accéder au gradient 
via l'attribut `grad` de `x`.
:end_tab:

:begin_tab:`pytorch`
[**Nous pouvons maintenant prendre le gradient de `y`
par rapport à `x`**] en appelant 
sa méthode `backward`.
Ensuite, nous pouvons accéder au gradient 
via l'attribut `grad` de `x`.
:end_tab:

:begin_tab:`tensorflow`
[**Nous pouvons maintenant calculer le gradient de `y`
par rapport à `x`**] en appelant 
la méthode `gradient`.
:end_tab:

:begin_tab:`jax`
[**Nous pouvons maintenant prendre le gradient de `y`
par rapport à `x`**] en passant par la
transformation `grad`.
:end_tab:

```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input}
%%tab jax
from jax import grad
# The `grad` transform returns a Python function that
# computes the gradient of the original function
x_grad = grad(y)(x)
x_grad
```

(**Nous savons déjà que le gradient de la fonction $y = 2\mathbf{x}^{\top}\mathbf{x}$
par rapport à $\mathbf{x}$ devrait être $4\mathbf{x}$.**)
Nous pouvons maintenant vérifier que le calcul automatique du gradient
et le résultat attendu sont identiques.

```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

```{.python .input}
%%tab jax
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**Calculons maintenant 
une autre fonction de `x`
et prenons son gradient.**] 
Notez que MXNet réinitialise le tampon de gradient 
chaque fois que nous enregistrons un nouveau gradient. 
:end_tab:

:begin_tab:`pytorch`
[**Calculons maintenant 
une autre fonction de `x`
et prenons son gradient.**]
Notez que PyTorch ne réinitialise pas automatiquement 
le tampon de gradient 
lorsque nous enregistrons un nouveau gradient. 
Au lieu de cela, le nouveau gradient
est ajouté au gradient déjà stocké.
Ce comportement est utile
lorsque nous voulons optimiser la somme 
de plusieurs fonctions d'objectif.
Pour réinitialiser le tampon de gradient,
nous pouvons appeler `x.grad.zero_()` comme suit :
:end_tab:

:begin_tab:`tensorflow`
[**Calculons maintenant 
une autre fonction de `x`
et prenons son gradient.**]
Notez que TensorFlow réinitialise le tampon de gradient 
chaque fois que nous enregistrons un nouveau gradient. 
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

```{.python .input}
%%tab jax
y = lambda x: x.sum()
grad(y)(x)
```

## Rétropropagation pour les variables non scalaires

Lorsque `y` est un vecteur, 
la représentation la plus naturelle 
de la dérivée de `y`
par rapport à un vecteur `x` 
est une matrice appelée la *Jacobienne*
qui contient les dérivées partielles
de chaque composante de `y` 
par rapport à chaque composante de `x`.
De même, pour des `y` et `x` d'ordre supérieur,
le résultat de la différenciation pourrait être un tenseur d'ordre encore plus élevé.

Bien que les Jacobiennes apparaissent dans certaines
techniques de machine learning avancées,
plus couramment nous voulons sommer 
les gradients de chaque composante de `y`
par rapport au vecteur complet `x`,
donnant un vecteur de la même forme que `x`.
Par exemple, nous avons souvent un vecteur 
représentant la valeur de notre fonction de perte
calculée séparément pour chaque exemple parmi
un *lot* (batch) d'exemples d'entraînement.
Ici, nous voulons simplement (**sommer les gradients
calculés individuellement pour chaque exemple**).

:begin_tab:`mxnet`
MXNet gère ce problème en réduisant tous les tenseurs en scalaires 
en sommant avant de calculer un gradient. 
En d'autres termes, plutôt que de renvoyer la Jacobienne 
$\partial_{\mathbf{x}} \mathbf{y}$,
il renvoie le gradient de la somme
$\partial_{\mathbf{x}} \sum_i y_i$. 
:end_tab:

:begin_tab:`pytorch`
Parce que les frameworks de deep learning varient 
dans leur façon d'interpréter les gradients de
tenseurs non scalaires,
PyTorch prend des mesures pour éviter toute confusion.
Invoquer `backward` sur un non-scalaire provoque une erreur 
à moins que nous n'indiquions à PyTorch comment réduire l'objet en un scalaire. 
Plus formellement, nous devons fournir un vecteur $\mathbf{v}$ 
tel que `backward` calculera 
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$ 
plutôt que $\partial_{\mathbf{x}} \mathbf{y}$. 
Cette partie suivante peut être déroutante,
mais pour des raisons qui deviendront claires plus tard, 
cet argument (représentant $\mathbf{v}$) est nommé `gradient`. 
Pour une description plus détaillée, voir le 
[bilan sur Medium](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29) de Yang Zhang. 
:end_tab:

:begin_tab:`tensorflow`
Par défaut, TensorFlow renvoie le gradient de la somme.
En d'autres termes, plutôt que de renvoyer 
la Jacobienne $\partial_{\mathbf{x}} \mathbf{y}$,
il renvoie le gradient de la somme
$\partial_{\mathbf{x}} \sum_i y_i$. 
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Equals the gradient of y = sum(x * x)
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as y = tf.reduce_sum(x * x)
```

```{.python .input}
%%tab jax
y = lambda x: x * x
# grad is only defined for scalar output functions
grad(lambda x: y(x).sum())(x)
```

## Détacher le calcul

Parfois, nous souhaitons [**déplacer certains calculs
en dehors du graphe de calcul enregistré.**]
Par exemple, supposons que nous utilisions l'entrée 
pour créer certains termes intermédiaires auxiliaires 
pour lesquels nous ne voulons pas calculer de gradient. 
Dans ce cas, nous devons *détacher* 
le graphe de calcul respectif
du résultat final. 
L'exemple jouet suivant rend cela plus clair : 
supposons que nous ayons `z = x * y` et `y = x * x` 
mais que nous voulions nous concentrer sur l'influence *directe* de `x` sur `z` 
plutôt que sur l'influence transmise via `y`. 
Dans ce cas, nous pouvons créer une nouvelle variable `u`
qui prend la même valeur que `y` 
mais dont la *provenance* (comment elle a été créée)
a été effacée.
Ainsi, `u` n'a pas d'ancêtres dans le graphe
et les gradients ne coulent pas à travers `u` vers `x`.
Par exemple, prendre le gradient de `z = x * u`
donnera le résultat `u`,
(pas `3 * x * x` comme vous auriez pu 
vous y attendre puisque `z = x * x * x`).

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# Set persistent=True to preserve the compute graph. 
# This lets us run t.gradient more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
%%tab jax
import jax

y = lambda x: x * x
# jax.lax primitives are Python wrappers around XLA operations
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x

grad(lambda x: z(x).sum())(x) == y(x)
```

Notez que bien que cette procédure
détache les ancêtres de `y`
du graphe menant à `z`, 
le graphe de calcul menant à `y` 
persiste et nous pouvons donc calculer
le gradient de `y` par rapport à `x`.

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input}
%%tab jax
grad(lambda x: y(x).sum())(x) == 2 * x
```

## Gradients et flux de contrôle Python

Jusqu'à présent, nous avons passé en revue des cas où le chemin de l'entrée à la sortie 
était bien défini via une fonction telle que `z = x * x * x`.
La programmation nous offre beaucoup plus de liberté dans la façon dont nous calculons les résultats. 
Par exemple, nous pouvons les faire dépendre de variables auxiliaires 
ou conditionner des choix sur des résultats intermédiaires. 
L'un des avantages de l'utilisation de la différentiation automatique
est que [**même si**] la construction du graphe de calcul de 
(**une fonction nécessitait de passer par un labyrinthe de flux de contrôle Python**)
(par exemple, des conditions, des boucles et des appels de fonction arbitraires),
(**nous pouvons toujours calculer le gradient de la variable résultante.**)
Pour illustrer cela, considérons l'extrait de code suivant où 
le nombre d'itérations de la boucle `while`
et l'évaluation de l'instruction `if`
dépendent tous deux de la valeur de l'entrée `a`.

```{.python .input}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab jax
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Ci-dessous, nous appelons cette fonction, en passant une valeur aléatoire, comme entrée.
Comme l'entrée est une variable aléatoire, 
nous ne savons pas quelle forme 
le graphe de calcul prendra.
Cependant, chaque fois que nous exécutons `f(a)` 
sur une entrée spécifique, nous réalisons 
un graphe de calcul spécifique
et pouvons ensuite exécuter `backward`.

```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input}
%%tab jax
from jax import random
a = random.normal(random.PRNGKey(1), ())
d = f(a)
d_grad = grad(f)(a)
```

Même si notre fonction `f` est, à des fins de démonstration, un peu artificielle,
sa dépendance par rapport à l'entrée est assez simple : 
c'est une fonction *linéaire* de `a` 
avec une échelle définie par morceaux. 
À ce titre, `f(a) / a` est un vecteur d'entrées constantes 
et, de plus, `f(a) / a` doit correspondre 
au gradient de `f(a)` par rapport à `a`.

```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

```{.python .input}
%%tab jax
d_grad == d / a
```

Le flux de contrôle dynamique est très courant en deep learning. 
Par exemple, lors du traitement de texte, le graphe de calcul
dépend de la longueur de l'entrée. 
Dans ces cas, la différentiation automatique 
devient vitale pour la modélisation statistique 
puisqu'il est impossible de calculer le gradient *a priori*. 

## Discussion

Vous avez maintenant eu un aperçu de la puissance de la différentiation automatique. 
Le développement de bibliothèques pour calculer les dérivées
de manière à la fois automatique et efficace 
a été un gain de productivité massif
pour les praticiens du deep learning,
les libérant pour qu'ils puissent se concentrer sur des tâches moins ingrates.
De plus, autograd nous permet de concevoir des modèles massifs
pour lesquels les calculs de gradient au stylo et au papier 
seraient prohibitifs en temps.
Il est intéressant de noter que, alors que nous utilisons autograd pour *optimiser* des modèles
(au sens statistique),
l'*optimisation* des bibliothèques d'autograd elles-mêmes
(au sens informatique)
est un sujet riche
d'un intérêt vital pour les concepteurs de frameworks.
Ici, des outils issus des compilateurs et de la manipulation de graphes 
sont exploités pour calculer les résultats 
de la manière la plus rapide et la plus économe en mémoire. 

Pour l'instant, essayez de vous souvenir de ces bases : (i) attacher des gradients aux variables par rapport auxquelles nous désirons des dérivées ; (ii) enregistrer le calcul de la valeur cible ; (iii) exécuter la fonction de rétropropagation ; et (iv) accéder au gradient résultant.


## Exercices

1. Pourquoi la dérivée seconde est-elle beaucoup plus coûteuse à calculer que la dérivée première ?
1. Après avoir exécuté la fonction de rétropropagation, exécutez-la à nouveau immédiatement et voyez ce qui se passe. Enquêtez.
1. Dans l'exemple de flux de contrôle où nous calculons la dérivée de `d` par rapport à `a`, que se passerait-il si nous changions la variable `a` en un vecteur aléatoire ou une matrice ? À ce stade, le résultat du calcul `f(a)` n'est plus un scalaire. Qu'advient-il du résultat ? Comment analyser cela ?
1. Soit $f(x) = \sin(x)$. Tracez le graphe de $f$ et de sa dérivée $f'$. N'exploitez pas le fait que $f'(x) = \cos(x)$ mais utilisez plutôt la différentiation automatique pour obtenir le résultat. 
1. Soit $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$. Écrivez un graphe de dépendance traçant les résultats de $x$ vers $f(x)$. 
1. Utilisez la règle de la chaîne pour calculer la dérivée $\frac{df}{dx}$ de la fonction susmentionnée, en plaçant chaque terme sur le graphe de dépendance que vous avez construit précédemment. 
1. Étant donné le graphe et les résultats de dérivées intermédiaires, vous avez un certain nombre d'options lors du calcul du gradient. Évaluez le résultat une fois en partant de $x$ vers $f$ et une fois de $f$ en remontant vers $x$. Le chemin de $x$ vers $f$ est communément appelé *différenciation avant* (forward differentiation), tandis que le chemin de $f$ vers $x$ est appelé différenciation arrière (backward differentiation). 
1. Quand pourriez-vous vouloir utiliser la différenciation avant, et quand la différenciation arrière ? Indice : considérez la quantité de données intermédiaires nécessaires, la capacité à paralléliser les étapes, et la taille des matrices et vecteurs impliqués. 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17970)
:end_tab: