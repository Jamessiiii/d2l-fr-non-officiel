```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Stabilité numérique et initialisation
:label:`sec_numerical_stability`


Jusqu'à présent, chaque modèle que nous avons implémenté
nécessitait que nous initialisions ses paramètres
selon une certaine distribution spécifiée au préalable.
Jusqu'à présent, nous avons considéré le schéma d'initialisation comme acquis,
en passant sous silence les détails de la manière dont ces choix sont faits.
Vous pourriez même avoir eu l'impression que ces choix
ne sont pas particulièrement importants.
Au contraire, le choix du schéma d'initialisation
joue un rôle significatif dans l'apprentissage des réseaux de neurones,
et il peut être crucial pour maintenir la stabilité numérique.
De plus, ces choix peuvent être liés de manières intéressantes
au choix de la fonction d'activation non linéaire.
La fonction que nous choisissons et la manière dont nous initialisons les paramètres
peuvent déterminer la rapidité avec laquelle notre algorithme d'optimisation converge.
De mauvais choix ici peuvent nous amener à rencontrer
des gradients qui explosent ou qui disparaissent lors de l'entraînement.
Dans cette section, nous approfondissons ces sujets plus en détail
et discutons de quelques heuristiques utiles
que vous trouverez profitables
tout au long de votre carrière en deep learning.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from jax import grad, vmap
```

## Disparition et explosion du gradient

Considérons un réseau profond avec $L$ couches,
une entrée $\mathbf{x}$ et une sortie $\mathbf{o}$.
Avec chaque couche $l$ définie par une transformation $f_l$
paramétrée par des poids $\mathbf{W}^{(l)}$,
dont la sortie de la couche cachée est $\mathbf{h}^{(l)}$ (soit $\mathbf{h}^{(0)} = \mathbf{x}$),
notre réseau peut être exprimé comme :

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ et donc } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).$$

Si toutes les sorties des couches cachées et l'entrée sont des vecteurs,
nous pouvons écrire le gradient de $\mathbf{o}$ par rapport à
tout ensemble de paramètres $\mathbf{W}^{(l)}$ comme suit :

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.$$

En d'autres termes, ce gradient est
le produit de $L-l$ matrices
$\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$
et du vecteur gradient $\mathbf{v}^{(l)}$.
Ainsi, nous sommes susceptibles de rencontrer les mêmes
problèmes de dépassement de capacité inférieur (underflow) numérique qui surgissent souvent
lors de la multiplication d'un trop grand nombre de probabilités entre elles.
Lorsqu'on traite des probabilités, une astuce courante consiste à
passer dans l'espace logarithmique, c'est-à-dire à déplacer la
pression de la mantisse vers l'exposant
de la représentation numérique.
Malheureusement, notre problème ci-dessus est plus sérieux :
initialement, les matrices $\mathbf{M}^{(l)}$ peuvent avoir une grande variété de valeurs propres.
Elles peuvent être petites ou grandes, et
leur produit peut être *très grand* ou *très petit*.

Les risques posés par les gradients instables
vont au-delà de la représentation numérique.
Des gradients de magnitude imprévisible
menacent également la stabilité de nos algorithmes d'optimisation.
Nous pouvons être confrontés à des mises à jour de paramètres qui sont soit
(i) excessivement grandes, détruisant notre modèle
(le problème de l'*explosion du gradient*) ;
soit (ii) excessivement petites
(le problème de la *disparition du gradient*),
rendant l'apprentissage impossible car les paramètres
bougent à peine à chaque mise à jour.


### (**Disparition du gradient**)

L'un des coupables fréquents à l'origine du problème de disparition du gradient
est le choix de la fonction d'activation $\sigma$
qui est ajoutée après les opérations linéaires de chaque couche.
Historiquement, la fonction sigmoïde
$1/(1 + \exp(-x))$ (introduite dans :numref:`sec_mlp`)
était populaire car elle ressemble à une fonction de seuillage.
Étant donné que les premiers réseaux de neurones artificiels étaient inspirés
par les réseaux de neurones biologiques,
l'idée de neurones qui s'activent soit *complètement*, soit *pas du tout*
(comme les neurones biologiques) semblait séduisante.
Jetons un coup d'œil de plus près à la sigmoïde
pour voir pourquoi elle peut provoquer la disparition des gradients.

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.sigmoid(x)
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, [y, grad_sigmoid(x)],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

Comme vous pouvez le voir, (**le gradient de la sigmoïde disparaît
à la fois lorsque ses entrées sont grandes et lorsqu'elles sont petites**).
De plus, lors de la rétropropagation à travers de nombreuses couches,
à moins que nous ne soyons dans la zone « Boucle d'or » (Goldilocks zone), où
les entrées de bon nombre de sigmoïdes sont proches de zéro,
les gradients du produit global peuvent disparaître.
Lorsque notre réseau compte de nombreuses couches,
à moins d'être prudents, le gradient
sera probablement coupé à une certaine couche.
En effet, ce problème entravait autrefois l'entraînement des réseaux profonds.
Par conséquent, les ReLUs, qui sont plus stables
(mais moins plausibles d'un point de vue neuronal),
sont devenues le choix par défaut pour les praticiens.


### [**Explosion du gradient**]

Le problème inverse, lorsque les gradients explosent,
peut être tout aussi contrariant.
Pour mieux illustrer cela,
nous tirons 100 matrices aléatoires gaussiennes
et les multiplions par une certaine matrice initiale.
Pour l'échelle que nous avons choisie
(le choix de la variance $\sigma^2=1$),
le produit matriciel explose.
Lorsque cela se produit à cause de l'initialisation
d'un réseau profond, nous n'avons aucune chance de faire
converger un optimiseur par descente de gradient.

```{.python .input}
%%tab mxnet
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))
print('after multiplying 100 matrices', M)
```

```{.python .input}
%%tab pytorch
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)
```

```{.python .input}
%%tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))
print('after multiplying 100 matrices\n', M.numpy())
```

```{.python .input}
%%tab jax
get_key = lambda: jax.random.PRNGKey(d2l.get_seed())  # Generate PRNG keys
M = jax.random.normal(get_key(), (4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = jnp.matmul(M, jax.random.normal(get_key(), (4, 4)))
print('after multiplying 100 matrices\n', M)
```

### Briser la symétrie

Un autre problème dans la conception des réseaux de neurones
est la symétrie inhérente à leur paramétrage.
Supposons que nous ayons un MLP simple
avec une couche cachée et deux unités.
Dans ce cas, nous pourrions permuter les poids $\mathbf{W}^{(1)}$
de la première couche et de même permuter
les poids de la couche de sortie
pour obtenir la même fonction.
Il n'y a rien de spécial qui différencie
la première et la deuxième unité cachée.
En d'autres termes, nous avons une symétrie de permutation
parmi les unités cachées de chaque couche.

C'est plus qu'une simple nuisance théorique.
Considérons le MLP à une couche cachée susmentionné
avec deux unités cachées.
À titre d'illustration,
supposons que la couche de sortie transforme les deux unités cachées en une seule unité de sortie.
Imaginez ce qui se passerait si nous initialisions
tous les paramètres de la couche cachée
comme $\mathbf{W}^{(1)} = c$ pour une certaine constante $c$.
Dans ce cas, pendant la propagation avant,
l'une ou l'autre unité cachée reçoit les mêmes entrées et paramètres,
produisant la même activation
qui est transmise à l'unité de sortie.
Pendant la rétropropagation,
la dérivation de l'unité de sortie par rapport aux paramètres $\mathbf{W}^{(1)}$ donne un gradient dont tous les éléments prennent la même valeur.
Ainsi, après une itération basée sur le gradient (par exemple, la descente de gradient stochastique par mini-lots),
tous les éléments de $\mathbf{W}^{(1)}$ conservent la même valeur.
De telles itérations ne
*briseraient jamais la symétrie* d'elles-mêmes
et nous pourrions ne jamais être en mesure de réaliser
la puissance expressive du réseau.
La couche cachée se comporterait
comme si elle n'avait qu'une seule unité.
Notez que bien que la descente de gradient stochastique par mini-lots ne briserait pas cette symétrie,
la régularisation par dropout (qui sera introduite plus tard) le ferait !


## Initialisation des paramètres

Une façon d'aborder --- ou du moins d'atténuer --- les
problèmes soulevés ci-dessus consiste à procéder à une initialisation soignée.
Comme nous le verrons plus tard,
une attention particulière lors de l'optimisation
et une régularisation appropriée peuvent encore renforcer la stabilité.


### Initialisation par défaut

Dans les sections précédentes, par exemple dans :numref:`sec_linear_concise`,
nous avons utilisé une distribution normale
pour initialiser les valeurs de nos poids.
Si nous ne spécifions pas la méthode d'initialisation, le framework utilisera
une méthode d'initialisation aléatoire par défaut, qui fonctionne souvent bien en pratique
pour des problèmes de taille modérée.






### Initialisation de Xavier
:label:`subsec_xavier`

Examinons la distribution d'échelle de
l'une des sorties $o_{i}$ pour une certaine couche entièrement connectée
*sans non-linéarités*.
Avec $n_\textrm{in}$ entrées $x_j$
et leurs poids associés $w_{ij}$ pour cette couche,
une sortie est donnée par

$$o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.$$

Les poids $w_{ij}$ sont tous tirés
indépendamment de la même distribution.
De plus, supposons que cette distribution
ait une moyenne nulle et une variance $\sigma^2$.
Notez que cela ne signifie pas que la distribution doit être gaussienne,
mais seulement que la moyenne et la variance doivent exister.
Pour l'instant, supposons que les entrées de la couche $x_j$
aient également une moyenne nulle et une variance $\gamma^2$
et qu'elles soient indépendantes de $w_{ij}$ et indépendantes les unes des autres.
Dans ce cas, nous pouvons calculer la moyenne de $o_i$ :

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, \end{aligned}$$

et la variance :

$$
\begin{aligned}
    \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

Une façon de maintenir la variance fixe
est de poser $n_\textrm{in} \sigma^2 = 1$.
Considérons maintenant la rétropropagation.
Nous y sommes confrontés à un problème similaire,
bien que les gradients soient propagés à partir des couches les plus proches de la sortie.
En utilisant le même raisonnement que pour la propagation avant,
nous voyons que la variance des gradients peut exploser
à moins que $n_\textrm{out} \sigma^2 = 1$,
où $n_\textrm{out}$ est le nombre de sorties de cette couche.
Cela nous laisse face à un dilemme :
nous ne pouvons pas satisfaire les deux conditions simultanément.
Au lieu de cela, nous essayons simplement de satisfaire :

$$
\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ ou de manière équivalente }
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}
$$

C'est le raisonnement qui sous-tend l'*initialisation de Xavier*,
désormais standard et pratiquement bénéfique,
nommée d'après le premier auteur de ses créateurs :cite:`Glorot.Bengio.2010`.
Typiquement, l'initialisation de Xavier
échantillonne les poids à partir d'une distribution gaussienne
de moyenne nulle et de variance
$\sigma^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}$.
Nous pouvons également adapter cela pour
choisir la variance lors de l'échantillonnage des poids
à partir d'une distribution uniforme.
Notez que la distribution uniforme $U(-a, a)$ a une variance de $\frac{a^2}{3}$.
L'introduction de $\frac{a^2}{3}$ dans notre condition sur $\sigma^2$
nous incite à initialiser selon

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$

Bien que l'hypothèse d'inexistence de non-linéarités
dans le raisonnement mathématique ci-dessus
puisse être facilement violée dans les réseaux de neurones,
la méthode d'initialisation de Xavier
s'avère bien fonctionner en pratique.


### Au-delà

Le raisonnement ci-dessus effleure à peine la surface
des approches modernes d'initialisation des paramètres.
Un framework de deep learning implémente souvent plus d'une douzaine d'heuristiques différentes.
De plus, l'initialisation des paramètres continue d'être
un domaine de recherche fondamentale très actif en deep learning.
Parmi celles-ci figurent des heuristiques spécialisées pour
les paramètres liés (partagés), la super-résolution,
les modèles de séquence et d'autres situations.
Par exemple,
:citet:`Xiao.Bahri.Sohl-Dickstein.ea.2018` a démontré la possibilité d'entraîner
des réseaux de neurones de 10 000 couches sans astuces architecturales
en utilisant une méthode d'initialisation soigneusement conçue.

Si le sujet vous intéresse, nous vous suggérons
de vous plonger dans les offres de ce module,
de lire les articles qui ont proposé et analysé chaque heuristique,
puis d'explorer les dernières publications sur le sujet.
Peut-être tomberez-vous sur une idée ingénieuse ou en inventerez-vous une,
et contribuerez-vous à une implémentation dans les frameworks de deep learning.


## Résumé

La disparition et l'explosion des gradients sont des problèmes courants dans les réseaux profonds. Un grand soin dans l'initialisation des paramètres est nécessaire pour s'assurer que les gradients et les paramètres restent bien contrôlés.
Des heuristiques d'initialisation sont nécessaires pour s'assurer que les gradients initiaux ne sont ni trop grands ni trop petits.
L'initialisation aléatoire est essentielle pour garantir que la symétrie est brisée avant l'optimisation.
L'initialisation de Xavier suggère que, pour chaque couche, la variance de toute sortie n'est pas affectée par le nombre d'entrées, et la variance de tout gradient n'est pas affectée par le nombre de sorties.
Les fonctions d'activation ReLU atténuent le problème de disparition du gradient. Cela peut accélérer la convergence.

## Exercices

1. Pouvez-vous concevoir d'autres cas où un réseau de neurones pourrait présenter une symétrie qui nécessite d'être brisée, outre la symétrie de permutation dans les couches d'un MLP ?
1. Pouvons-nous initialiser tous les paramètres de poids dans la régression linéaire ou dans la régression softmax à la même valeur ?
1. Recherchez des bornes analytiques sur les valeurs propres du produit de deux matrices. Que vous dit cela sur la garantie que les gradients soient bien conditionnés ?
1. Si nous savons que certains termes divergent, pouvons-nous corriger cela après coup ? Consultez l'article sur la mise à l'échelle adaptative du taux par couche (LARS) pour vous en inspirer :cite:`You.Gitman.Ginsburg.2017`.


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17986)
:end_tab:
