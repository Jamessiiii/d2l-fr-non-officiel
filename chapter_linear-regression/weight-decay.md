```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Décroissance des poids
:label:`sec_weight_decay`

Maintenant que nous avons caractérisé le problème du surapprentissage,
nous pouvons introduire notre première technique de *régularisation*.
Rappelez-vous que nous pouvons toujours atténuer le surapprentissage
en collectant davantage de données d'entraînement.
Cependant, cela peut être coûteux, chronophage,
ou entièrement hors de notre contrôle,
ce qui le rend impossible à court terme.
Pour l'instant, nous pouvons supposer que nous disposons déjà
d'autant de données de haute qualité que nos ressources le permettent
et nous concentrer sur les outils à notre disposition
lorsque le jeu de données est considéré comme une donnée.

Rappelez-vous que dans notre exemple de régression polynomiale
(:numref:`subsec_polynomial-curve-fitting`),
nous pouvions limiter la capacité de notre modèle
en ajustant le degré
du polynôme ajusté.
En effet, limiter le nombre de caractéristiques
est une technique populaire pour atténuer le surapprentissage.
Cependant, le simple fait d'écarter des caractéristiques
peut être un instrument trop brutal.
En restant sur l'exemple de la régression polynomiale,
considérez ce qui pourrait arriver
avec une entrée de haute dimension.
Les extensions naturelles des polynômes
aux données multivariées sont appelées *monômes*,
qui sont simplement des produits de puissances de variables.
Le degré d'un monôme est la somme des puissances.
Par exemple, $x_1^2 x_2$ et $x_3 x_5^2$
sont tous deux des monômes de degré 3.

Notez que le nombre de termes de degré $d$
explose rapidement à mesure que $d$ augmente.
Étant donné $k$ variables, le nombre de monômes
de degré $d$ est ${k - 1 + d} \choose {k - 1}$.
Même de petits changements de degré, disons de $2$ à $3$,
augmentent considérablement la complexité de notre modèle.
Par conséquent, nous avons souvent besoin d'un outil plus précis
pour ajuster la complexité de la fonction.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
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
import optax
```

## Normes et décroissance des poids

(**Plutôt que de manipuler directement le nombre de paramètres,
la *décroissance des poids* fonctionne en restreignant les valeurs
que les paramètres peuvent prendre.**)
Plus communément appelée régularisation $\ell_2$
en dehors des cercles de l'apprentissage profond
lorsqu'elle est optimisée par la descente de gradient stochastique par mini-lots,
la décroissance des poids est peut-être la technique la plus largement utilisée
pour régulariser les modèles d'apprentissage automatique paramétriques.
La technique est motivée par l'intuition de base
que parmi toutes les fonctions $f$,
la fonction $f = 0$
(attribuant la valeur $0$ à toutes les entrées)
est en quelque sorte la plus *simple*,
et que nous pouvons mesurer la complexité
d'une fonction par la distance de ses paramètres par rapport à zéro.
Mais comment mesurer précisément
la distance entre une fonction et zéro ?
Il n'y a pas de réponse unique correcte.
En fait, des branches entières des mathématiques,
notamment certaines parties de l'analyse fonctionnelle
et de la théorie des espaces de Banach,
sont consacrées à l'étude de ces questions.

Une interprétation simple pourrait être
de mesurer la complexité d'une fonction linéaire
$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$
par une certaine norme de son vecteur de poids, par exemple, $\| \mathbf{w} \|^2$.
Rappelez-vous que nous avons introduit la norme $\ell_2$ et la norme $\ell_1$,
qui sont des cas particuliers de la norme plus générale $\ell_p$,
dans :numref:`subsec_lin-algebra-norms`.
La méthode la plus courante pour garantir un petit vecteur de poids
consiste à ajouter sa norme comme terme de pénalité
au problème de minimisation de la perte.
Ainsi, nous remplaçons notre objectif initial,
*minimiser la perte de prédiction sur les étiquettes d'entraînement*,
par un nouvel objectif,
*minimiser la somme de la perte de prédiction et du terme de pénalité*.
Désormais, si notre vecteur de poids devient trop grand,
notre algorithme d'apprentissage pourrait se concentrer
sur la minimisation de la norme des poids $\| \mathbf{w} \|^2$
plutôt que sur la minimisation de l'erreur d'entraînement.
C'est exactement ce que nous voulons.
Pour illustrer cela par du code,
nous reprenons notre exemple précédent
de :numref:`sec_linear_regression` pour la régression linéaire.
Là, notre perte était donnée par

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Rappelez-vous que $\mathbf{x}^{(i)}$ sont les caractéristiques,
$y^{(i)}$ est l'étiquette pour tout exemple de données $i$, et $(\mathbf{w}, b)$
sont respectivement les paramètres de poids et de biais.
Pour pénaliser la taille du vecteur de poids,
nous devons d'une manière ou d'une autre ajouter $\| \mathbf{w} \|^2$ à la fonction de perte,
mais comment le modèle doit-il arbitrer entre la
perte standard et cette nouvelle pénalité additive ?
En pratique, nous caractérisons cet arbitrage
via la *constante de régularisation* $\lambda$,
un hyperparamètre non négatif
que nous ajustons à l'aide des données de validation :

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$


Pour $\lambda = 0$, nous retrouvons notre fonction de perte d'origine.
Pour $\lambda > 0$, nous restreignons la taille de $\| \mathbf{w} \|$.
Nous divisons par $2$ par convention :
lorsque nous prenons la dérivée d'une fonction quadratique,
le $2$ et le $1/2$ s'annulent, garantissant que l'expression
de la mise à jour soit simple et élégante.
Le lecteur attentif pourrait se demander pourquoi nous travaillons avec le carré de la
norme et non la norme standard (c'est-à-dire la distance euclidienne).
Nous faisons cela pour des raisons de commodité de calcul.
En élevant la norme $\ell_2$ au carré, nous supprimons la racine carrée,
laissant la somme des carrés de
chaque composante du vecteur de poids.
Cela rend la dérivée de la pénalité facile à calculer :
la somme des dérivées est égale à la dérivée de la somme.


De plus, vous pourriez vous demander pourquoi nous travaillons avec la norme $\ell_2$
en premier lieu et non, par exemple, avec la norme $\ell_1$.
En fait, d'autres choix sont valables et
populaires en statistiques.
Alors que les modèles linéaires régularisés par $\ell_2$ constituent
l'algorithme classique de la *régression ridge*,
la régression linéaire régularisée par $\ell_1$
est une méthode tout aussi fondamentale en statistiques,
populairement connue sous le nom de *régression lasso*.
Une raison de travailler avec la norme $\ell_2$
est qu'elle impose une pénalité démesurée
sur les grandes composantes du vecteur de poids.
Cela oriente notre algorithme d'apprentissage
vers des modèles qui répartissent le poids uniformément
sur un plus grand nombre de caractéristiques.
En pratique, cela peut les rendre plus robustes
aux erreurs de mesure dans une seule variable.
En revanche, les pénalités $\ell_1$ conduisent à des modèles
qui concentrent les poids sur un petit ensemble de caractéristiques
en annulant les autres poids.
Cela nous donne une méthode efficace pour la *sélection de caractéristiques*,
ce qui peut être souhaitable pour d'autres raisons.
Par exemple, si notre modèle ne repose que sur quelques caractéristiques,
alors nous n'aurons peut-être pas besoin de collecter, stocker ou transmettre des données
pour les autres caractéristiques (abandonnées).

En utilisant la même notation que dans :eqref:`eq_linreg_batch_update`,
les mises à jour de la descente de gradient stochastique par mini-lots
pour la régression régularisée par $\ell_2$ sont les suivantes :

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

Comme précédemment, nous mettons à jour $\mathbf{w}$ en fonction de la quantité
dont notre estimation diffère de l'observation.
Cependant, nous réduisons également la taille de $\mathbf{w}$ vers zéro.
C'est pourquoi la méthode est parfois appelée « décroissance des poids » :
compte tenu du seul terme de pénalité,
notre algorithme d'optimisation fait *décroître*
le poids à chaque étape de l'entraînement.
Contrairement à la sélection de caractéristiques,
la décroissance des poids nous offre un mécanisme pour ajuster continuellement la complexité d'une fonction.
Des valeurs plus petites de $\lambda$ correspondent
à un $\mathbf{w}$ moins contraint,
tandis que des valeurs plus grandes de $\lambda$
contraignent $\mathbf{w}$ de manière plus considérable.
Le fait d'inclure une pénalité de biais correspondante $b^2$
peut varier selon les implémentations,
et peut varier selon les couches d'un réseau de neurones.
Souvent, nous ne régularisons pas le terme de biais.
Par ailleurs,
bien que la régularisation $\ell_2$ puisse ne pas être équivalente à la décroissance des poids pour d'autres algorithmes d'optimisation,
l'idée de régularisation par la
réduction de la taille des poids
reste vraie.

## Régression linéaire en haute dimension

Nous pouvons illustrer les avantages de la décroissance des poids
à travers un exemple synthétique simple.

Tout d'abord, nous [**générons des données comme auparavant**] :

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \textrm{ où }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

Dans ce jeu de données synthétique, notre étiquette est donnée
par une fonction linéaire sous-jacente de nos entrées,
corrompue par un bruit gaussien
de moyenne nulle et d'écart type 0,01.
À des fins d'illustration,
nous pouvons rendre les effets du surapprentissage prononcés,
en augmentant la dimensionnalité de notre problème à $d = 200$
et en travaillant avec un petit ensemble d'entraînement de seulement 20 exemples.

```{.python .input}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        if tab.selected('jax'):
            self.X = jax.random.normal(jax.random.PRNGKey(0), (n, num_inputs))
            noise = jax.random.normal(jax.random.PRNGKey(0), (n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## Implémentation à partir de zéro

Maintenant, essayons d'implémenter la décroissance des poids à partir de zéro.
Puisque la descente de gradient stochastique par mini-lots
est notre optimiseur,
il nous suffit d'ajouter la pénalité $\ell_2$ au carré
à la fonction de perte d'origine.

### (**Définition de la pénalité de norme $\ell_2$**)

La façon la plus pratique d'implémenter cette pénalité
est peut-être d'élever tous les termes au carré sur place et d'en faire la somme.

```{.python .input}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### Définition du modèle

Dans le modèle final,
la régression linéaire et la perte quadratique n'ont pas changé depuis :numref:`sec_linear_scratch`,
nous allons donc simplement définir une sous-classe de `d2l.LinearRegressionScratch`. Le seul changement ici est que notre perte inclut désormais le terme de pénalité.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
```

```{.python .input}
%%tab jax
class WeightDecayScratch(d2l.LinearRegressionScratch):
    lambd: int = 0
        
    def loss(self, params, X, y, state):
        return (super().loss(params, X, y, state) +
                self.lambd * l2_penalty(params['w']))
```

Le code suivant ajuste notre modèle sur l'ensemble d'entraînement avec 20 exemples et l'évalue sur l'ensemble de validation avec 100 exemples.

```{.python .input}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        print('L2 norm of w:', float(l2_penalty(model.w)))
    if tab.selected('jax'):
        print('L2 norm of w:',
              float(l2_penalty(trainer.state.params['w'])))
```

### [**Entraînement sans régularisation**]

Nous exécutons maintenant ce code avec `lambd = 0`,
en désactivant la décroissance des poids.
Notez que nous surapprenons gravement,
en diminuant l'erreur d'entraînement mais pas l'erreur de
validation --- un cas d'école de surapprentissage.

```{.python .input}
%%tab all
train_scratch(0)
```

### [**Utilisation de la décroissance des poids**]

Ci-dessous, nous exécutons le code avec une décroissance des poids substantielle.
Notez que l'erreur d'entraînement augmente
mais que l'erreur de validation diminue.
C'est précisément l'effet
que nous attendons de la régularisation.

```{.python .input}
%%tab all
train_scratch(3)
```

## [**Implémentation concise**]

Parce que la décroissance des poids est omniprésente
dans l'optimisation des réseaux de neurones,
le framework d'apprentissage profond la rend particulièrement pratique,
en intégrant la décroissance des poids dans l'algorithme d'optimisation lui-même
pour une utilisation facile en combinaison avec n'importe quelle fonction de perte.
De plus, cette intégration présente un avantage informatique,
permettant des astuces d'implémentation pour ajouter la décroissance des poids à l'algorithme,
sans aucun surcoût informatique supplémentaire.
Comme la partie décroissance des poids de la mise à jour
ne dépend que de la valeur actuelle de chaque paramètre,
l'optimiseur doit de toute façon toucher chaque paramètre une fois.

:begin_tab:`mxnet`
Ci-dessous, nous spécifions
l'hyperparamètre de décroissance des poids directement
via `wd` lors de l'instanciation de notre `Trainer`.
Par défaut, Gluon fait décroître à la fois
les poids et les biais simultanément.
Notez que l'hyperparamètre `wd`
sera multiplié par `wd_mult`
lors de la mise à jour des paramètres du modèle.
Ainsi, si nous réglons `wd_mult` à zéro,
le paramètre de biais $b$ ne décroîtra pas.
:end_tab:

:begin_tab:`pytorch`
Ci-dessous, nous spécifions
l'hyperparamètre de décroissance des poids directement
via `weight_decay` lors de l'instanciation de notre optimiseur.
Par défaut, PyTorch fait décroître à la fois
les poids et les biais simultanément, mais
nous pouvons configurer l'optimiseur pour gérer différents paramètres
selon différentes politiques.
Ici, nous ne définissons `weight_decay` que pour
les poids (les paramètres `net.weight`), par conséquent le
biais (le paramètre `net.bias`) ne décroîtra pas.
:end_tab:

:begin_tab:`tensorflow`
Ci-dessous, nous créons un régularisateur $\ell_2$ avec
l'hyperparamètre de décroissance des poids `wd` et nous l'appliquons aux poids de la couche
via l'argument `kernel_regularizer`.
:end_tab:

```{.python .input}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', 
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

```{.python .input}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

```{.python .input}
%%tab jax
class WeightDecay(d2l.LinearRegression):
    wd: int = 0
    
    def configure_optimizers(self):
        # Weight Decay is not available directly within optax.sgd, but
        # optax allows chaining several transformations together
        return optax.chain(optax.additive_weight_decay(self.wd),
                           optax.sgd(self.lr))
```

[**Le graphique ressemble à celui obtenu lorsque
nous avons implémenté la décroissance des poids à partir de zéro**].
Cependant, cette version s'exécute plus rapidement
et est plus facile à implémenter,
des avantages qui deviendront plus
marqués à mesure que vous aborderez des problèmes plus importants
et que ce travail deviendra plus routinier.

```{.python .input}
%%tab all
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

if tab.selected('jax'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b(trainer.state)[0])))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

Jusqu'à présent, nous avons abordé une notion de
ce qui constitue une fonction linéaire simple.
Cependant, même pour des fonctions non linéaires simples, la situation peut être beaucoup plus complexe. Pour voir cela, le concept d'[espace de Hilbert à noyau reproduisant (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)
permet d'appliquer les outils introduits
pour les fonctions linéaires dans un contexte non linéaire.
Malheureusement, les algorithmes basés sur le RKHS
ont tendance à mal passer à l'échelle pour des données volumineuses et de haute dimension.
Dans ce livre, nous adopterons souvent l'heuristique commune
selon laquelle la décroissance des poids est appliquée
à toutes les couches d'un réseau profond.

## Résumé

La régularisation est une méthode courante pour traiter le surapprentissage. Les techniques de régularisation classiques ajoutent un terme de pénalité à la fonction de perte (pendant l'entraînement) pour réduire la complexité du modèle appris.
Un choix particulier pour garder le modèle simple consiste à utiliser une pénalité $\ell_2$. Cela conduit à une décroissance des poids dans les étapes de mise à jour de l'algorithme de descente de gradient stochastique par mini-lots.
En pratique, la fonctionnalité de décroissance des poids est fournie dans les optimiseurs des frameworks d'apprentissage profond.
Différents ensembles de paramètres peuvent avoir des comportements de mise à jour différents au sein de la même boucle d'entraînement.



## Exercices

1. Expérimentez avec la valeur de $\lambda$ dans le problème d'estimation de cette section. Tracez la précision d'entraînement et de validation en fonction de $\lambda$. Qu'observez-vous ?
1. Utilisez un ensemble de validation pour trouver la valeur optimale de $\lambda$. Est-ce vraiment la valeur optimale ? Est-ce important ?
1. À quoi ressembleraient les équations de mise à jour si, au lieu de $\|\mathbf{w}\|^2$, nous utilisions $\sum_i |w_i|$ comme pénalité de choix (régularisation $\ell_1$) ?
1. Nous savons que $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Pouvez-vous trouver une équation similaire pour les matrices (voir la norme de Frobenius dans :numref:`subsec_lin-algebra-norms`) ?
1. Revoyez la relation entre l'erreur d'entraînement et l'erreur de généralisation. En plus de la décroissance des poids, de l'augmentation de l'entraînement et de l'utilisation d'un modèle de complexité appropriée, quels autres moyens pourraient nous aider à faire face au surapprentissage ?
1. En statistiques bayésiennes, nous utilisons le produit de la priorité et de la vraisemblance pour arriver à une postériorité via $P(w \mid x) \propto P(x \mid w) P(w)$. Comment pouvez-vous identifier $P(w)$ avec la régularisation ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17979)
:end_tab:
