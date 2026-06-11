```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Implémentation concise de la régression linéaire
:label:`sec_linear_concise`

L'apprentissage profond a connu une sorte d'explosion cambrienne au cours de la dernière décennie.
Le nombre impressionnant de techniques, d'applications et d'algorithmes dépasse de loin les progrès des décennies précédentes. 
Cela est dû à une combinaison fortuite de multiples facteurs, dont l'un est la puissance des outils gratuits offerts par un certain nombre de frameworks d'apprentissage profond en open-source.
Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`,
DistBelief :cite:`Dean.Corrado.Monga.ea.2012`,
et Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`
représentent sans doute la première génération de tels modèles ayant connu une adoption généralisée.
Contrairement aux travaux plus anciens (fondateurs) comme SN2 (Simulateur Neuristique) :cite:`Bottou.Le-Cun.1988`, qui offrait une expérience de programmation de type Lisp, les frameworks modernes proposent la différenciation automatique et la commodité de Python.
Ces frameworks nous permettent d'automatiser et de modulariser le travail répétitif de mise en œuvre des algorithmes d'apprentissage basés sur le gradient.

Dans la :numref:`sec_linear_scratch`, nous nous sommes appuyés uniquement sur (i) des tenseurs pour le stockage des données et l'algèbre linéaire ; et (ii) la différenciation automatique pour le calcul des gradients.
En pratique, comme les itérateurs de données, les fonctions de perte, les optimiseurs et les couches de réseaux neuronaux sont très courants, les bibliothèques modernes implémentent également ces composants pour nous.
Dans cette section, (**nous allons vous montrer comment implémenter le modèle de régression linéaire**) de la :numref:`sec_linear_scratch` (**de manière concise en utilisant les API de haut niveau**) des frameworks d'apprentissage profond.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## Définition du modèle

Lorsque nous avons implémenté la régression linéaire à partir de zéro dans la :numref:`sec_linear_scratch`, nous avons défini les paramètres de notre modèle explicitement et codé les calculs pour produire des sorties en utilisant des opérations d'algèbre linéaire de base.
Vous *devriez* savoir comment faire cela.
Mais une fois que vos modèles deviennent plus complexes, et que vous devez le faire presque tous les jours, vous serez heureux de recevoir de l'aide.
La situation est similaire à celle de la création de votre propre blog à partir de zéro.
Le faire une ou deux fois est gratifiant et instructif, mais vous seriez un piètre développeur web si vous passiez un mois à réinventer la roue.

Pour les opérations standard, nous pouvons [**utiliser les couches prédéfinies d'un framework,**] ce qui nous permet de nous concentrer sur les couches utilisées pour construire le modèle plutôt que de nous soucier de leur implémentation.
Rappelez-vous de l'architecture d'un réseau à une seule couche telle que décrite dans la :numref:`fig_single_neuron`.
La couche est appelée *complètement connectée* (ou *fully connected*), car chacune de ses entrées est connectée à chacune de ses sorties au moyen d'une multiplication matrice-vecteur.

:begin_tab:`mxnet`
Dans Gluon, la couche complètement connectée est définie dans la classe `Dense`.
Comme nous voulons seulement générer une seule sortie scalaire, nous fixons ce nombre à 1.
Il convient de noter que, par commodité, Gluon ne nous oblige pas à spécifier la forme d'entrée (input shape) pour chaque couche.
Par conséquent, nous n'avons pas besoin de dire à Gluon combien d'entrées vont dans cette couche linéaire.
Lorsque nous passerons pour la première fois des données à travers notre modèle, par exemple lors de l'exécution ultérieure de `net(X)`, Gluon déduira automatiquement le nombre d'entrées pour chaque couche et instanciera ainsi le modèle correct.
Nous décrirons comment cela fonctionne plus en détail ultérieurement.
:end_tab:

:begin_tab:`pytorch`
Dans PyTorch, la couche complètement connectée est définie dans les classes `Linear` et `LazyLinear` (disponibles depuis la version 1.8.0).
Cette dernière permet aux utilisateurs de spécifier *uniquement* la dimension de sortie, tandis que la première demande en plus combien d'entrées vont dans cette couche.
Spécifier les formes d'entrée est peu pratique et peut nécessiter des calculs non triviaux (comme dans les couches convolutionnelles).
Ainsi, pour plus de simplicité, nous utiliserons ces couches "lazy" (paresseuses) chaque fois que nous le pourrons. 
:end_tab:

:begin_tab:`tensorflow`
Dans Keras, la couche complètement connectée est définie dans la classe `Dense`.
Comme nous voulons seulement générer une seule sortie scalaire, nous fixons ce nombre à 1.
Il convient de noter que, par commodité, Keras ne nous oblige pas à spécifier la forme d'entrée pour chaque couche.
Nous n'avons pas besoin de dire à Keras combien d'entrées vont dans cette couche linéaire.
Lorsque nous essaierons pour la première fois de passer des données à travers notre modèle, par exemple lors de l'exécution ultérieure de `net(X)`, Keras déduira automatiquement le nombre d'entrées pour chaque couche.
Nous décrirons comment cela fonctionne plus en détail ultérieurement.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        if tab.selected('tensorflow'):
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        if tab.selected('pytorch'):
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
```

```{.python .input}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    lr: float

    def setup(self):
        self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))
```

Dans la méthode `forward`, nous invoquons simplement la méthode intégrée `__call__` des couches prédéfinies pour calculer les sorties.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

## Définition de la fonction de perte

:begin_tab:`mxnet`
Le module `loss` définit de nombreuses fonctions de perte utiles.
Pour plus de rapidité et de commodité, nous renonçons à implémenter la nôtre et choisissons à la place la fonction intégrée `loss.L2Loss`.
Étant donné que la perte (`loss`) qu'elle renvoie est l'erreur quadratique pour chaque exemple, nous utilisons `mean` pour faire la moyenne de la perte sur le mini-lot.
:end_tab:

:begin_tab:`pytorch`
[**La classe `MSELoss` calcule l'erreur quadratique moyenne (sans le facteur $1/2$ dans :eqref:`eq_mse`).**]
Par défaut, `MSELoss` renvoie la perte moyenne sur les exemples.
C'est plus rapide (et plus facile à utiliser) que d'implémenter la nôtre.
:end_tab:

:begin_tab:`tensorflow`
La classe `MeanSquaredError` calcule l'erreur quadratique moyenne (sans le facteur $1/2$ dans :eqref:`eq_mse`).
Par défaut, elle renvoie la perte moyenne sur les exemples.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)
    return d2l.reduce_mean(optax.l2_loss(y_hat, y))
```

## Définition de l'algorithme d'optimisation

:begin_tab:`mxnet`
Le SGD par mini-lot (Minibatch SGD) est un outil standard pour optimiser les réseaux de neurones et Gluon le prend donc en charge, ainsi qu'un certain nombre de variantes de cet algorithme, via sa classe `Trainer`.
Notez que la classe `Trainer` de Gluon représente l'algorithme d'optimisation, tandis que la classe `Trainer` que nous avons créée dans la :numref:`sec_oo-design` contient la méthode d'entraînement, c'est-à-dire l'appel répété de l'optimiseur pour mettre à jour les paramètres du modèle.
Lorsque nous instancions `Trainer`, nous spécifions les paramètres à optimiser, accessibles depuis notre modèle `net` via `net.collect_params()`, l'algorithme d'optimisation que nous souhaitons utiliser (`sgd`), et un dictionnaire d'hyperparamètres requis par notre algorithme d'optimisation.
:end_tab:

:begin_tab:`pytorch`
Le SGD par mini-lot (Minibatch SGD) est un outil standard pour optimiser les réseaux de neurones et PyTorch le prend donc en charge, ainsi qu'un certain nombre de variantes de cet algorithme, dans le module `optim`.
Lorsque nous (**instancions une instance de `SGD`,**) nous spécifions les paramètres à optimiser, accessibles depuis notre modèle via `self.parameters()`, et le taux d'apprentissage (`self.lr`) requis par notre algorithme d'optimisation.
:end_tab:

:begin_tab:`tensorflow`
Le SGD par mini-lot (Minibatch SGD) est un outil standard pour optimiser les réseaux de neurones et Keras le prend donc en charge, ainsi qu'un certain nombre de variantes de cet algorithme, dans le module `optimizers`.
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
    if tab.selected('jax'):
        return optax.sgd(self.lr)
```

## Entraînement

Vous avez peut-être remarqué que l'expression de notre modèle via les API de haut niveau d'un framework d'apprentissage profond nécessite moins de lignes de code.
Nous n'avons pas eu à allouer les paramètres individuellement, à définir notre fonction de perte ou à implémenter le SGD par mini-lot.
Une fois que nous commencerons à travailler avec des modèles beaucoup plus complexes, les avantages de l'API de haut niveau augmenteront considérablement.

Maintenant que nous avons toutes les pièces de base en place, [**la boucle d'entraînement elle-même est la même que celle que nous avons implémentée à partir de zéro.**]
Nous appelons donc simplement la méthode `fit` (introduite dans la :numref:`oo-design-training`), qui s'appuie sur l'implémentation de la méthode `fit_epoch` de la :numref:`sec_linear_scratch`, pour entraîner notre modèle.

```{.python .input}
%%tab all
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Ci-dessous, nous [**comparons les paramètres du modèle appris par l'entraînement sur des données finies et les paramètres réels**] qui ont généré notre jeu de données.
Pour accéder aux paramètres, nous accédons aux poids et au biais de la couche dont nous avons besoin.
Comme dans notre implémentation à partir de zéro, notez que nos paramètres estimés sont proches de leurs homologues réels.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self, state):
    net = state.params['net']
    return net['kernel'], net['bias']

w, b = model.get_w_b(trainer.state)
```

```{.python .input}
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## Résumé

Cette section contient la première implémentation d'un réseau profond (dans ce livre) pour tirer parti des commodités offertes par les frameworks modernes d'apprentissage profond, tels que MXNet :cite:`Chen.Li.Li.ea.2015`, JAX :cite:`Frostig.Johnson.Leary.2018`, PyTorch :cite:`Paszke.Gross.Massa.ea.2019`, et TensorFlow :cite:`Abadi.Barham.Chen.ea.2016`.
Nous avons utilisé les paramètres par défaut du framework pour charger les données, définir une couche, une fonction de perte, un optimiseur et une boucle d'entraînement.
Chaque fois que le framework fournit toutes les fonctionnalités nécessaires, c'est généralement une bonne idée de les utiliser, car les implémentations de ces composants dans les bibliothèques ont tendance à être fortement optimisées pour la performance et correctement testées pour la fiabilité.
En même temps, essayez de ne pas oublier que ces modules *peuvent* être implémentés directement.
Ceci est particulièrement important pour les chercheurs en herbe qui souhaitent se situer à la pointe du développement de modèles, où vous inventerez de nouveaux composants qui ne peuvent pas encore exister dans une bibliothèque actuelle.

:begin_tab:`mxnet`
Dans Gluon, le module `data` fournit des outils pour le traitement des données, le module `nn` définit un grand nombre de couches de réseaux neuronaux, et le module `loss` définit de nombreuses fonctions de perte courantes.
De plus, `initializer` donne accès à de nombreux choix pour l'initialisation des paramètres.
Par commodité pour l'utilisateur, la dimensionnalité et le stockage sont automatiquement déduits.
Une conséquence de cette initialisation paresseuse (lazy initialization) est que vous ne devez pas tenter d'accéder aux paramètres avant qu'ils n'aient été instanciés (et initialisés).
:end_tab:

:begin_tab:`pytorch`
Dans PyTorch, le module `data` fournit des outils pour le traitement des données, le module `nn` définit un grand nombre de couches de réseaux neuronaux et de fonctions de perte courantes.
Nous pouvons initialiser les paramètres en remplaçant leurs valeurs par des méthodes se terminant par `_`.
Notez que nous devons spécifier les dimensions d'entrée du réseau.
Bien que cela soit trivial pour l'instant, cela peut avoir des répercussions importantes lorsque nous voulons concevoir des réseaux complexes avec de nombreuses couches.
Des considérations attentives sur la manière de paramétrer ces réseaux sont nécessaires pour permettre la portabilité.
:end_tab:

:begin_tab:`tensorflow`
Dans TensorFlow, le module `data` fournit des outils pour le traitement des données, le module `keras` définit un grand nombre de couches de réseaux neuronaux et de fonctions de perte courantes.
De plus, le module `initializers` fournit diverses méthodes pour l'initialisation des paramètres du modèle.
La dimensionnalité et le stockage pour les réseaux sont automatiquement déduits (mais faites attention à ne pas tenter d'accéder aux paramètres avant qu'ils n'aient été initialisés).
:end_tab:

## Exercices

1. Comment devriez-vous modifier le taux d'apprentissage si vous remplacez la perte agrégée sur le mini-lot par une moyenne de la perte sur le mini-lot ?
1. Consultez la documentation du framework pour voir quelles fonctions de perte sont proposées. En particulier, remplacez la perte quadratique par la fonction de perte robuste de Huber. C'est-à-dire, utilisez la fonction de perte
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \textrm{ si } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \textrm{ sinon}\end{cases}$$
1. Comment accède-t-index au gradient des poids du modèle ?
1. Quel est l'effet sur la solution si vous modifiez le taux d'apprentissage et le nombre d'époques ? Est-ce qu'elle continue à s'améliorer ?
1. Comment la solution change-t-elle à mesure que vous faites varier la quantité de données générées ?
    1. Tracez l'erreur d'estimation pour $\hat{\mathbf{w}} - \mathbf{w}$ et $\hat{b} - b$ en fonction de la quantité de données. Conseil : augmentez la quantité de données de manière logarithmique plutôt que linéaire, c'est-à-dire 5, 10, 20, 50, ..., 10 000 plutôt que 1 000, 2 000, ..., 10 000.
    2. Pourquoi la suggestion du conseil est-elle appropriée ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17977)
:end_tab:
