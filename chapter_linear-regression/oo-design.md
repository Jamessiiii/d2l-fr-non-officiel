```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Conception orientée objet pour l'implémentation
:label:`sec_oo-design`

Dans notre introduction à la régression linéaire,
nous avons passé en revue divers composants,
notamment
les données, le modèle, la fonction de perte,
et l'algorithme d'optimisation.
En effet,
la régression linéaire est
l'un des modèles d'apprentissage automatique les plus simples.
Son entraînement,
cependant, utilise un grand nombre des mêmes composants que ceux requis par les autres modèles de ce livre.
Par conséquent,
avant de plonger dans les détails de l'implémentation,
il est utile
de concevoir certaines des API
que nous utiliserons tout au long de l'ouvrage.
En traitant les composants du deep learning
comme des objets,
nous pouvons commencer par
définir des classes pour ces objets
et leurs interactions.
Cette conception orientée objet
pour l'implémentation
simplifiera grandement
la présentation et vous pourriez même vouloir l'utiliser dans vos propres projets.


Inspirés par des bibliothèques open-source telles que [PyTorch Lightning](https://www.pytorchlightning.ai/),
à haut niveau,
nous souhaitons avoir trois classes :
(i) `Module` contient les modèles, les pertes et les méthodes d'optimisation ;
(ii) `DataModule` fournit des chargeurs de données pour l'entraînement et la validation ;
(iii) les deux classes sont combinées à l'aide de la classe `Trainer`, qui nous permet d'entraîner des modèles sur une variété de plateformes matérielles.
La plupart du code de ce livre adapte `Module` et `DataModule`. Nous n'aborderons la classe `Trainer` que lorsque nous discuterons des GPU, des CPU, de l'entraînement parallèle et des algorithmes d'optimisation.

```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from dataclasses import field
from d2l import jax as d2l
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
import numpy as np
import jax
import time
from typing import Any
```

## Utilitaires
:label:`oo-design-utilities`

Nous avons besoin de quelques utilitaires pour simplifier la programmation orientée objet dans les notebooks Jupyter. L'un des défis est que les définitions de classes ont tendance à être d'assez longs blocs de code. La lisibilité des notebooks exige des fragments de code courts, entrecoupés d'explications, une exigence incompatible avec le style de programmation courant pour les bibliothèques Python. La première fonction utilitaire nous permet d'enregistrer des fonctions en tant que méthodes dans une classe *après* que la classe a été créée. En fait, nous pouvons le faire *même après* avoir créé des instances de la classe ! Cela nous permet de diviser l'implémentation d'une classe en plusieurs blocs de code.

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

Jetons un coup d'œil rapide à la façon de l'utiliser. Nous prévoyons d'implémenter une classe `A` avec une méthode `do`. Au lieu d'avoir le code pour `A` et `do` dans le même bloc de code, nous pouvons d'abord déclarer la classe `A` et créer une instance `a`.

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```

Ensuite, nous définissons la méthode `do` comme nous le ferions normalement, mais pas dans la portée de la classe `A`. Au lieu de cela, nous décorons cette méthode par `add_to_class` avec la classe `A` comme argument. Ce faisant, la méthode est capable d'accéder aux variables membres de `A` exactement comme nous nous y attendrions si elle avait été incluse dans la définition de `A`. Voyons ce qui se passe quand nous l'invoquons pour l'instance `a`.

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

La seconde est une classe utilitaire qui enregistre tous les arguments de la méthode `__init__` d'une classe en tant qu'attributs de classe. Cela nous permet d'étendre implicitement les signatures d'appel des constructeurs sans code supplémentaire.

```{.python .input}
%%tab all
class HyperParameters:  #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

Nous reportons son implémentation dans le :numref:`sec_utils`. Pour l'utiliser, nous définissons notre classe qui hérite de `HyperParameters` et appelle `save_hyperparameters` dans la méthode `__init__`.

```{.python .input}
%%tab all
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

Le dernier utilitaire nous permet de tracer la progression de l'expérience de manière interactive pendant qu'elle se déroule. Par déférence pour le bien plus puissant (et complexe) [TensorBoard](https://www.tensorflow.org/tensorboard), nous le nommons `ProgressBoard`. L'implémentation est reportée au :numref:`sec_utils`. Pour l'instant, voyons-le simplement en action.

La méthode `draw` trace un point `(x, y)` dans la figure, avec le `label` spécifié dans la légende. Le paramètre optionnel `every_n` lisse la ligne en n'affichant que $1/n$ points dans la figure. Leurs valeurs sont la moyenne des $n$ points voisins de la figure originale.

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

Dans l'exemple suivant, nous dessinons `sin` et `cos` avec un lissage différent. Si vous exécutez ce bloc de code, vous verrez les lignes s'allonger en animation.

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## Modèles
:label:`subsec_oo-design-models`

La classe `Module` est la classe de base de tous les modèles que nous implémenterons. Nous avons besoin au minimum de trois méthodes. La première, `__init__`, stocke les paramètres apprenables, la méthode `training_step` accepte un lot de données pour renvoyer la valeur de la perte, et enfin, `configure_optimizers` renvoie la méthode d'optimisation, ou une liste d'entre elles, qui est utilisée pour mettre à jour les paramètres apprenables. En option, nous pouvons définir `validation_step` pour rapporter les mesures d'évaluation.
Parfois, nous plaçons le code de calcul de la sortie dans une méthode `forward` séparée pour le rendre plus réutilisable.

:begin_tab:`jax`
Avec l'introduction des [dataclasses](https://docs.python.org/3/library/dataclasses.html)
dans Python 3.7, les classes décorées avec `@dataclass` ajoutent automatiquement des méthodes magiques telles que `__init__` et `__repr__`. Les variables membres sont définies à l'aide d'annotations de type. Tous les modules Flax sont des dataclasses Python 3.7.
:end_tab:

```{.python .input}
%%tab pytorch
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input}
%%tab mxnet, tensorflow, jax
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    if tab.selected('mxnet', 'tensorflow'):
        def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
            super().__init__()
            self.save_hyperparameters()
            self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    if tab.selected('jax'):
        # No need for save_hyperparam when using Python dataclass
        plot_train_per_epoch: int = field(default=2, init=False)
        plot_valid_per_epoch: int = field(default=1, init=False)
        # Use default_factory to make sure new plots are generated on each run
        board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                     init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    if tab.selected('mxnet', 'tensorflow'):
        def forward(self, X):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, **kwargs):
            if kwargs and "training" in kwargs:
                self.training = kwargs['training']
            return self.forward(X, *args)

    if tab.selected('jax'):
        # JAX & Flax do not have a forward-method-like syntax. Flax uses setup
        # and built-in __call__ magic methods for forward pass. Adding here
        # for consistency
        def forward(self, X, *args, **kwargs):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X, *args, **kwargs)

        def __call__(self, X, *args, **kwargs):
            return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key, every_n=int(n))
        if tab.selected('jax'):
            self.board.draw(x, d2l.to(value, d2l.cpu()),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    if tab.selected('mxnet', 'tensorflow'):
        def training_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=True)
            return l

        def validation_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=False)

    if tab.selected('jax'):
        def training_step(self, params, batch, state):
            l, grads = jax.value_and_grad(self.loss)(params, batch[:-1],
                                                     batch[-1], state)
            self.plot("loss", l, train=True)
            return l, grads

        def validation_step(self, params, batch, state):
            l = self.loss(params, batch[:-1], batch[-1], state)
            self.plot('loss', l, train=False)
        
        def apply_init(self, dummy_input, key):
            """To be defined later in :numref:`sec_lazy_init`"""
            raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
Vous remarquerez peut-être que `Module` est une sous-classe de `nn.Block`, la classe de base des réseaux neuronaux dans Gluon.
Elle offre des fonctionnalités pratiques pour la gestion des réseaux neuronaux. Par exemple, si nous définissons une méthode `forward`, telle que `forward(self, X)`, alors pour une instance `a`, nous pouvons invoquer cette méthode par `a(X)`. Cela fonctionne car elle appelle la méthode `forward` dans la méthode intégrée `__call__`. Vous trouverez plus de détails et d'exemples sur `nn.Block` dans le :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`pytorch`
Vous remarquerez peut-être que `Module` est une sous-classe de `nn.Module`, la classe de base des réseaux neuronaux dans PyTorch.
Elle offre des fonctionnalités pratiques pour la gestion des réseaux neuronaux. Par exemple, si nous définissons une méthode `forward`, telle que `forward(self, X)`, alors pour une instance `a`, nous pouvons invoquer cette méthode par `a(X)`. Cela fonctionne car elle appelle la méthode `forward` dans la méthode intégrée `__call__`. Vous trouverez plus de détails et d'exemples sur `nn.Module` dans le :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`tensorflow`
Vous remarquerez peut-être que `Module` est une sous-classe de `tf.keras.Model`, la classe de base des réseaux neuronaux dans TensorFlow.
Elle offre des fonctionnalités pratiques pour la gestion des réseaux neuronaux. Par exemple, elle invoque la méthode `call` dans la méthode intégrée `__call__`. Ici, nous redirigeons `call` vers la méthode `forward`, en enregistrant ses arguments en tant qu'attribut de classe. Nous faisons cela pour rendre notre code plus proche des implémentations d'autres frameworks.
:end_tab:

:begin_tab:`jax`
Vous remarquerez peut-être que `Module` est une sous-classe de `linen.Module`, la classe de base des réseaux neuronaux dans Flax.
Elle offre des fonctionnalités pratiques pour la gestion des réseaux neuronaux. Par exemple, elle gère les paramètres du modèle, fournit le décorateur `nn.compact` pour simplifier le code, invoque la méthode `__call__` entre autres choses.
Ici, nous redirigeons également `__call__` vers la méthode `forward`. Nous faisons cela pour rendre notre code plus proche des implémentations d'autres frameworks.
:end_tab:

## Données
:label:`oo-design-data`

La classe `DataModule` est la classe de base pour les données. Très fréquemment, la méthode `__init__` est utilisée pour préparer les données. Cela inclut le téléchargement et le prétraitement si nécessaire. La méthode `train_dataloader` renvoie le chargeur de données pour le jeu de données d'entraînement. Un chargeur de données est un générateur (Python) qui produit un lot de données à chaque fois qu'il est utilisé. Ce lot est ensuite transmis à la méthode `training_step` de `Module` pour calculer la perte. Il existe une méthode optionnelle `val_dataloader` pour renvoyer le chargeur du jeu de données de validation. Elle se comporte de la même manière, si ce n'est qu'elle produit des lots de données pour la méthode `validation_step` de `Module`.

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow', 'jax'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## Entraînement
:label:`oo-design-training`

:begin_tab:`pytorch, mxnet, tensorflow`
La classe `Trainer` entraîne les paramètres apprenables de la classe `Module` avec les données spécifiées dans `DataModule`. La méthode clé est `fit`, qui accepte deux arguments : `model`, une instance de `Module`, et `data`, une instance de `DataModule`. Elle itère ensuite sur l'ensemble du jeu de données `max_epochs` fois pour entraîner le modèle. Comme précédemment, nous reporterons l'implémentation de cette méthode à des chapitres ultérieurs.
:end_tab:

:begin_tab:`jax`
La classe `Trainer` entraîne les paramètres apprenables `params` avec les données spécifiées dans `DataModule`. La méthode clé est `fit`, qui accepte trois arguments : `model`, une instance de `Module`, `data`, une instance de `DataModule`, et `key`, un `PRNGKeyArray` de JAX. Nous rendons ici l'argument `key` optionnel pour simplifier l'interface, mais il est recommandé de toujours passer et initialiser les paramètres du modèle avec une clé racine dans JAX et Flax. Elle itère ensuite sur l'ensemble du jeu de données `max_epochs` fois pour entraîner le modèle. Comme précédemment, nous reporterons l'implémentation de cette méthode à des chapitres ultérieurs.
:end_tab:

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        def fit(self, model, data):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    if tab.selected('jax'):
        def fit(self, model, data, key=None):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()

            if key is None:
                root_key = d2l.get_key()
            else:
                root_key = key
            params_key, dropout_key = jax.random.split(root_key)
            key = {'params': params_key, 'dropout': dropout_key}

            dummy_input = next(iter(self.train_dataloader))[:-1]
            variables = model.apply_init(dummy_input, key=key)
            params = variables['params']

            if 'batch_stats' in variables.keys():
                # Here batch_stats will be used later (e.g., for batch norm)
                batch_stats = variables['batch_stats']
            else:
                batch_stats = {}

            # Flax uses optax under the hood for a single state obj TrainState.
            # More will be discussed later in the dropout and batch
            # normalization section
            class TrainState(train_state.TrainState):
                batch_stats: Any
                dropout_rng: jax.random.PRNGKeyArray

            self.state = TrainState.create(apply_fn=model.apply,
                                           params=params,
                                           batch_stats=batch_stats,
                                           dropout_rng=dropout_key,
                                           tx=model.configure_optimizers())
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## Résumé

Pour mettre en évidence la conception orientée objet
de nos futures implémentations de deep learning,
les classes ci-dessus montrent simplement comment leurs objets
stockent les données et interagissent entre eux.
Nous continuerons à enrichir les implémentations de ces classes,
par exemple via `@add_to_class`,
dans le reste du livre.
De plus,
ces classes entièrement implémentées
sont enregistrées dans la [bibliothèque D2L](https://github.com/d2l-ai/d2l-en/tree/master/d2l),
une *boîte à outils légère* qui facilite la modélisation structurée pour le deep learning.
En particulier, elle facilite la réutilisation de nombreux composants entre les projets sans avoir à changer grand-chose. Par exemple, nous pouvons remplacer uniquement l'optimiseur, uniquement le modèle, uniquement le jeu de données, etc. ;
ce degré de modularité porte ses fruits tout au long du livre en termes de concision et de simplicité (c'est pourquoi nous l'avons ajouté) et il peut en faire de même pour vos propres projets.


## Exercices

1. Localisez les implémentations complètes des classes ci-dessus qui sont enregistrées dans la [bibliothèque D2L](https://github.com/d2l-ai/d2l-en/tree/master/d2l). Nous vous recommandons vivement d'examiner l'implémentation en détail une fois que vous aurez acquis un peu plus de familiarité avec la modélisation en deep learning.
1. Supprimez l'instruction `save_hyperparameters` dans la classe `B`. Pouvez-vous toujours afficher `self.a` et `self.b` ? Facultatif : si vous avez plongé dans l'implémentation complète de la classe `HyperParameters`, pouvez-vous expliquer pourquoi ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6645)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6646)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6647)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17974)
:end_tab:
