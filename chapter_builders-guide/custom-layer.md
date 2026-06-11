```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Couches personnalisées

L'un des facteurs du succès du deep learning réside dans la disponibilité d'une large gamme de couches qui peuvent être combinées de manière créative pour concevoir des architectures adaptées à une grande variété de tâches.
Par exemple, les chercheurs ont inventé des couches spécifiquement pour manipuler des images, du texte, boucler sur des données séquentielles et effectuer de la programmation dynamique.
Tôt ou tard, vous aurez besoin d'une couche qui n'existe pas encore dans le framework de deep learning.
Dans ces cas, vous devez construire une couche personnalisée.
Dans cette section, nous vous montrons comment faire.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## (**Couches sans paramètres**)

Pour commencer, nous construisons une couche personnalisée qui n'a pas de paramètres propres.
Cela devrait vous sembler familier si vous vous rappelez notre introduction aux modules dans :numref:`sec_model_construction`.
La classe `CenteredLayer` suivante soustrait simplement la moyenne de son entrée.
Pour la construire, il nous suffit d'hériter de la classe de couche de base et d'implémenter la fonction de propagation avant.

```{.python .input}
%%tab mxnet
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab tensorflow
class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return X - tf.reduce_mean(X)
```

```{.python .input}
%%tab jax
class CenteredLayer(nn.Module):
    def __call__(self, X):
        return X - X.mean()
```

Vérifions que notre couche fonctionne comme prévu en y faisant passer quelques données.

```{.python .input}
%%tab all
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

Nous pouvons maintenant [**incorporer notre couche en tant que composant dans la construction de modèles plus complexes.**]

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

```{.python .input}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(128), CenteredLayer()])
```

À titre de vérification supplémentaire, nous pouvons envoyer des données aléatoires à travers le réseau et vérifier que la moyenne est bien de 0.
Comme nous manipulons des nombres à virgule flottante, nous pouvons toujours voir un très petit nombre non nul dû à la quantification.

:begin_tab:`jax`
Ici, nous utilisons la méthode `init_with_output` qui renvoie à la fois la sortie du réseau ainsi que les paramètres. Dans ce cas, nous nous concentrons uniquement sur la sortie.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet
Y = net(d2l.rand(4, 8))
Y.mean()
```

```{.python .input}
%%tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input}
%%tab jax
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (4, 8)))
Y.mean()
```

## [**Couches avec paramètres**]

Maintenant que nous savons comment définir des couches simples, passons à la définition de couches avec des paramètres qui peuvent être ajustés par l'entraînement.
Nous pouvons utiliser des fonctions intégrées pour créer des paramètres, ce qui fournit des fonctionnalités de base.
En particulier, elles régissent l'accès, l'initialisation, le partage, la sauvegarde et le chargement des paramètres du modèle.
De cette façon, entre autres avantages, nous n'aurons pas besoin d'écrire des routines de sérialisation personnalisées pour chaque couche personnalisée.

Implémentons maintenant notre propre version de la couche entièrement connectée.
Rappelons que cette couche nécessite deux paramètres, l'un pour représenter le poids et l'autre pour le biais.
Dans cette implémentation, nous intégrons l'activation ReLU par défaut.
Cette couche nécessite deux arguments d'entrée : `in_units` et `units`, qui désignent respectivement le nombre d'entrées et de sorties.

```{.python .input}
%%tab mxnet
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
%%tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
%%tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

```{.python .input}
%%tab jax
class MyDense(nn.Module):
    in_units: int
    units: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.normal(stddev=1),
                                 (self.in_units, self.units))
        self.bias = self.param('bias', nn.initializers.zeros, self.units)

    def __call__(self, X):
        linear = jnp.matmul(X, self.weight) + self.bias
        return nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow, jax`
Ensuite, nous instancions la classe `MyDense` et accédons à ses paramètres de modèle.
:end_tab:

:begin_tab:`pytorch`
Ensuite, nous instancions la classe `MyLinear` et accédons à ses paramètres de modèle.
:end_tab:

```{.python .input}
%%tab mxnet
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
%%tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
%%tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

```{.python .input}
%%tab jax
dense = MyDense(5, 3)
params = dense.init(d2l.get_key(), jnp.zeros((3, 5)))
params
```

Nous pouvons [**effectuer directement des calculs de propagation avant en utilisant des couches personnalisées.**]

```{.python .input}
%%tab mxnet
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
%%tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
%%tab tensorflow
dense(tf.random.uniform((2, 5)))
```

```{.python .input}
%%tab jax
dense.apply(params, jax.random.uniform(d2l.get_key(),
                                       (2, 5)))
```

Nous pouvons également (**construire des modèles à l'aide de couches personnalisées.**)
Une fois que nous avons cela, nous pouvons l'utiliser tout comme la couche entièrement connectée intégrée.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

```{.python .input}
%%tab jax
net = nn.Sequential([MyDense(64, 8), MyDense(8, 1)])
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (2, 64)))
Y
```

## Résumé

Nous pouvons concevoir des couches personnalisées via la classe de couche de base. Cela nous permet de définir de nouvelles couches flexibles qui se comportent différemment de toutes les couches existantes dans la bibliothèque.
Une fois définies, les couches personnalisées peuvent être invoquées dans des contextes et des architectures arbitraires.
Les couches peuvent avoir des paramètres locaux, qui peuvent être créés par des fonctions intégrées.


## Exercices

1. Concevez une couche qui prend une entrée et calcule une réduction de tenseur, c'est-à-dire qu'elle renvoie $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Concevez une couche qui renvoie la première moitié des coefficients de Fourier des données.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/279)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17993)
:end_tab:
