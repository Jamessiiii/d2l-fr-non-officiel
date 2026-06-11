```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Gestion des paramètres

Une fois que nous avons choisi une architecture
et défini nos hyperparamètres,
nous passons à la boucle d'entraînement,
où notre objectif est de trouver des valeurs de paramètres
qui minimisent notre fonction de perte.
Après l'entraînement, nous aurons besoin de ces paramètres
afin de faire des prédictions futures.
De plus, nous souhaiterons parfois
extraire les paramètres
peut-être pour les réutiliser dans un autre contexte,
pour sauvegarder notre modèle sur disque afin qu'il
puisse être exécuté dans d'autres logiciels,
ou pour les examiner dans l'espoir d'acquérir
une compréhension scientifique.

La plupart du temps, nous serons en mesure
d'ignorer les détails pratiques
de la façon dont les paramètres sont déclarés
et manipulés, en nous appuyant sur les frameworks de deep learning
pour faire le gros du travail.
Cependant, lorsque nous nous éloignons des
architectures empilées avec des couches standard,
nous devrons parfois entrer dans le vif du sujet
en déclarant et en manipulant les paramètres.
Dans cette section, nous couvrons les points suivants :

* Accéder aux paramètres pour le débogage, les diagnostics et les visualisations.
* Partager les paramètres entre différents composants du modèle.

```{.python .input}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

(**Nous commençons par nous concentrer sur un MLP avec une couche cachée.**)

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])

X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## [**Accès aux paramètres**]
:label:`subsec_param-access`

Commençons par la manière d'accéder aux paramètres
à partir des modèles que vous connaissez déjà.

:begin_tab:`mxnet, pytorch, tensorflow`
Lorsqu'un modèle est défini via la classe `Sequential`,
nous pouvons d'abord accéder à n'importe quelle couche en indexant
le modèle comme s'il s'agissait d'une liste.
Les paramètres de chaque couche sont commodément
situés dans son attribut.
:end_tab:

:begin_tab:`jax`
Flax et JAX découplent le modèle et les paramètres comme vous
l'avez peut-être observé dans les modèles définis précédemment.
Lorsqu'un modèle est défini via la classe `Sequential`,
nous devons d'abord initialiser le réseau pour générer
le dictionnaire des paramètres. Nous pouvons accéder
aux paramètres de n'importe quelle couche via les clés de ce dictionnaire.
:end_tab:

Nous pouvons inspecter les paramètres de la deuxième couche entièrement connectée comme suit.

```{.python .input}
%%tab mxnet
net[1].params
```

```{.python .input}
%%tab pytorch
net[2].state_dict()
```

```{.python .input}
%%tab tensorflow
net.layers[2].weights
```

```{.python .input}
%%tab jax
params['params']['layers_2']
```

Nous pouvons voir que cette couche entièrement connectée
contient deux paramètres,
correspondant respectivement aux
poids et aux biais de cette couche.


### [**Paramètres ciblés**]

Notez que chaque paramètre est représenté
comme une instance de la classe de paramètre.
Pour faire quoi que ce soit d'utile avec les paramètres,
nous devons d'abord accéder aux valeurs numériques sous-jacentes.
Il existe plusieurs façons de le faire.
Certaines sont plus simples tandis que d'autres sont plus générales.
Le code suivant extrait le biais
de la deuxième couche du réseau neuronal, qui renvoie une instance de classe de paramètre, et
accède ensuite à la valeur de ce paramètre.

```{.python .input}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

```{.python .input}
%%tab jax
bias = params['params']['layers_2']['bias']
type(bias), bias
```

:begin_tab:`mxnet,pytorch`
Les paramètres sont des objets complexes,
contenant des valeurs, des gradients
et des informations supplémentaires.
C'est pourquoi nous devons demander la valeur explicitement.

En plus de la valeur, chaque paramètre nous permet également d'accéder au gradient. Parce que nous n'avons pas encore invoqué la rétropropagation pour ce réseau, il est dans son état initial.
:end_tab:

:begin_tab:`jax`
Contrairement aux autres frameworks, JAX ne conserve pas de trace des gradients sur les
paramètres du réseau neuronal ; au lieu de cela, les paramètres et le réseau sont découplés.
Il permet à l'utilisateur d'exprimer son calcul sous forme de
fonction Python et d'utiliser la transformation `grad` dans le même but.
:end_tab:

```{.python .input}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input}
%%tab pytorch
net[2].weight.grad == None
```

### [**Tous les paramètres à la fois**]

Lorsque nous devons effectuer des opérations sur tous les paramètres,
y accéder un par un peut devenir fastidieux.
La situation peut devenir particulièrement difficile à gérer
lorsque nous travaillons avec des modules plus complexes, par exemple imbriqués,
car nous devrions parcourir
toute l'arborescence de manière récursive pour extraire
les paramètres de chaque sous-module. Ci-dessous, nous démontrons l'accès aux paramètres de toutes les couches.

```{.python .input}
%%tab mxnet
net.collect_params()
```

```{.python .input}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input}
%%tab tensorflow
net.get_weights()
```

```{.python .input}
%%tab jax
jax.tree_util.tree_map(lambda x: x.shape, params)
```

## [**Paramètres liés**]

Souvent, nous voulons partager des paramètres entre plusieurs couches.
Voyons comment faire cela élégamment.
Dans ce qui suit, nous allouons une couche entièrement connectée
puis nous utilisons ses paramètres spécifiquement
pour définir ceux d'une autre couche.
Ici, nous devons exécuter la propagation avant
`net(X)` avant d'accéder aux paramètres.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))

net(X)
# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
%%tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
%%tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

```{.python .input}
%%tab jax
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8)
net = nn.Sequential([nn.Dense(8), nn.relu,
                     shared, nn.relu,
                     shared, nn.relu,
                     nn.Dense(1)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)

# Check whether the parameters are different
print(len(params['params']) == 3)
```

Cet exemple montre que les paramètres
de la deuxième et de la troisième couche sont liés.
Ils ne sont pas seulement égaux, ils sont
représentés par le même tenseur exact.
Ainsi, si nous modifions l'un des paramètres,
l'autre change également.

:begin_tab:`mxnet, pytorch, tensorflow`
Vous vous demandez peut-être,
lorsque les paramètres sont liés
qu'advient-il des gradients ?
Puisque les paramètres du modèle contiennent des gradients,
les gradients de la deuxième couche cachée
et de la troisième couche cachée sont additionnés
pendant la rétropropagation.
:end_tab:


## Résumé

Nous avons plusieurs façons d'accéder et de lier les paramètres du modèle.


## Exercices

1. Utilisez le modèle `NestMLP` défini dans la :numref:`sec_model_construction` et accédez aux paramètres des différentes couches.
1. Construisez un MLP contenant une couche de paramètres partagée et entraînez-le. Pendant le processus d'entraînement, observez les paramètres du modèle et les gradients de chaque couche.
1. Pourquoi le partage de paramètres est-il une bonne idée ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17990)
:end_tab:
