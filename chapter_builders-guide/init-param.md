```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Initialisation des paramètres

Maintenant que nous savons comment accéder aux paramètres,
examinons comment les initialiser correctement.
Nous avons discuté de la nécessité d'une initialisation appropriée dans :numref:`sec_numerical_stability`.
Le framework de deep learning fournit des initialisations aléatoires par défaut à ses couches.
Cependant, nous voulons souvent initialiser nos poids
selon divers autres protocoles. Le framework fournit les protocoles les plus
couramment utilisés, et permet également de créer un initialisateur personnalisé.

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

:begin_tab:`mxnet`
Par défaut, MXNet initialise les paramètres de poids en tirant aléatoirement dans une distribution uniforme $U(-0,07, 0,07)$,
et en mettant les paramètres de biais à zéro.
Le module `init` de MXNet propose une variété
de méthodes d'initialisation prédéfinies.
:end_tab:

:begin_tab:`pytorch`
Par défaut, PyTorch initialise les matrices de poids et de biais
uniformément en tirant dans une plage calculée en fonction des dimensions d'entrée et de sortie.
Le module `nn.init` de PyTorch propose une variété
de méthodes d'initialisation prédéfinies.
:end_tab:

:begin_tab:`tensorflow`
Par défaut, Keras initialise les matrices de poids uniformément en tirant dans une plage calculée en fonction des dimensions d'entrée et de sortie, et les paramètres de biais sont tous mis à zéro.
TensorFlow propose une variété de méthodes d'initialisation tant dans le module racine que dans le module `keras.initializers`.
:end_tab:

:begin_tab:`jax`
Par défaut, Flax initialise les poids en utilisant `jax.nn.initializers.lecun_normal`,
c'est-à-dire en tirant des échantillons d'une distribution normale tronquée centrée sur 0 avec
l'écart-type défini comme la racine carrée de $1 / \textrm{fan}_{\textrm{in}}$
où `fan_in` est le nombre d'unités d'entrée dans le tenseur de poids. Les paramètres de biais
sont tous mis à zéro.
Le module `nn.initializers` de Jax propose une variété
de méthodes d'initialisation prédéfinies.
:end_tab:

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
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
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

## [**Initialisation intégrée**]

Commençons par faire appel aux initialisateurs intégrés.
Le code ci-dessous initialise tous les paramètres de poids
comme des variables aléatoires gaussiennes
avec un écart-type de 0,01, tandis que les paramètres de biais sont mis à zéro.

```{.python .input}
%%tab mxnet
# Here force_reinit ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.normal(0.01)
bias_init = nn.initializers.zeros

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

Nous pouvons également initialiser tous les paramètres
à une valeur constante donnée (par exemple, 1).

```{.python .input}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.constant(1)

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

[**Nous pouvons également appliquer différents initialisateurs pour certains blocs.**]
Par exemple, ci-dessous nous initialisons la première couche
avec l'initialisateur de Xavier
et initialisons la deuxième couche
à une valeur constante de 42.

```{.python .input}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
%%tab pytorch
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8, kernel_init=nn.initializers.xavier_uniform(),
                              bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=nn.initializers.constant(42),
                              bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
params['params']['layers_0']['kernel'][:, 0], params['params']['layers_2']['kernel']
```

### [**Initialisation personnalisée**]

Parfois, les méthodes d'initialisation dont nous avons besoin
ne sont pas fournies par le framework de deep learning.
Dans l'exemple ci-dessous, nous définissons un initialisateur
pour n'importe quel paramètre de poids $w$ en utilisant la distribution étrange suivante :

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ avec une probabilité de } \frac{1}{4} \\
            0    & \textrm{ avec une probabilité de } \frac{1}{2} \\
        U(-10, -5) & \textrm{ avec une probabilité de } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Ici, nous définissons une sous-classe de la classe `Initializer`.
Habituellement, nous n'avons qu'à implémenter la fonction `_init_weight`
qui prend un argument de tenseur (`data`)
et lui assigne les valeurs initialisées souhaitées.
:end_tab:

:begin_tab:`pytorch`
À nouveau, nous implémentons une fonction `my_init` à appliquer au `net`.
:end_tab:

:begin_tab:`tensorflow`
Ici, nous définissons une sous-classe de `Initializer` et implémentons la fonction `__call__`
qui renvoie un tenseur souhaité étant donné la forme et le type de données.
:end_tab:

:begin_tab:`jax`
Les fonctions d'initialisation de Jax prennent comme arguments la `PRNGKey`, `shape` et
`dtype`. Ici, nous implémentons la fonction `my_init` qui renvoie un tenseur
souhaité étant donné la forme et le type de données.
:end_tab:

```{.python .input}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
%%tab pytorch
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

```{.python .input}
%%tab jax
def my_init(key, shape, dtype=jnp.float_):
    data = jax.random.uniform(key, shape, minval=-10, maxval=10)
    return data * (jnp.abs(data) >= 5)

net = nn.Sequential([nn.Dense(8, kernel_init=my_init), nn.relu, nn.Dense(1)])
params = net.init(d2l.get_key(), X)
print(params['params']['layers_0']['kernel'][:, :2])
```

:begin_tab:`mxnet, pytorch, tensorflow`
Notez que nous avons toujours la possibilité
de définir les paramètres directement.
:end_tab:

:begin_tab:`jax`
Lors de l'initialisation des paramètres dans JAX et Flax, le dictionnaire de paramètres
renvoyé est de type `flax.core.frozen_dict.FrozenDict`. Il n'est pas conseillé dans
l'écosystème Jax de modifier directement les valeurs d'un tableau, par conséquent les types de données
sont généralement immuables. On peut utiliser `params.unfreeze()` pour effectuer des modifications.
:end_tab:

```{.python .input}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## Résumé

Nous pouvons initialiser les paramètres en utilisant des initialisateurs intégrés et personnalisés.

## Exercices

Recherchez dans la documentation en ligne d'autres initialisateurs intégrés.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/8089)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8090)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8091)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17991)
:end_tab:
