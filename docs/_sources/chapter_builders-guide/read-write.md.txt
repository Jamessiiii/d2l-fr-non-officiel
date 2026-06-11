```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# E/S de fichiers

Jusqu'à présent, nous avons vu comment traiter les données et comment
construire, entraîner et tester des modèles de deep learning.
Cependant, à un moment donné, nous serons, espérons-le, suffisamment satisfaits
des modèles appris pour vouloir
sauvegarder les résultats en vue d'une utilisation ultérieure dans divers contextes
(peut-être même pour faire des prédictions en production).
De plus, lors de l'exécution d'un long processus d'entraînement,
la meilleure pratique consiste à sauvegarder périodiquement les résultats intermédiaires (points de contrôle ou *checkpointing*)
pour s'assurer de ne pas perdre plusieurs jours de calcul
si nous trébuchons sur le cordon d'alimentation de notre serveur.
Il est donc temps d'apprendre à charger et à stocker
à la fois des vecteurs de poids individuels et des modèles entiers.
Cette section aborde ces deux questions.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
import numpy as np
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp
```

## (**Chargement et sauvegarde de tenseurs**)

Pour les tenseurs individuels, nous pouvons directement
invoquer les fonctions `load` et `save`
pour les lire et les écrire respectivement.
Les deux fonctions exigent que nous fournissions un nom,
et `save` nécessite en entrée la variable à sauvegarder.

```{.python .input}
%%tab mxnet
x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
%%tab pytorch
x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
%%tab tensorflow
x = tf.range(4)
np.save('x-file.npy', x)
```

```{.python .input}
%%tab jax
x = jnp.arange(4)
jnp.save('x-file.npy', x)
```

Nous pouvons maintenant relire les données du fichier stocké dans la mémoire.

```{.python .input}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input}
%%tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```{.python .input}
%%tab jax
x2 = jnp.load('x-file.npy', allow_pickle=True)
x2
```

Nous pouvons [**stocker une liste de tenseurs et les relire en mémoire.**]

```{.python .input}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```{.python .input}
%%tab jax
y = jnp.zeros(4)
jnp.save('xy-files.npy', [x, y])
x2, y2 = jnp.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

Nous pouvons même [**écrire et lire un dictionnaire qui associe des
chaînes de caractères à des tenseurs.**]
C'est pratique lorsque nous voulons
lire ou écrire tous les poids d'un modèle.

```{.python .input}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```{.python .input}
%%tab jax
mydict = {'x': x, 'y': y}
jnp.save('mydict.npy', mydict)
mydict2 = jnp.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**Chargement et sauvegarde des paramètres du modèle**]

Sauvegarder des vecteurs de poids individuels (ou d'autres tenseurs) est utile,
mais cela devient très fastidieux si nous voulons sauvegarder
(et charger plus tard) un modèle entier.
Après tout, nous pourrions avoir des centaines de
groupes de paramètres répartis un peu partout.
C'est pourquoi le framework de deep learning fournit des fonctionnalités intégrées
pour charger et sauvegarder des réseaux entiers.
Un détail important à noter est que cela
sauvegarde les *paramètres* du modèle et non le modèle entier.
Par exemple, si nous avons un MLP à 3 couches,
nous devons spécifier l'architecture séparément.
La raison en est que les modèles eux-mêmes peuvent contenir du code arbitraire,
ils ne peuvent donc pas être sérialisés aussi naturellement.
Ainsi, afin de rétablir un modèle, nous devons
générer l'architecture dans le code
puis charger les paramètres à partir du disque.
(**Commençons par notre MLP familier.**)

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)

    def __call__(self, x):
        return self.output(nn.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 20))
Y, params = net.init_with_output(jax.random.PRNGKey(d2l.get_seed()), X)
```

Ensuite, nous [**stockons les paramètres du modèle sous forme de fichier**] portant le nom « mlp.params ».

```{.python .input}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
%%tab tensorflow
net.save_weights('mlp.params')
```

```{.python .input}
%%tab jax
checkpoints.save_checkpoint('ckpt_dir', params, step=1, overwrite=True)
```

Pour récupérer le modèle, nous instancions un clone
du modèle MLP d'origine.
Au lieu d'initialiser aléatoirement les paramètres du modèle,
nous [**lisons directement les paramètres stockés dans le fichier**].

```{.python .input}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
%%tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

```{.python .input}
%%tab jax
clone = MLP()
cloned_params = flax.core.freeze(checkpoints.restore_checkpoint('ckpt_dir',
                                                                target=None))
```

Puisque les deux instances ont les mêmes paramètres de modèle,
le résultat du calcul pour la même entrée `X` devrait être le même.
Vérifions-le.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
%%tab jax
Y_clone = clone.apply(cloned_params, X)
Y_clone == Y
```

## Résumé

Les fonctions `save` et `load` peuvent être utilisées pour effectuer des E/S de fichiers pour les objets tenseurs.
Nous pouvons sauvegarder et charger l'ensemble des paramètres d'un réseau via un dictionnaire de paramètres.
La sauvegarde de l'architecture doit être faite dans le code plutôt que dans les paramètres.

## Exercices

1. Même s'il n'est pas nécessaire de déployer des modèles entraînés sur un autre appareil, quels sont les avantages pratiques du stockage des paramètres du modèle ?
1. Supposons que nous voulions réutiliser seulement des parties d'un réseau pour les incorporer dans un réseau ayant une architecture différente. Comment feriez-vous pour utiliser, par exemple, les deux premières couches d'un réseau précédent dans un nouveau réseau ?
1. Comment feriez-vous pour sauvegarder l'architecture et les paramètres du réseau ? Quelles restrictions imposeriez-vous à l'architecture ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17994)
:end_tab: