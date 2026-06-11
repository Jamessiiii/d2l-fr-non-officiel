```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Initialisation paresseuse
:label:`sec_lazy_init`

Jusqu'à présent, il peut sembler que nous ayons été négligents
dans la configuration de nos réseaux.
Plus précisément, nous avons fait les choses peu intuitives suivantes,
qui pourraient sembler ne pas devoir fonctionner :

* Nous avons défini les architectures de réseaux
  sans spécifier la dimensionnalité d'entrée.
* Nous avons ajouté des couches sans spécifier
  la dimension de sortie de la couche précédente.
* Nous avons même "initialisé" ces paramètres
  avant de fournir suffisamment d'informations pour déterminer
  combien de paramètres nos modèles devraient contenir.

Vous pourriez être surpris que notre code fonctionne.
Après tout, il n'y a aucun moyen pour le framework de deep learning
de savoir quelle serait la dimensionnalité d'entrée d'un réseau.
L'astuce ici est que le framework *diffère l'initialisation*,
attendant jusqu'à la première fois que nous passons des données à travers le modèle,
pour inférer les tailles de chaque couche à la volée.


Plus tard, en travaillant avec des réseaux de neurones convolutionnels,
cette technique deviendra encore plus pratique
puisque la dimensionnalité d'entrée
(par exemple, la résolution d'une image)
affectera la dimensionnalité
de chaque couche suivante.
Par conséquent, la capacité de définir des paramètres
sans avoir besoin de connaître,
au moment de l'écriture du code,
la valeur de la dimension
peut grandement simplifier la tâche de spécification
et de modification ultérieure de nos modèles.
Ensuite, nous approfondissons les mécanismes de l'initialisation.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
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

Pour commencer, instancions un MLP.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])
```

À ce stade, le réseau ne peut pas connaître
les dimensions des poids de la couche d'entrée
car la dimension d'entrée reste inconnue.

:begin_tab:`mxnet, pytorch, tensorflow`
Par conséquent, le framework n'a pas encore initialisé de paramètres.
Nous le confirmons en essayant d'accéder aux paramètres ci-dessous.
:end_tab:

:begin_tab:`jax`
Comme mentionné dans :numref:`subsec_param-access`, les paramètres et la définition du réseau sont découplés
dans Jax et Flax, et l'utilisateur gère les deux manuellement. Les modèles Flax sont sans état (stateless)
il n'y a donc pas d'attribut `parameters`.
:end_tab:

```{.python .input}
%%tab mxnet
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
%%tab pytorch
net[0].weight
```

```{.python .input}
%%tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
Notez que bien que les objets de paramètres existent,
la dimension d'entrée de chaque couche est répertoriée comme -1.
MXNet utilise la valeur spéciale -1 pour indiquer
que la dimension du paramètre reste inconnue.
À ce stade, les tentatives d'accès à `net[0].weight.data()`
déclencheraient une erreur d'exécution indiquant que le réseau
doit être initialisé avant que les paramètres ne puissent être consultés.
Voyons maintenant ce qui se passe lorsque nous essayons d'initialiser
les paramètres via la méthode `initialize`.
:end_tab:

:begin_tab:`tensorflow`
Notez que chaque objet de couche existe mais que les poids sont vides.
L'utilisation de `net.get_weights()` lancerait une erreur puisque les poids
n'ont pas encore été initialisés.
:end_tab:

```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
Comme nous pouvons le voir, rien n'a changé.
Lorsque les dimensions d'entrée sont inconnues,
les appels à `initialize` n'initialisent pas vraiment les paramètres.
Au lieu de cela, cet appel enregistre auprès de MXNet que nous souhaitons
(et éventuellement, selon quelle distribution)
initialiser les paramètres.
:end_tab:

Passons ensuite des données à travers le réseau
pour que le framework initialise enfin les paramètres.

```{.python .input}
%%tab mxnet
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
%%tab pytorch
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

```{.python .input}
%%tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

```{.python .input}
%%tab jax
params = net.init(d2l.get_key(), jnp.zeros((2, 20)))
jax.tree_util.tree_map(lambda x: x.shape, params).tree_flatten_with_keys()
```

Dès que nous connaissons la dimensionnalité d'entrée,
20,
le framework peut identifier la forme de la matrice de poids de la première couche en utilisant la valeur 20.
Ayant reconnu la forme de la première couche, le framework passe
à la seconde couche,
et ainsi de suite à travers le graphe de calcul
jusqu'à ce que toutes les formes soient connues.
Notez que dans ce cas,
seule la première couche nécessite une initialisation paresseuse,
mais le framework initialise séquentiellement.
Une fois que toutes les formes de paramètres sont connues,
le framework peut enfin initialiser les paramètres.

:begin_tab:`pytorch`
La méthode suivante
passe des entrées fictives (dummy inputs)
à travers le réseau
pour un essai à blanc (dry run)
afin d'inférer toutes les formes de paramètres
et initialise ensuite les paramètres.
Elle sera utilisée plus tard lorsque les initialisations aléatoires par défaut ne sont pas souhaitées.
:end_tab:

:begin_tab:`jax`
L'initialisation des paramètres dans Flax est toujours effectuée manuellement et gérée par l'utilisateur.
La méthode suivante prend une entrée fictive et un dictionnaire de clés comme argument.
Ce dictionnaire de clés contient les générateurs de nombres aléatoires (rngs) pour initialiser les paramètres du modèle
et le rng de dropout pour générer le masque de dropout pour les modèles avec des couches de dropout.
Plus de détails sur le dropout seront couverts plus tard dans :numref:`sec_dropout`.
Enfin, la méthode initialise le modèle en renvoyant les paramètres.
Nous l'avons également utilisée en arrière-plan dans les sections précédentes.
:end_tab:

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, dummy_input, key):
    params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
    return params
```

## Résumé

L'initialisation paresseuse peut être pratique, permettant au framework d'inférer automatiquement les formes des paramètres, ce qui facilite la modification des architectures et élimine une source courante d'erreurs.
Nous pouvons passer des données à travers le modèle pour que le framework initialise enfin les paramètres.


## Exercices

1. Que se passe-t-il si vous spécifiez les dimensions d'entrée de la première couche mais pas des couches suivantes ? Obtenez-vous une initialisation immédiate ?
1. Que se passe-t-il si vous spécifiez des dimensions incohérentes ?
1. Que devriez-vous faire si vous avez une entrée de dimensionnalité variable ? Indice : regardez le partage de paramètres (parameter tying).

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8092)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17992)
:end_tab:
