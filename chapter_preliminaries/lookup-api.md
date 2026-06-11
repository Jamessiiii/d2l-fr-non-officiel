```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Documentation
:begin_tab:`mxnet`
Bien qu'il nous soit impossible de présenter chaque fonction et classe de MXNet (et que les informations puissent devenir obsolètes rapidement), la [documentation de l'API](https://mxnet.apache.org/versions/1.8.0/api) ainsi que des [tutoriels](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) et exemples supplémentaires fournissent une telle documentation. Cette section donne quelques conseils sur la manière d'explorer l'API MXNet.
:end_tab:

:begin_tab:`pytorch`
Bien qu'il nous soit impossible de présenter chaque fonction et classe de PyTorch (et que les informations puissent devenir obsolètes rapidement), la [documentation de l'API](https://pytorch.org/docs/stable/index.html) ainsi que des [tutoriels](https://pytorch.org/tutorials/beginner/basics/intro.html) et exemples supplémentaires fournissent une telle documentation. Cette section donne quelques conseils sur la manière d'explorer l'API PyTorch.
:end_tab:

:begin_tab:`tensorflow`
Bien qu'il nous soit impossible de présenter chaque fonction et classe de TensorFlow (et que les informations puissent devenir obsolètes rapidement), la [documentation de l'API](https://www.tensorflow.org/api_docs) ainsi que des [tutoriels](https://www.tensorflow.org/tutorials) et exemples supplémentaires fournissent une telle documentation. Cette section donne quelques conseils sur la manière d'explorer l'API TensorFlow.
:end_tab:

```{.python .input}
%%tab mxnet
from mxnet import np
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
import jax
```

## Fonctions et classes dans un module

Pour savoir quelles fonctions et classes peuvent être appelées dans un module, nous invoquons la fonction `dir`. Par exemple, nous pouvons (**interroger toutes les propriétés du module pour générer des nombres aléatoires**) :

```{.python .input  n=1}
%%tab mxnet
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
print(dir(tf.random))
```

```{.python .input}
%%tab jax
print(dir(jax.random))
```

En général, nous pouvons ignorer les fonctions qui commencent et se terminent par `__` (objets spéciaux en Python) ou les fonctions qui commencent par un simple `_` (généralement des fonctions internes). Sur la base des noms de fonctions ou d'attributs restants, nous pourrions risquer de deviner que ce module offre diverses méthodes pour générer des nombres aléatoires, notamment l'échantillonnage à partir de la distribution uniforme (`uniform`), de la distribution normale (`normal`) et de la distribution multinomiale (`multinomial`).

## Fonctions et classes spécifiques

Pour des instructions spécifiques sur la façon d'utiliser une fonction ou une classe donnée, nous pouvons invoquer la fonction `help`. À titre d'exemple, [**explorons les instructions d'utilisation de la fonction `ones` des tenseurs**].

```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

```{.python .input}
%%tab jax
help(jax.numpy.ones)
```

D'après la documentation, nous pouvons voir que la fonction `ones` crée un nouveau tenseur avec la forme spécifiée et définit tous les éléments à la valeur 1. Dans la mesure du possible, vous devriez (**effectuer un test rapide**) pour confirmer votre interprétation :

```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

```{.python .input}
%%tab jax
jax.numpy.ones(4)
```

Dans le notebook Jupyter, nous pouvons utiliser `?` pour afficher le document dans une autre fenêtre. Par exemple, `list?` créera un contenu presque identique à `help(list)`, en l'affichant dans une nouvelle fenêtre du navigateur. De plus, si nous utilisons deux points d'interrogation, comme `list??`, le code Python implémentant la fonction sera également affiché.

La documentation officielle fournit de nombreuses descriptions et exemples qui dépassent le cadre de ce livre. Nous mettons l'accent sur les cas d'utilisation importants qui vous permettront de démarrer rapidement avec des problèmes pratiques, plutôt que sur l'exhaustivité de la couverture. Nous vous encourageons également à étudier le code source des bibliothèques pour voir des exemples d'implémentations de haute qualité de code de production. En faisant cela, vous deviendrez un meilleur ingénieur en plus de devenir un meilleur scientifique.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17972)
:end_tab:
