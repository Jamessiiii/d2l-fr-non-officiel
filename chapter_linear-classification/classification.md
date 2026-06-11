```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Le modèle de base pour la classification
:label:`sec_classification`

Vous avez peut-être remarqué que les implémentations à partir de zéro et l'implémentation concise utilisant les fonctionnalités du framework étaient assez similaires dans le cas de la régression. Il en va de même pour la classification. Étant donné que de nombreux modèles de ce livre traitent de la classification, il est utile d'ajouter des fonctionnalités pour prendre en charge spécifiquement ce cadre. Cette section fournit une classe de base pour les modèles de classification afin de simplifier le code futur.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## La classe `Classifier`

:begin_tab:`pytorch, mxnet, tensorflow`
Nous définissons la classe `Classifier` ci-dessous. Dans `validation_step`, nous rapportons à la fois la valeur de la perte et l'exactitude de la classification sur un lot de validation. Nous effectuons une mise à jour tous les `num_val_batches` lots. Cela présente l'avantage de générer la perte et l'exactitude moyennes sur l'ensemble des données de validation. Ces chiffres moyens ne sont pas tout à fait corrects si le lot final contient moins d'exemples, mais nous ignorons cette différence mineure pour garder le code simple.
:end_tab:


:begin_tab:`jax`
Nous définissons la classe `Classifier` ci-dessous. Dans `validation_step`, nous rapportons à la fois la valeur de la perte et l'exactitude de la classification sur un lot de validation. Nous effectuons une mise à jour tous les `num_val_batches` lots. Cela présente l'avantage de générer la perte et l'exactitude moyennes sur l'ensemble des données de validation. Ces chiffres moyens ne sont pas tout à fait corrects si le dernier lot contient moins d'exemples, mais nous ignorons cette différence mineure pour garder le code simple.

Nous redéfinissons également la méthode `training_step` pour JAX, car tous les modèles qui hériteront plus tard de `Classifier` auront une perte qui renvoie des données auxiliaires. Ces données auxiliaires peuvent être utilisées pour les modèles avec normalisation par lots (batch normalization) (qui sera expliquée dans la :numref:`sec_batch_norm`), tandis que dans tous les autres cas, nous ferons en sorte que la perte renvoie également un espace réservé (placeholder) (dictionnaire vide) pour représenter les données auxiliaires.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

```{.python .input}
%%tab jax
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def training_step(self, params, batch, state):
        # Here value is a tuple since models with BatchNorm layers require
        # the loss to return auxiliary data
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # Discard the second returned value. It is used for training models
        # with BatchNorm layers since loss also returns auxiliary data
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)
```

Par défaut, nous utilisons un optimiseur de descente de gradient stochastique, opérant sur des mini-lots, tout comme nous l'avons fait dans le contexte de la régression linéaire.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

## Exactitude

Étant donné la distribution de probabilité prédite `y_hat`, nous choisissons généralement la classe ayant la probabilité prédite la plus élevée chaque fois que nous devons produire une prédiction stricte. En effet, de nombreuses applications exigent que nous fassions un choix. Par exemple, Gmail doit classer un e-mail dans les catégories « Principale », « Réseaux sociaux », « Promotions », « Forums » ou « Spam ». Il peut estimer les probabilités en interne, mais au bout du compte, il doit choisir l'une des classes.

Lorsque les prédictions sont cohérentes avec la classe de l'étiquette `y`, elles sont correctes. L'exactitude de la classification est la fraction de toutes les prédictions qui sont correctes. Bien qu'il puisse être difficile d'optimiser directement l'exactitude (elle n'est pas dérivable), c'est souvent la mesure de performance qui nous intéresse le plus. C'est souvent *la* quantité pertinente dans les benchmarks. À ce titre, nous la rapporterons presque toujours lors de l'entraînement de classifieurs.

L'exactitude est calculée comme suit. Tout d'abord, si `y_hat` est une matrice, nous supposons que la deuxième dimension stocke les scores de prédiction pour chaque classe. Nous utilisons `argmax` pour obtenir la classe prédite par l'indice de l'entrée la plus grande dans chaque ligne. Ensuite, nous [**comparons la classe prédite avec la vérité terrain `y` élément par élément.**] Comme l'opérateur d'égalité `==` est sensible aux types de données, nous convertissons le type de données de `y_hat` pour qu'il corresponde à celui de `y`. Le résultat est un tenseur contenant des entrées de 0 (faux) et 1 (vrai). Faire la somme donne le nombre de prédictions correctes.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def accuracy(self, params, X, Y, state, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = state.apply_fn({'params': params,
                            'batch_stats': state.batch_stats},  # BatchNorm Only
                           *X)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## Résumé

La classification est un problème suffisamment courant pour justifier ses propres fonctions de commodité. L'exactitude du classifieur est d'une importance centrale dans la classification. Notez que si nous nous soucions souvent principalement de l'exactitude, nous entraînons les classifieurs pour optimiser une variété d'autres objectifs pour des raisons statistiques et computationnelles. Cependant, quel que soit la fonction de perte qui a été minimisée pendant l'entraînement, il est utile d'avoir une méthode pratique pour évaluer empiriquement l'exactitude de notre classifieur.


## Exercices

1. Notons $L_\textrm{v}$ la perte de validation, et soit $L_\textrm{v}^\textrm{q}$ son estimation rapide et approximative calculée par la moyenne de la fonction de perte dans cette section. Enfin, notons $l_\textrm{v}^\textrm{b}$ la perte sur le dernier mini-lot. Exprimez $L_\textrm{v}$ en fonction de $L_\textrm{v}^\textrm{q}$, $l_\textrm{v}^\textrm{b}$, et des tailles de l'échantillon et du mini-lot.
1. Montrez que l'estimation rapide et approximative $L_\textrm{v}^\textrm{q}$ est sans biais. C'est-à-dire, montrez que $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$. Pourquoi voudriez-vous tout de même utiliser $L_\textrm{v}$ à la place ?
1. Étant donné une perte de classification multiclasse, en notant $l(y,y')$ la pénalité liée à l'estimation de $y'$ lorsque nous voyons $y$ et étant donné une probabilité $p(y \mid x)$, formulez la règle pour une sélection optimale de $y'$. Indice : exprimez la perte attendue, en utilisant $l$ et $p(y \mid x)$.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6808)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6810)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17981)
:end_tab:
