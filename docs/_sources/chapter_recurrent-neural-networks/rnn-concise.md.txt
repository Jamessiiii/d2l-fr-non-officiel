# Implémentation concise des réseaux de neurones récurrents
:label:`sec_rnn-concise`

Comme la plupart de nos implémentations à partir de zéro,
la :numref:`sec_rnn-scratch` a été conçue 
pour donner un aperçu du fonctionnement de chaque composant.
Mais lorsque vous utilisez des RNN au quotidien 
ou que vous écrivez du code de production,
vous voudrez vous appuyer davantage sur des bibliothèques
qui réduisent à la fois le temps d'implémentation 
(en fournissant du code de bibliothèque pour les modèles et fonctions courants)
et le temps de calcul 
(en optimisant au maximum ces implémentations de bibliothèque).
Cette section vous montrera comment implémenter 
le même modèle de langage plus efficacement
en utilisant l'API de haut niveau fournie 
par votre framework de deep learning.
Nous commençons, comme précédemment, par charger 
le jeu de données *The Time Machine*.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
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
from jax import numpy as jnp
```

## [**Définition du modèle**]

Nous définissons la classe suivante
en utilisant le RNN implémenté
par les API de haut niveau.

:begin_tab:`mxnet`
Plus précisément, pour initialiser l'état caché,
nous appelons la méthode membre `begin_state`.
Cela renvoie une liste qui contient
un état caché initial
pour chaque exemple du mini-lot,
dont la forme est
(nombre de couches cachées, taille du lot, nombre d'unités cachées).
Pour certains modèles qui seront présentés plus tard
(par exemple, la mémoire à long court terme ou LSTM),
cette liste contiendra également d'autres informations.
:end_tab:

:begin_tab:`jax`
À ce jour, Flax ne fournit pas d'RNNCell pour une implémentation concise des RNN classiques.
Il existe des variantes plus avancées de RNN comme les LSTM et les GRU
qui sont disponibles dans l'API `linen` de Flax.
:end_tab:

```{.python .input}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input}
%%tab pytorch
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

```{.python .input}
%%tab jax
class RNN(nn.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None):
        raise NotImplementedError
```

Héritant de la classe `RNNLMScratch` de la :numref:`sec_rnn-scratch`, 
la classe `RNNLM` suivante définit un modèle de langage complet basé sur un RNN.
Notez que nous devons créer une couche de sortie entièrement connectée séparée.

```{.python .input}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    training: bool = True

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state, self.training)
        return self.output_layer(rnn_outputs)
```

## Entraînement et prédiction

Avant d'entraîner le modèle, [**effectuons une prédiction 
avec un modèle initialisé avec des poids aléatoires.**]
Étant donné que nous n'avons pas entraîné le réseau, 
il générera des prédictions absurdes.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

Ensuite, nous [**entraînons notre modèle, en tirant parti de l'API de haut niveau**].

```{.python .input}
%%tab pytorch, mxnet, tensorflow
if tab.selected('mxnet', 'pytorch'):
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

Par rapport à la :numref:`sec_rnn-scratch`,
ce modèle atteint une perplexité comparable,
mais s'exécute plus rapidement grâce aux implémentations optimisées.
Comme auparavant, nous pouvons générer des jetons prédits 
à la suite de la chaîne de préfixe spécifiée.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## Résumé

Les API de haut niveau dans les frameworks de deep learning fournissent des implémentations de RNN standard.
Ces bibliothèques vous aident à éviter de perdre du temps à réimplémenter des modèles standards.
De plus,
les implémentations des frameworks sont souvent hautement optimisées, 
  entraînant des gains de performance (de calcul) significatifs 
  par rapport aux implémentations à partir de zéro.

## Exercices

1. Pouvez-vous faire en sorte que le modèle RNN sur-apprenne (overfit) en utilisant les API de haut niveau ?
1. Implémentez le modèle autorégressif de la :numref:`sec_sequence` en utilisant un RNN.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2211)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18015)
:end_tab:
