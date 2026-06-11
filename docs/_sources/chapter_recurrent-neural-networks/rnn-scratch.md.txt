# Implémentation de réseau de neurones récurrent à partir de zéro
:label:`sec_rnn-scratch`

Nous sommes maintenant prêts à implémenter un RNN à partir de zéro.
En particulier, nous allons entraîner ce RNN pour qu'il fonctionne
comme un modèle de langage au niveau des caractères
(voir :numref:`sec_rnn`)
et nous l'entraînerons sur un corpus constitué de
l'intégralité du texte de *La Machine à explorer le temps* de H. G. Wells,
en suivant les étapes de traitement des données
décrites dans :numref:`sec_text-sequence`.
Nous commençons par charger le jeu de données.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import math
```

## Modèle RNN

Nous commençons par définir une classe
pour implémenter le modèle RNN
(:numref:`subsec_rnn_w_hidden_states`).
Notez que le nombre d'unités cachées `num_hiddens`
est un hyperparamètre ajustable.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
            self.W_hh = d2l.randn(
                num_hiddens, num_hiddens) * sigma
            self.b_h = d2l.zeros(num_hiddens)
        if tab.selected('pytorch'):
            self.W_xh = nn.Parameter(
                d2l.randn(num_inputs, num_hiddens) * sigma)
            self.W_hh = nn.Parameter(
                d2l.randn(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

```{.python .input  n=7}
%%tab jax
class RNNScratch(nn.Module):  #@save
    """The RNN model implemented from scratch."""
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.W_xh = self.param('W_xh', nn.initializers.normal(self.sigma),
                               (self.num_inputs, self.num_hiddens))
        self.W_hh = self.param('W_hh', nn.initializers.normal(self.sigma),
                               (self.num_hiddens, self.num_hiddens))
        self.b_h = self.param('b_h', nn.initializers.zeros, (self.num_hiddens))
```

[**La méthode `forward` ci-dessous définit comment calculer
la sortie et l'état caché à n'importe quel pas de temps,
étant donné l'entrée actuelle et l'état du modèle
au pas de temps précédent.**]
Notez que le modèle RNN boucle à travers
la dimension la plus externe de `inputs`,
mettant à jour l'état caché
un pas de temps à la fois.
Le modèle utilise ici une fonction d'activation $\tanh$ (:numref:`subsec_tanh`).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              ctx=inputs.ctx)
        if tab.selected('pytorch'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        if tab.selected('tensorflow'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.num_hiddens))
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(RNNScratch)  #@save
def __call__(self, inputs, state=None):
    if state is not None:
        state, = state
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

Nous pouvons alimenter un modèle RNN avec un mini-lot de séquences d'entrée comme suit.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

```{.python .input  n=11}
%%tab jax
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
(outputs, state), _ = rnn.init_with_output(d2l.get_key(), X)
```

Vérifions si le modèle RNN
produit des résultats aux formes correctes
pour s'assurer que la dimensionnalité
de l'état caché reste inchangée.

```{.python .input}
%%tab all
def check_len(a, n):  #@save
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'
    
def check_shape(a, shape):  #@save
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

## Modèle de langage basé sur un RNN

La classe `RNNLMScratch` suivante définit
un modèle de langage basé sur un RNN,
où nous passons notre RNN
via l'argument `rnn`
de la méthode `__init__`.
Lors de l'entraînement de modèles de langage,
les entrées et les sorties proviennent
du même vocabulaire.
Par conséquent, elles ont la même dimension,
qui est égale à la taille du vocabulaire.
Notez que nous utilisons la perplexité pour évaluer le modèle.
Comme discuté dans :numref:`subsec_perplexity`, cela garantit
que les séquences de différentes longueurs sont comparables.

```{.python .input}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        if tab.selected('mxnet'):
            self.W_hq = d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
            self.b_q = d2l.zeros(self.vocab_size)        
            for param in self.get_scratch_params():
                param.attach_grad()
        if tab.selected('tensorflow'):
            self.W_hq = tf.Variable(d2l.normal(
                (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
            self.b_q = tf.Variable(d2l.zeros(self.vocab_size))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input  n=14}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    rnn: nn.Module
    vocab_size: int
    lr: float = 0.01

    def setup(self):
        self.W_hq = self.param('W_hq', nn.initializers.normal(self.rnn.sigma),
                               (self.rnn.num_hiddens, self.vocab_size))
        self.b_q = self.param('b_q', nn.initializers.zeros, (self.vocab_size))

    def training_step(self, params, batch, state):
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot('ppl', d2l.exp(l), train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('ppl', d2l.exp(l), train=False)
```

### [**Codage One-Hot**]

Rappelez-vous que chaque jeton est représenté
par un index numérique indiquant la
position dans le vocabulaire du
mot/caractère/élément de mot correspondant.
Vous pourriez être tenté de construire un réseau de neurones
avec un seul nœud d'entrée (à chaque pas de temps),
où l'index pourrait être fourni comme une valeur scalaire.
Cela fonctionne lorsque nous traitons des entrées numériques
comme le prix ou la température, où deux valeurs
suffisamment proches
doivent être traitées de manière similaire.
Mais cela n'a pas tout à fait de sens ici.
Les $45^{\textrm{ème}}$ et $46^{\textrm{ème}}$ mots
de notre vocabulaire s'avèrent être "their" et "said",
dont les significations ne sont absolument pas similaires.

Lorsqu'on traite de telles données catégorielles,
la stratégie la plus courante consiste à représenter
chaque élément par un *codage one-hot*
(rappelez-vous de la :numref:`subsec_classification-problem`).
Un codage one-hot est un vecteur dont la longueur
est donnée par la taille du vocabulaire $N$,
où toutes les entrées sont fixées à $0$,
sauf l'entrée correspondant
à notre jeton, qui est fixée à $1$.
Par exemple, si le vocabulaire comptait cinq éléments,
alors les vecteurs one-hot correspondant
aux indices 0 et 2 seraient les suivants.

```{.python .input}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

```{.python .input  n=18}
%%tab jax
jax.nn.one_hot(jnp.array([0, 2]), 5)
```

(**Les mini-lots que nous échantillonnons à chaque itération
auront la forme (taille du lot, nombre de pas de temps).
Une fois chaque entrée représentée sous forme de vecteur one-hot,
nous pouvons considérer chaque mini-lot comme un tenseur tridimensionnel,
où la longueur le long du troisième axe
est donnée par la taille du vocabulaire (`len(vocab)`).**)
Nous transposons souvent l'entrée de manière à obtenir une sortie
de forme (nombre de pas de temps, taille du lot, taille du vocabulaire).
Cela nous permettra de boucler plus commodément à travers la dimension la plus externe
pour mettre à jour les états cachés d'un mini-lot,
pas à pas dans le temps
(par exemple, dans la méthode `forward` ci-dessus).

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def one_hot(self, X):    
    # Output shape: (num_steps, batch_size, vocab_size)    
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
    if tab.selected('jax'):
        return jax.nn.one_hot(X.T, self.vocab_size)
```

### Transformation des sorties du RNN

Le modèle de langage utilise une couche de sortie entièrement connectée
pour transformer les sorties du RNN en prédictions de jetons à chaque pas de temps.

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

[**Vérifions si le calcul de propagation avant
produit des sorties avec la forme correcte.**]

```{.python .input}
%%tab pytorch, mxnet, tensorflow
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

```{.python .input  n=23}
%%tab jax
model = RNNLMScratch(rnn, num_inputs)
outputs, _ = model.init_with_output(d2l.get_key(),
                                    d2l.ones((batch_size, num_steps),
                                             dtype=d2l.int32))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**Écrêtage des gradients**]


Bien que vous ayez déjà l'habitude de considérer les réseaux de neurones
comme « profonds » dans le sens où de nombreuses couches
séparent l'entrée et la sortie
même au sein d'un seul pas de temps,
la longueur de la séquence introduit
une nouvelle notion de profondeur.
En plus du passage à travers le réseau
dans le sens entrée-sortie,
les entrées du premier pas de temps
doivent traverser une chaîne de $T$ couches
le long des pas de temps afin
d'influencer la sortie du modèle
au pas de temps final.
En prenant la vue inverse, à chaque itération,
nous rétropropageons les gradients à travers le temps,
ce qui donne une chaîne de produits matriciels
de longueur $\mathcal{O}(T)$.
Comme mentionné dans :numref:`sec_numerical_stability`,
cela peut entraîner une instabilité numérique,
faisant en sorte que les gradients explosent ou disparaissent,
selon les propriétés des matrices de poids.

Gérer la disparition et l'explosion des gradients
est un problème fondamental lors de la conception des RNN
et a inspiré certaines des plus grandes avancées
dans les architectures modernes de réseaux de neurones.
Dans le chapitre suivant, nous parlerons
d'architectures spécialisées conçues
dans l'espoir d'atténuer le problème de disparition du gradient.
Cependant, même les RNN modernes souffrent souvent
d'explosion des gradients.
Une solution peu élégante mais omniprésente
consiste simplement à écrêter les gradients
en forçant les gradients « écrêtés » résultants
à prendre des valeurs plus petites.


De manière générale, lors de l'optimisation d'un objectif
par descente de gradient, nous mettons à jour itérativement
le paramètre d'intérêt, disons un vecteur $\mathbf{x}$,
en le poussant dans la direction du
gradient négatif $\mathbf{g}$
(dans la descente de gradient stochastique,
nous calculons ce gradient
sur un mini-lot échantillonné au hasard).
Par exemple, avec un taux d'apprentissage $\eta > 0$,
chaque mise à jour prend la forme
$\mathbf{x} \gets \mathbf{x} - \eta \mathbf{g}$.
Supposons en outre que la fonction objectif $f$
est suffisamment lisse.
Formellement, nous disons que l'objectif
est *Lipschitzien* avec une constante $L$,
ce qui signifie que pour tout $\mathbf{x}$ et $\mathbf{y}$, nous avons

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

Comme vous pouvez le voir, lorsque nous mettons à jour le vecteur de paramètres en soustrayant $\eta \mathbf{g}$,
le changement de la valeur de l'objectif
dépend du taux d'apprentissage,
de la norme du gradient et de $L$ comme suit :

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|.$$

En d'autres termes, l'objectif ne peut pas
changer de plus de $L \eta \|\mathbf{g}\|$.
Le fait d'avoir une petite valeur pour cette borne supérieure
peut être considéré comme bon ou mauvais.
D'un côté, nous limitons la vitesse
à laquelle nous pouvons réduire la valeur de l'objectif.
D'un autre côté, cela limite l'ampleur
de l'erreur que nous pouvons commettre à n'importe quelle étape du gradient.


Quand on dit que les gradients explosent,
cela signifie que $\|\mathbf{g}\|$
devient excessivement grand.
Dans ce pire des cas, nous pourrions faire tellement de
dégâts en une seule étape de gradient que nous
pourrions annuler tous les progrès réalisés au
cours de milliers d'itérations d'entraînement.
Lorsque les gradients peuvent être aussi importants,
l'entraînement des réseaux de neurones diverge souvent,
échouant à réduire la valeur de l'objectif.
D'autres fois, l'entraînement finit par converger
mais est instable en raison de pics massifs dans la perte.


Une façon de limiter la taille de $L \eta \|\mathbf{g}\|$
est de réduire le taux d'apprentissage $\eta$ à des valeurs infimes.
Cela présente l'avantage de ne pas biaiser les mises à jour.
Mais que se passe-t-il si nous n'obtenons que *rarement* des gradients élevés ?
Cette décision drastique ralentit nos progrès à toutes les étapes,
juste pour faire face aux rares événements d'explosion de gradient.
Une alternative populaire consiste à adopter une heuristique d' *écrêtage de gradient*
en projetant les gradients $\mathbf{g}$ sur une boule
d'un certain rayon donné $\theta$ comme suit :

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

Cela garantit que la norme du gradient ne dépasse jamais $\theta$
et que le gradient mis à jour est entièrement aligné
avec la direction originale de $\mathbf{g}$.
Cela a également l'effet secondaire souhaitable
de limiter l'influence qu'un mini-lot donné
(et en son sein n'importe quel échantillon donné)
peut exercer sur le vecteur de paramètres.
Cela confère un certain degré de robustesse au modèle.
Soyons clairs, c'est une astuce.
L'écrêtage de gradient signifie que nous ne suivons pas toujours
le vrai gradient et il est difficile
de raisonner analytiquement sur les effets secondaires possibles.
Cependant, c'est une astuce très utile,
et elle est largement adoptée dans les implémentations de RNN
dans la plupart des frameworks de deep learning.


Ci-dessous, nous définissons une méthode pour écrêter les gradients,
qui est invoquée par la méthode `fit_epoch` de
la classe `d2l.Trainer` (voir :numref:`sec_linear_scratch`).
Notez que lors du calcul de la norme du gradient,
nous concaténons tous les paramètres du modèle,
les traitant comme un seul vecteur de paramètres géant.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]    
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

```{.python .input  n=27}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)
```

## Entraînement

En utilisant le jeu de données de *La Machine à explorer le temps* (`data`),
nous entraînons un modèle de langage au niveau des caractères (`model`)
basé sur le RNN (`rnn`) implémenté à partir de zéro.
Notez que nous calculons d'abord les gradients,
puis nous les écrêtons, et enfin
nous mettons à jour les paramètres du modèle
en utilisant les gradients écrêtés.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## Décodage

Une fois qu'un modèle de langage a été appris,
nous pouvons l'utiliser non seulement pour prédire le jeton suivant
mais aussi pour continuer à prédire chaque jeton subséquent,
en traitant le jeton précédemment prédit comme s'il
était le suivant dans l'entrée.
Parfois, nous voudrons simplement générer du texte
comme si nous commencions au début
d'un document.
Cependant, il est souvent utile de conditionner
le modèle de langage sur un préfixe fourni par l'utilisateur.
Par exemple, si nous développions une
fonction de saisie semi-automatique pour un moteur de recherche
ou pour aider les utilisateurs à rédiger des e-mails,
nous voudrions fournir ce qu'ils
ont écrit jusqu'à présent (le préfixe),
puis générer une suite probable.


[**La méthode `predict` suivante
génère une suite, un caractère à la fois,
après avoir ingéré un `prefix` fourni par l'utilisateur**].
Lors de la boucle à travers les caractères de `prefix`,
nous continuons à transmettre l'état caché
au pas de temps suivant
mais nous ne générons aucune sortie.
C'est ce qu'on appelle la période de *préchauffage*.
Après avoir ingéré le préfixe, nous sommes maintenant
prêts à commencer à émettre les caractères suivants,
chacun d'entre eux étant réinjecté dans le modèle
comme entrée au pas de temps suivant.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        if tab.selected('mxnet'):
            X = d2l.tensor([[outputs[-1]]], ctx=device)
        if tab.selected('pytorch'):
            X = d2l.tensor([[outputs[-1]]], device=device)
        if tab.selected('tensorflow'):
            X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, params):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn.apply({'params': params['rnn']},
                                            embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.apply({'params': params}, rnn_outputs,
                           method=self.output_layer)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

Dans ce qui suit, nous spécifions le préfixe
et lui faisons générer 20 caractères supplémentaires.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

Bien que l'implémentation du modèle RNN ci-dessus à partir de zéro soit instructive, elle n'est pas pratique.
Dans la section suivante, nous verrons comment tirer parti des frameworks de deep learning pour élaborer rapidement des RNN
en utilisant des architectures standard, et pour obtenir des gains de performance
en s'appuyant sur des fonctions de bibliothèque hautement optimisées.


## Résumé

Nous pouvons entraîner des modèles de langage basés sur des RNN pour générer du texte en suivant le préfixe de texte fourni par l'utilisateur.
Un modèle de langage RNN simple se compose d'un codage d'entrée, d'une modélisation RNN et d'une génération de sortie.
Pendant l'entraînement, l'écrêtage de gradient peut atténuer le problème de l'explosion des gradients mais ne résout pas le problème de la disparition des gradients. Dans l'expérience, nous avons implémenté un modèle de langage RNN simple et l'avons entraîné avec un écrêtage de gradient sur des séquences de texte, découpées en jetons au niveau des caractères. En se conditionnant sur un préfixe, nous pouvons utiliser un modèle de langage pour générer des suites probables, ce qui s'avère utile dans de nombreuses applications, par exemple les fonctions de saisie semi-automatique.


## Exercices

1. Le modèle de langage implémenté prédit-il le jeton suivant en se basant sur tous les jetons passés jusqu'au tout premier jeton de *La Machine à explorer le temps* ?
1. Quel hyperparamètre contrôle la longueur de l'historique utilisé pour la prédiction ?
1. Montrez que le codage one-hot est équivalent au choix d'un plongement différent pour chaque objet.
1. Ajustez les hyperparamètres (par exemple, le nombre d'époques, le nombre d'unités cachées, le nombre de pas de temps dans un mini-lot et le taux d'apprentissage) pour améliorer la perplexité. Jusqu'où pouvez-vous descendre en restant avec cette architecture simple ?
1. Remplacez le codage one-hot par des plongements apprenables. Cela conduit-il à de meilleures performances ?
1. Menez une expérience pour déterminer dans quelle mesure ce modèle de langage
   entraîné sur *La Machine à explorer le temps* fonctionne sur d'autres livres de H. G. Wells,
   par exemple *La Guerre des mondes*.
1. Menez une autre expérience pour évaluer la perplexité de ce modèle
   sur des livres écrits par d'autres auteurs.
1. Modifiez la méthode de prédiction afin d'utiliser l'échantillonnage
   plutôt que de choisir le caractère suivant le plus probable.
    * Que se passe-t-il ?
    * Biaiisez le modèle vers des sorties plus probables, par exemple
    en échantillonnant à partir de $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$ pour $\alpha > 1$.
1. Exécutez le code de cette section sans écrêter le gradient. Que se passe-t-il ?
1. Remplacez la fonction d'activation utilisée dans cette section par ReLU
   et répétez les expériences de cette section. Avons-nous encore besoin d'un écrêtage de gradient ? Pourquoi ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18014)
:end_tab:
