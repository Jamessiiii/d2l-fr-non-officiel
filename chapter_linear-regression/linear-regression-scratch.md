```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Implémentation de la régression linéaire à partir de zéro
:label:`sec_linear_scratch`

Nous sommes maintenant prêts à travailler sur 
une implémentation complète et fonctionnelle 
de la régression linéaire. 
Dans cette section, 
(**nous implémenterons l'intégralité de la méthode à partir de zéro,
y compris (i) le modèle ; (ii) la fonction de perte ;
(iii) un optimiseur par descente de gradient stochastique par mini-lots ;
et (iv) la fonction d'entraînement 
qui assemble toutes ces pièces.**)
Enfin, nous exécuterons notre générateur de données synthétiques
de la :numref:`sec_synthetic-regression-data`
et appliquerons notre modèle
sur le jeu de données résultant. 
Bien que les frameworks de deep learning modernes 
puissent automatiser presque tout ce travail,
implémenter les choses à partir de zéro est le seul moyen
de s'assurer que vous savez vraiment ce que vous faites.
De plus, lorsqu'il sera temps de personnaliser les modèles,
de définir nos propres couches ou fonctions de perte,
comprendre comment les choses fonctionnent sous le capot s'avérera utile.
Dans cette section, nous nous appuierons uniquement 
sur les tenseurs et la différentiation automatique.
Plus tard, nous introduirons une implémentation plus concise,
profitant des options avancées des frameworks de deep learning 
tout en conservant la structure de ce qui suit ci-dessous.

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## Définir le modèle

[**Avant de pouvoir commencer à optimiser les paramètres de notre modèle**] par SGD par mini-lots,
(**nous devons d'abord avoir des paramètres.**)
Dans ce qui suit, nous initialisons les poids en tirant
des nombres aléatoires d'une distribution normale de moyenne 0
et d'un écart-type de 0,01. 
Le nombre magique 0,01 fonctionne souvent bien en pratique, 
mais vous pouvez spécifier une valeur différente 
via l'argument `sigma`.
De plus, nous fixons le biais à 0.
Notez que pour la conception orientée objet,
nous ajoutons le code à la méthode `__init__` d'une sous-classe de `d2l.Module` (introduite dans la :numref:`subsec_oo-design-models`).

```{.python .input  n=6}
%%tab pytorch, mxnet, tensorflow
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

```{.python .input  n=7}
%%tab jax
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))
```

Ensuite, nous devons [**définir notre modèle,
en reliant son entrée et ses paramètres à sa sortie.**]
En utilisant la même notation que dans l' :eqref:`eq_linreg-y-vec`
pour notre modèle linéaire, nous prenons simplement le produit matrice-vecteur
des caractéristiques d'entrée $\mathbf{X}$ 
et des poids du modèle $\mathbf{w}$,
et ajoutons le décalage $b$ à chaque exemple.
Le produit $\mathbf{Xw}$ est un vecteur et $b$ est un scalaire.
En raison du mécanisme de diffusion 
(voir la :numref:`subsec_broadcasting`),
lorsque nous ajoutons un vecteur et un scalaire,
le scalaire est ajouté à chaque composante du vecteur.
La méthode `forward` résultante 
est enregistrée dans la classe `LinearRegressionScratch`
via `add_to_class` (introduit dans la :numref:`oo-design-utilities`).

```{.python .input  n=8}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## Définir la fonction de perte

Puisque [**la mise à jour de notre modèle nécessite de prendre
le gradient de notre fonction de perte,**]
nous devrions (**définir d'abord la fonction de perte.**)
Ici, nous utilisons la fonction de perte quadratique
de l' :eqref:`eq_mse`.
Dans l'implémentation, nous devons transformer la valeur réelle `y`
dans la forme de la valeur prédite `y_hat`.
Le résultat renvoyé par la méthode suivante
aura également la même forme que `y_hat`. 
Nous renvoyons également la valeur de perte moyenne
parmi tous les exemples du mini-lot.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return d2l.reduce_mean(l)
```

```{.python .input  n=10}
%%tab jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)  # X unpacked from a tuple
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## Définir l'algorithme d'optimisation

Comme discuté dans la :numref:`sec_linear_regression`,
la régression linéaire possède une solution analytique.
Cependant, notre objectif ici est d'illustrer 
comment entraîner des réseaux de neurones plus généraux,
et cela nécessite que nous vous enseignions 
comment utiliser la SGD par mini-lots.
Par conséquent, nous saisirons cette opportunité
pour introduire votre premier exemple fonctionnel de SGD.
À chaque étape, en utilisant un mini-lot 
tiré au hasard de notre jeu de données,
nous estimons le gradient de la perte
par rapport aux paramètres.
Ensuite, nous mettons à jour les paramètres
dans la direction qui peut réduire la perte.

Le code suivant applique la mise à jour, 
étant donné un ensemble de paramètres et un taux d'apprentissage `lr`.
Puisque notre perte est calculée comme une moyenne sur le mini-lot, 
nous n'avons pas besoin d'ajuster le taux d'apprentissage en fonction de la taille du lot. 
Dans les chapitres suivants, nous étudierons 
comment les taux d'apprentissage devraient être ajustés
pour de très grands mini-lots tels qu'ils apparaissent 
dans l'apprentissage distribué à grande échelle.
Pour l'instant, nous pouvons ignorer cette dépendance.

:begin_tab:`mxnet`
Nous définissons notre classe `SGD`, 
une sous-classe de `d2l.HyperParameters` (introduite dans la :numref:`oo-design-utilities`),
pour avoir une API similaire
à l'optimiseur SGD intégré.
Nous mettons à jour les paramètres dans la méthode `step`.
Elle accepte un argument `batch_size` qui peut être ignoré.
:end_tab:

:begin_tab:`pytorch`
Nous définissons notre classe `SGD`,
une sous-classe de `d2l.HyperParameters` (introduite dans la :numref:`oo-design-utilities`),
pour avoir une API similaire 
à l'optimiseur SGD intégré.
Nous mettons à jour les paramètres dans la méthode `step`.
La méthode `zero_grad` met tous les gradients à 0,
ce qui doit être exécuté avant une étape de rétropropagation.
:end_tab:

:begin_tab:`tensorflow`
Nous définissons notre classe `SGD`,
une sous-classe de `d2l.HyperParameters` (introduite dans la :numref:`oo-design-utilities`),
pour avoir une API similaire
à l'optimiseur SGD intégré.
Nous mettons à jour les paramètres dans la méthode `apply_gradients`.
Elle accepte une liste de paires de paramètres et de gradients.
:end_tab:

```{.python .input  n=11}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad

    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=12}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input  n=13}
%%tab jax
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    # The key transformation of Optax is the GradientTransformation
    # defined by two methods, the init and the update.
    # The init initializes the state and the update transforms the gradients.
    # https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        # Delete unused params
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # When state.apply_gradients method is called to update flax's
        # train_state object, it internally calls optax.apply_updates method
        # adding the params to the update equation defined below.
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__():
        return optax.GradientTransformation(self.init, self.update)
```

Nous définissons ensuite la méthode `configure_optimizers`, qui renvoie une instance de la classe `SGD`.

```{.python .input  n=14}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow', 'jax'):
        return SGD(self.lr)
```

## Entraînement

Maintenant que nous avons toutes les pièces en place
(paramètres, fonction de perte, modèle et optimiseur),
nous sommes prêts à [**implémenter la boucle d'entraînement principale.**]
Il est crucial que vous compreniez parfaitement ce code,
car vous utiliserez des boucles d'entraînement similaires
pour tous les autres modèles de deep learning
abordés dans ce livre.
Dans chaque *époque*, nous itérons à travers 
l'ensemble du jeu de données d'entraînement, 
en passant une fois par chaque exemple
(en supposant que le nombre d'exemples 
est divisible par la taille du lot). 
À chaque *itération*, nous récupérons un mini-lot d'exemples d'entraînement
et calculons sa perte via la méthode `training_step` du modèle. 
Ensuite, nous calculons les gradients par rapport à chaque paramètre. 
Enfin, nous appellerons l'algorithme d'optimisation
pour mettre à jour les paramètres du modèle. 
En résumé, nous exécuterons la boucle suivante :

* Initialiser les paramètres $(\mathbf{w}, b)$
* Répéter jusqu'à la fin
    * Calculer le gradient $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Mettre à jour les paramètres $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$
 
Rappelez-vous que le jeu de données de régression synthétique 
que nous avons généré dans la :numref:``sec_synthetic-regression-data`` 
ne fournit pas de jeu de données de validation. 
Dans la plupart des cas, cependant, 
nous voudrons un jeu de données de validation 
pour mesurer la qualité de notre modèle. 
Ici, nous passons le chargeur de données de validation 
une fois par époque pour mesurer la performance du modèle.
Suivant notre conception orientée objet,
les méthodes `prepare_batch` et `fit_epoch` 
sont enregistrées dans la classe `d2l.Trainer`
(introduite dans la :numref:`oo-design-training`).

```{.python .input  n=15}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=16}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=17}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=18}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=19}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    if self.state.batch_stats:
        # Mutable states will be used later (e.g., for batch norm)
        for batch in self.train_dataloader:
            (_, mutated_vars), grads = self.model.training_step(self.state.params,
                                                           self.prepare_batch(batch),
                                                           self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
            self.train_batch_idx += 1
    else:
        for batch in self.train_dataloader:
            _, grads = self.model.training_step(self.state.params,
                                                self.prepare_batch(batch),
                                                self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.train_batch_idx += 1

    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        self.model.validation_step(self.state.params,
                                   self.prepare_batch(batch),
                                   self.state)
        self.val_batch_idx += 1
```

Nous sommes presque prêts à entraîner le modèle,
mais nous avons d'abord besoin de données d'entraînement.
Ici, nous utilisons la classe `SyntheticRegressionData` 
et passons certains paramètres de vérité terrain.
Ensuite, nous entraînons notre modèle avec 
le taux d'apprentissage `lr=0,03` 
et fixons `max_epochs=3`. 
Notez qu'en général, le nombre d'époques 
et le taux d'apprentissage sont tous deux des hyperparamètres.
En général, le réglage des hyperparamètres est délicat
et nous voudrons généralement utiliser une séparation en trois parties,
un ensemble pour l'entraînement, 
un deuxième pour la sélection des hyperparamètres,
et le troisième réservé à l'évaluation finale.
Nous omettons ces détails pour l'instant mais nous y reviendrons
plus tard.

```{.python .input  n=20}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Parce que nous avons nous-mêmes synthétisé le jeu de données,
nous savons précisément quels sont les vrais paramètres.
Ainsi, nous pouvons [**évaluer le succès de notre entraînement
en comparant les vrais paramètres
avec ceux que nous avons appris**] via notre boucle d'entraînement.
En effet, ils s'avèrent être très proches les uns des autres.

```{.python .input  n=21}
%%tab pytorch
with torch.no_grad():
    print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=22}
%%tab mxnet, tensorflow
print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=23}
%%tab jax
params = trainer.state.params
print(f"error in estimating w: {data.w - d2l.reshape(params['w'], data.w.shape)}")
print(f"error in estimating b: {data.b - params['b']}")
```

On ne devrait pas tenir pour acquise la capacité à retrouver 
exactement les paramètres de la vérité terrain.
En général, pour les modèles profonds, il n'existe pas de solutions uniques
pour les paramètres,
et même pour les modèles linéaires,
retrouver exactement les paramètres
n'est possible que lorsqu'aucune caractéristique 
n'est linéairement dépendante des autres.
Cependant, en apprentissage automatique, 
nous sommes souvent moins préoccupés
par la récupération des vrais paramètres sous-jacents,
mais plutôt par des paramètres 
qui conduisent à une prédiction hautement précise :cite:`Vapnik.1992`.
Heureusement, même sur des problèmes d'optimisation difficiles,
la descente de gradient stochastique peut souvent trouver des solutions remarquablement bonnes,
en partie grâce au fait que, pour les réseaux profonds,
il existe de nombreuses configurations des paramètres
qui conduisent à une prédiction hautement précise.


## Résumé

Dans cette section, nous avons franchi une étape importante 
vers la conception de systèmes de deep learning 
en implémentant un modèle de réseau de neurones 
et une boucle d'entraînement entièrement fonctionnels.
Au cours de ce processus, nous avons construit un chargeur de données, 
un modèle, une fonction de perte, une procédure d'optimisation,
ainsi qu'un outil de visualisation et de surveillance. 
Nous avons fait cela en composant un objet Python 
qui contient tous les composants pertinents pour l'entraînement d'un modèle. 
Bien qu'il ne s'agisse pas encore d'une implémentation de qualité professionnelle,
elle est parfaitement fonctionnelle et un code comme celui-ci 
pourrait déjà vous aider à résoudre rapidement de petits problèmes.
Dans les sections à venir, nous verrons comment faire cela
à la fois de manière *plus concise* (en évitant le code répétitif)
et de manière *plus efficace* (en utilisant nos GPU à leur plein potentiel).



## Exercices

1. Que se passerait-il si nous devions initialiser les poids à zéro. L'algorithme fonctionnerait-il toujours ? Et si nous
   initialisions les paramètres avec une variance de $1000$ plutôt que $0,01$ ?
1. Supposez que vous soyez [Georg Simon Ohm](https://fr.wikipedia.org/wiki/Georg_Ohm) essayant de concevoir
   un modèle pour la résistance qui relie la tension et le courant. Pouvez-vous utiliser la différentiation
   automatique pour apprendre les paramètres de votre modèle ?
1. Pouvez-vous utiliser la [loi de Planck](https://fr.wikipedia.org/wiki/Loi_de_Planck) pour déterminer la température d'un objet
   en utilisant la densité spectrale d'énergie ? Pour référence, la densité spectrale $B$ du rayonnement émanant d'un corps noir est
   $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$. Ici,
   $\lambda$ est la longueur d'onde, $T$ est la température, $c$ est la vitesse de la lumière, $h$ est la constante de Planck, et $k$ est la
   constante de Boltzmann. Vous mesurez l'énergie pour différentes longueurs d'onde $\lambda$ et vous devez maintenant ajuster la courbe de
   densité spectrale à la loi de Planck.
1. Quels sont les problèmes que vous pourriez rencontrer si vous vouliez calculer les dérivées secondes de la perte ? Comment les
   fixeriez-vous ?
1. Pourquoi la méthode `reshape` est-elle nécessaire dans la fonction `loss` ?
1. Expérimentez en utilisant différents taux d'apprentissage pour découvrir à quelle vitesse la valeur de la fonction de perte chute. Pouvez-vous réduire
   l'erreur en augmentant le nombre d'époques d'entraînement ?
1. Si le nombre d'exemples ne peut pas être divisé par la taille du lot, qu'arrive-t-il à `data_iter` à la fin d'une époque ?
1. Essayez d'implémenter une fonction de perte différente, telle que la perte en valeur absolue `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`.
    1. Vérifiez ce qui se passe pour des données régulières.
    2. Vérifiez s'il y a une différence de comportement si vous perturbez activement certaines entrées, telles que $y_5 = 10000$, de $\mathbf{y}$.
    3. Pouvez-vous imaginer une solution simple pour combiner les meilleurs aspects de la perte quadratique et de la perte en valeur absolue ?
       Indice : comment pouvez-vous éviter des valeurs de gradient vraiment importantes ?
1. Pourquoi devons-nous remélanger le jeu de données ? Pouvez-vous concevoir un cas où un jeu de données construit de manière malveillante briserait l'algorithme d'optimisation autrement ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17976)
:end_tab:
