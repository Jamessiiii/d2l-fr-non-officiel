```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Normalisation par lots
:label:`sec_batch_norm`

L'entraînement de réseaux de neurones profonds est difficile.
Réussir à les faire converger dans un temps raisonnable peut être délicat.
Dans cette section, nous décrivons la *normalisation par lots* (batch normalization), une technique populaire et efficace
qui accélère systématiquement la convergence des réseaux profonds :cite:`Ioffe.Szegedy.2015`.
Avec les blocs résiduels --- abordés plus tard dans la :numref:`sec_resnet` --- la normalisation par lots
a permis aux praticiens d'entraîner couramment des réseaux de plus de 100 couches.
Un avantage secondaire (fortuit) de la normalisation par lots réside dans sa régularisation intrinsèque.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
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
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## Entraîner des réseaux profonds

Lorsque nous travaillons avec des données, nous effectuons souvent un prétraitement avant l'entraînement.
Les choix concernant le prétraitement des données font souvent une énorme différence dans les résultats finaux.
Rappelez-vous notre application des MLP à la prédiction des prix des maisons (:numref:`sec_kaggle_house`).
Notre première étape lors du travail avec des données réelles
était de normaliser nos caractéristiques d'entrée pour avoir
une moyenne nulle $\boldsymbol{\mu} = 0$ et une variance unitaire $\boldsymbol{\Sigma} = \boldsymbol{1}$ à travers plusieurs observations :cite:`friedman1987exploratory`, en redimensionnant fréquemment cette dernière pour que la diagonale soit l'unité, c'est-à-dire $\Sigma_{ii} = 1$.
Une autre stratégie consiste à redimensionner les vecteurs à une longueur unitaire, éventuellement avec une moyenne nulle *par observation*.
Cela peut bien fonctionner, par exemple, pour les données de capteurs spatiaux. Ces techniques de prétraitement et bien d'autres sont
bénéfiques pour garder le problème d'estimation bien contrôlé. 
Pour une revue de la sélection et de l'extraction de caractéristiques, voir l'article de :citet:`guyon2008feature`, par exemple.
La normalisation des vecteurs a également l'effet secondaire agréable de contraindre la complexité des fonctions qui agissent sur eux. Par exemple, la célèbre borne rayon-marge :cite:`Vapnik95` dans les machines à vecteurs de support (SVM) et le théorème de convergence du Perceptron :cite:`Novikoff62` reposent sur des entrées de norme bornée. 

Intuitivement, cette normalisation s'accorde bien avec nos optimiseurs
car elle place les paramètres *a priori* sur une échelle similaire.
Il est donc tout naturel de se demander si une étape de normalisation correspondante *à l'intérieur* d'un réseau profond
ne pourrait pas être bénéfique. Bien que ce ne soit pas tout à fait le raisonnement qui a conduit à l'invention de la normalisation par lots :cite:`Ioffe.Szegedy.2015`, c'est une façon utile de la comprendre, ainsi que sa cousine, la normalisation par couche :cite:`Ba.Kiros.Hinton.2016`, au sein d'un cadre unifié.

Deuxièmement, pour un MLP ou un CNN typique, au fur et à mesure que nous entraînons,
les variables 
dans les couches intermédiaires (par exemple, les sorties des transformations affines dans un MLP)
peuvent prendre des valeurs avec des magnitudes variant considérablement :
que ce soit le long des couches de l'entrée vers la sortie, à travers les unités d'une même couche,
et au fil du temps en raison de nos mises à jour des paramètres du modèle.
Les inventeurs de la normalisation par lots ont postulé de manière informelle
que cette dérive dans la distribution de ces variables pourrait entraver la convergence du réseau.
Intuitivement, nous pourrions conjecturer que si une
couche possède des activations de variables 100 fois supérieures à celles d'une autre couche,
cela pourrait nécessiter des ajustements compensatoires dans les taux d'apprentissage. Des solveurs adaptatifs
tels qu'AdaGrad :cite:`Duchi.Hazan.Singer.2011`, Adam :cite:`Kingma.Ba.2014`, Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018`, ou Distributed Shampoo :cite:`anil2020scalable` visent à traiter cela du point de vue de l'optimisation, par exemple en ajoutant des aspects des méthodes de second ordre. 
L'alternative est d'empêcher le problème de se produire, simplement par une normalisation adaptative.

Troisièmement, les réseaux plus profonds sont complexes et ont tendance à être plus sujets au surapprentissage.
Cela signifie que la régularisation devient plus critique. Une technique courante de régularisation est l'injection de bruit.
Cela est connu depuis longtemps, par exemple en ce qui concerne l'injection de bruit pour les
entrées :cite:`Bishop.1995`. Cela constitue également la base du dropout dans la :numref:`sec_dropout`. Il s'avère que, de manière tout à fait fortuite, la normalisation par lots apporte ces trois avantages : prétraitement, stabilité numérique et régularisation.

La normalisation par lots est appliquée à des couches individuelles, ou optionnellement, à toutes les couches :
À chaque itération d'entraînement,
nous normalisons d'abord les entrées (de la normalisation par lots)
en soustrayant leur moyenne et
en les divisant par leur écart-type,
où les deux sont estimés sur la base des statistiques du mini-lot actuel.
Ensuite, nous appliquons un coefficient d'échelle et un décalage pour récupérer les degrés
de liberté perdus. C'est précisément en raison de cette *normalisation* basée sur les statistiques de *lot*
que la *normalisation par lots* tire son nom.

Notez que si nous essayions d'appliquer la normalisation par lots avec des mini-lots de taille 1,
nous ne pourrions rien apprendre.
C'est parce qu'après avoir soustrait les moyennes,
chaque unité cachée prendrait la valeur 0.
Comme vous pouvez le deviner, puisque nous consacrons une section entière à la normalisation par lots,
avec des mini-lots suffisamment grands, l'approche s'avère efficace et stable.
Un point à retenir ici est que lors de l'application de la normalisation par lots,
le choix de la taille du lot est
encore plus significatif que sans normalisation par lots, ou du moins,
un étalonnage approprié est nécessaire car nous pourrions ajuster la taille du lot.

Notons par $\mathcal{B}$ un mini-lot et soit $\mathbf{x} \in \mathcal{B}$ une entrée de 
la normalisation par lots ($\textrm{BN}$). Dans ce cas, la normalisation par lots est définie comme suit :

$$\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

Dans l' :eqref:`eq_batchnorm`,
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ est la moyenne de l'échantillon
et $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ est l'écart-type de l'échantillon du mini-lot $\mathcal{B}$.
Après avoir appliqué la normalisation,
le mini-lot résultant
a une moyenne nulle et une variance unitaire.
Le choix d'une variance unitaire
(plutôt qu'un autre nombre magique) est arbitraire. Nous récupérons ce degré de liberté
en incluant un *paramètre d'échelle* $\boldsymbol{\gamma}$ et un *paramètre de décalage* $\boldsymbol{\beta}$
appliqués élément par élément, qui ont la même forme que $\mathbf{x}$. Les deux sont des paramètres qui
doivent être appris dans le cadre de l'entraînement du modèle.

Les magnitudes des variables
pour les couches intermédiaires ne peuvent pas diverger pendant l'entraînement
car la normalisation par lots les centre activement et les redimensionne
selon une moyenne et une taille données (via $\hat{\boldsymbol{\mu}}_\mathcal{B}$ et ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$).
L'expérience pratique confirme que, comme suggéré lors de la discussion sur le redimensionnement des caractéristiques, la normalisation par lots semble permettre des taux d'apprentissage plus agressifs.
Nous calculons $\hat{\boldsymbol{\mu}}_\mathcal{B}$ et ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ dans l' :eqref:`eq_batchnorm` comme suit :

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ et }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

Notez que nous ajoutons une petite constante $\epsilon > 0$
à l'estimation de la variance
pour garantir que nous ne tentons jamais une division par zéro,
même dans les cas où l'estimation de la variance empirique pourrait être très petite ou nulle.
Les estimations $\hat{\boldsymbol{\mu}}_\mathcal{B}$ et ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ neutralisent le problème d'échelle
en utilisant des estimations bruitées de la moyenne et de la variance.
Vous pourriez penser que ce bruit devrait être un problème.
Au contraire, il est en fait bénéfique.

Cela s'avère être un thème récurrent dans l'apprentissage profond.
Pour des raisons qui ne sont pas encore bien caractérisées théoriquement,
diverses sources de bruit dans l'optimisation
conduisent souvent à un entraînement plus rapide et à moins de surapprentissage :
cette variation semble agir comme une forme de régularisation.
:citet:`Teye.Azizpour.Smith.2018` et :citet:`Luo.Wang.Shao.ea.2018`
ont relié les propriétés de la normalisation par lots aux priors bayésiens et aux pénalités, respectivement. 
En particulier, cela apporte un certain éclairage sur l'énigme
de savoir pourquoi la normalisation par lots fonctionne mieux pour des tailles de mini-lot modérées, de l'ordre de 50 à 100.
Cette taille particulière de mini-lot semble injecter juste la "bonne quantité" de bruit par couche, tant en termes d'échelle via $\hat{\boldsymbol{\sigma}}$ qu'en termes de décalage via $\hat{\boldsymbol{\mu}}$ : un
mini-lot plus grand régularise moins en raison d'estimations plus stables, tandis que de tout petits mini-lots
détruisent le signal utile en raison d'une variance élevée. En explorant davantage cette direction, l'examen d'autres types
de prétraitement et de filtrage pourrait encore mener à d'autres types de régularisation efficaces.

En fixant un modèle entraîné, vous pourriez penser
que nous préférerions utiliser l'ensemble du jeu de données
pour estimer la moyenne et la variance.
Une fois l'entraînement terminé, pourquoi voudrions-nous
que la même image soit classée différemment,
selon le lot dans lequel elle se trouve ?
Pendant l'entraînement, un tel calcul exact est irréalisable
car les variables intermédiaires
pour tous les exemples de données
changent à chaque fois que nous mettons à jour notre modèle.
Cependant, une fois le modèle entraîné,
nous pouvons calculer les moyennes et les variances
des variables de chaque couche sur la base de l'ensemble du jeu de données.
C'est en effet une pratique courante pour les
modèles employant la normalisation par lots ;
ainsi, les couches de normalisation par lots fonctionnent différemment
en *mode entraînement* (normalisation par les statistiques du mini-lot)
qu'en *mode prédiction* (normalisation par les statistiques du jeu de données).
Sous cette forme, elles ressemblent étroitement au comportement de la régularisation par dropout de la :numref:`sec_dropout`,
où le bruit n'est injecté que pendant l'entraînement.


## Couches de normalisation par lots

Les implémentations de la normalisation par lots pour les couches entièrement connectées
et les couches convolutives sont légèrement différentes.
Une différence clé entre la normalisation par lots et les autres couches
est que, puisque la première opère sur un mini-lot complet à la fois,
nous ne pouvons pas simplement ignorer la dimension du lot
comme nous le faisions auparavant lors de l'introduction d'autres couches.

### Couches entièrement connectées

Lors de l'application de la normalisation par lots aux couches entièrement connectées,
:citet:`Ioffe.Szegedy.2015`, dans leur article original, ont inséré la normalisation par lots après la transformation affine
et *avant* la fonction d'activation non linéaire. Des applications ultérieures ont expérimenté l'insertion de la normalisation par lots juste *après* les fonctions d'activation.
En notant $\mathbf{x}$ l'entrée de la couche entièrement connectée,
$\mathbf{W}\mathbf{x} + \mathbf{b}$ la transformation affine
(avec le paramètre de poids $\mathbf{W}$ et le paramètre de biais $\mathbf{b}$),
et $\phi$ la fonction d'activation,
nous pouvons exprimer le calcul de la sortie $\mathbf{h}$ d'une couche entièrement connectée avec normalisation par lots comme suit :

$$\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Rappelez-vous que la moyenne et la variance sont calculées
sur le *même* mini-lot
sur lequel la transformation est appliquée.

### Couches convolutives

De même, avec les couches convolutives,
nous pouvons appliquer la normalisation par lots après la convolution
mais avant la fonction d'activation non linéaire. La principale différence par rapport à la normalisation par lots
dans les couches entièrement connectées est que nous appliquons l'opération sur une base par canal
*sur tous les emplacements*. Cela est compatible avec notre hypothèse d'invariance par translation
qui a conduit aux convolutions : nous avons supposé que l'emplacement spécifique d'un motif
dans une image n'était pas critique pour la compréhension.

Supposons que nos mini-lots contiennent $m$ exemples
et que pour chaque canal,
la sortie de la convolution ait une hauteur $p$ et une largeur $q$.
Pour les couches convolutives, nous effectuons chaque normalisation par lots
simultanément sur les $m \cdot p \cdot q$ éléments par canal de sortie.
Ainsi, nous collectons les valeurs sur tous les emplacements spatiaux
lors du calcul de la moyenne et de la variance
et appliquons par conséquent
la même moyenne et la même variance
au sein d'un canal donné
pour normaliser la valeur à chaque emplacement spatial.
Chaque canal a ses propres paramètres d'échelle et de décalage,
qui sont tous deux des scalaires.

### Normalisation par couche
:label:`subsec_layer-normalization-in-bn`

Notez que dans le contexte des convolutions, la normalisation par lots est bien définie même pour
des mini-lots de taille 1 : après tout, nous avons tous les emplacements à travers une image pour faire la moyenne. Par conséquent,
la moyenne et la variance sont bien définies, même s'il ne s'agit que d'une seule observation. Cette considération
a conduit :citet:`Ba.Kiros.Hinton.2016` à introduire la notion de *normalisation par couche* (layer normalization). Elle fonctionne exactement comme
une normalisation par lots, si ce n'est qu'elle est appliquée à une seule observation à la fois. Par conséquent, le décalage et le facteur d'échelle sont tous deux des scalaires. Pour un vecteur $\mathbf{x}$ de dimension $n$, les normalisations par couche sont données par 

$$\mathbf{x} \rightarrow \textrm{LN}(\mathbf{x}) =  \frac{\mathbf{x} - \hat{\mu}}{\hat\sigma},$$

où l'échelle et le décalage sont appliqués coefficient par coefficient
et donnés par 

$$\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ et }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.$$

Comme précédemment, nous ajoutons un petit décalage $\epsilon > 0$ pour éviter la division par zéro. L'un des avantages majeurs de l'utilisation de la normalisation par couche est qu'elle empêche la divergence. Après tout, en ignorant $\epsilon$, la sortie de la normalisation par couche est indépendante de l'échelle. C'est-à-dire que nous avons $\textrm{LN}(\mathbf{x}) \approx \textrm{LN}(\alpha \mathbf{x})$ pour tout choix de $\alpha \neq 0$. Cela devient une égalité pour $|\alpha| \to \infty$ (l'égalité approximative est due au décalage $\epsilon$ pour la variance). 

Un autre avantage de la normalisation par couche est qu'elle ne dépend pas de la taille du mini-lot. Elle est également indépendante de savoir si nous sommes en régime d'entraînement ou de test. En d'autres termes, il s'agit simplement d'une transformation déterministe qui normalise les activations à une échelle donnée. Cela peut être très bénéfique pour prévenir la divergence dans l'optimisation. Nous passons sur les détails supplémentaires et recommandons aux lecteurs intéressés de consulter l'article original.

### Normalisation par lots pendant la prédiction

Comme nous l'avons mentionné précédemment, la normalisation par lots se comporte généralement différemment
en mode entraînement qu'en mode prédiction.
Premièrement, le bruit dans la moyenne et la variance de l'échantillon
provenant de l'estimation de chacune sur des mini-lots
n'est plus souhaitable une fois que nous avons entraîné le modèle.
Deuxièmement, nous n'avons peut-être pas le luxe
de calculer des statistiques de normalisation par lot.
Par exemple,
nous pourrions avoir besoin d'appliquer notre modèle pour faire une prédiction à la fois.

Généralement, après l'entraînement, nous utilisons l'ensemble du jeu de données
pour calculer des estimations stables des statistiques des variables
puis nous les fixons au moment de la prédiction.
Par conséquent, la normalisation par lots se comporte différemment pendant l'entraînement qu'au moment du test.
Rappelez-vous que le dropout présente également cette caractéristique.

## (**Implémentation à partir de zéro**)

Pour voir comment la normalisation par lots fonctionne en pratique, nous en implémentons une à partir de zéro ci-dessous.

```{.python .input}
%%tab mxnet
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use autograd to determine whether we are in training mode
    if not autograd.is_training():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
%%tab pytorch
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
%%tab tensorflow
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input}
%%tab jax
def batch_norm(X, deterministic, gamma, beta, moving_mean, moving_var, eps,
               momentum):
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    if deterministic:
        # In prediction mode, use mean and variance obtained by moving average
        # `linen.Module.variables` have a `value` attribute containing the array
        X_hat = (X - moving_mean.value) / jnp.sqrt(moving_var.value + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / jnp.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean.value = momentum * moving_mean.value + (1.0 - momentum) * mean
        moving_var.value = momentum * moving_var.value + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y
```

Nous pouvons maintenant [**créer une véritable couche `BatchNorm`.**]
Notre couche maintiendra les paramètres appropriés
pour l'échelle `gamma` et le décalage `beta`,
qui seront tous deux mis à jour au cours de l'entraînement.
De plus, notre couche maintiendra
des moyennes mobiles des moyennes et des variances
pour une utilisation ultérieure lors de la prédiction du modèle.

En mettant de côté les détails algorithmiques,
notez le modèle de conception sous-jacent à notre implémentation de la couche.
Généralement, nous définissons les mathématiques dans une fonction séparée, par exemple `batch_norm`.
Nous intégrons ensuite cette fonctionnalité dans une couche personnalisée,
dont le code s'occupe principalement des questions de gestion,
telles que le déplacement des données vers le bon contexte de périphérique,
l'allocation et l'initialisation des variables requises,
le suivi des moyennes mobiles (ici pour la moyenne et la variance), et ainsi de suite.
Ce modèle permet une séparation claire des mathématiques du code répétitif.
Notez également que, par souci de commodité,
nous ne nous sommes pas souciés d'inférer automatiquement la forme de l'entrée ici ;
nous devons donc spécifier le nombre de caractéristiques partout.
Désormais, tous les frameworks d'apprentissage profond modernes proposent une détection automatique de la taille et de la forme dans les
API de normalisation par lots de haut niveau (en pratique, nous utiliserons plutôt celles-ci).

```{.python .input}
%%tab mxnet
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.1)
        return Y
```

```{.python .input}
%%tab pytorch
class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```

```{.python .input}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.1
        delta = (1.0 - momentum) * variable + momentum * value
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input}
%%tab jax
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer.
    # `num_dims`: 2 for a fully connected layer and 4 for a convolutional layer
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    num_features: int
    num_dims: int
    deterministic: bool = False

    @nn.compact
    def __call__(self, X):
        if self.num_dims == 2:
            shape = (1, self.num_features)
        else:
            shape = (1, 1, 1, self.num_features)

        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        gamma = self.param('gamma', jax.nn.initializers.ones, shape)
        beta = self.param('beta', jax.nn.initializers.zeros, shape)

        # The variables that are not model parameters are initialized to 0 and
        # 1. Save them to the 'batch_stats' collection
        moving_mean = self.variable('batch_stats', 'moving_mean', jnp.zeros, shape)
        moving_var = self.variable('batch_stats', 'moving_var', jnp.ones, shape)
        Y = batch_norm(X, self.deterministic, gamma, beta,
                       moving_mean, moving_var, eps=1e-5, momentum=0.9)

        return Y
```

Nous avons utilisé `momentum` pour régir l'agrégation sur les estimations passées de la moyenne et de la variance. C'est un peu un abus de langage car cela n'a rien à voir avec le terme de *momentum* de l'optimisation. Néanmoins, c'est le nom couramment adopté pour ce terme et, par respect pour la convention de nommage des API, nous utilisons le même nom de variable dans notre code.

## [**LeNet avec normalisation par lots**]

Pour voir comment appliquer `BatchNorm` en contexte,
nous l'appliquons ci-dessous à un modèle LeNet traditionnel (:numref:`sec_lenet`).
Rappelez-vous que la normalisation par lots est appliquée
après les couches convolutives ou les couches entièrement connectées
mais avant les fonctions d'activation correspondantes.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2), nn.Dense(120),
                BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
                nn.Dense(84), BatchNorm(84, num_dims=2),
                nn.Activation('sigmoid'), nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120),
                BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
                BatchNorm(84, num_dims=2), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84), BatchNorm(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNetScratch(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            BatchNorm(6, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            BatchNorm(16, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            BatchNorm(120, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(84),
            BatchNorm(84, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

:begin_tab:`jax`
Étant donné que les couches `BatchNorm` doivent calculer les statistiques de lot
(moyenne et variance), Flax assure le suivi du dictionnaire `batch_stats`, en les mettant à jour
avec chaque mini-lot. Les collections comme `batch_stats` peuvent être stockées dans l'objet
`TrainState` (dans la classe `d2l.Trainer` définie dans la
:numref:`oo-design-training`) comme un attribut et, lors de la passe avant du modèle,
celles-ci doivent être passées à l'argument `mutable`, afin que Flax renvoie les variables
mutées.
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat, updates = state.apply_fn({'params': params,
                                     'batch_stats': state.batch_stats},
                                    *X, mutable=['batch_stats'],
                                    rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)
```

Comme précédemment, nous allons [**entraîner notre réseau sur le jeu de données Fashion-MNIST**].
Ce code est pratiquement identique à celui de la première fois que nous avons entraîné LeNet.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNetScratch(lr=0.5)
    trainer.fit(model, data)
```

Jetons [**un coup d'œil au paramètre d'échelle `gamma`
et au paramètre de décalage `beta`**] appris
de la première couche de normalisation par lots.

```{.python .input}
%%tab mxnet
model.net[1].gamma.data().reshape(-1,), model.net[1].beta.data().reshape(-1,)
```

```{.python .input}
%%tab pytorch
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

```{.python .input}
%%tab tensorflow
tf.reshape(model.net.layers[1].gamma, (-1,)), tf.reshape(
    model.net.layers[1].beta, (-1,))
```

```{.python .input}
%%tab jax
trainer.state.params['net']['layers_1']['gamma'].reshape((-1,)), \
trainer.state.params['net']['layers_1']['beta'].reshape((-1,))
```

## [**Implémentation concise**]

Par rapport à la classe `BatchNorm` que nous venons de définir nous-mêmes,
nous pouvons utiliser directement la classe `BatchNorm` définie dans les API de haut niveau du framework d'apprentissage profond.
Le code semble pratiquement identique
à notre implémentation ci-dessus, sauf que nous n'avons plus besoin de fournir des arguments supplémentaires pour que les dimensions soient correctes.

```{.python .input}
%%tab pytorch, tensorflow, mxnet
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(84),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

Ci-dessous, nous [**utilisons les mêmes hyperparamètres pour entraîner notre modèle.**]
Notez que, comme d'habitude, la variante de l'API de haut niveau s'exécute beaucoup plus rapidement
car son code a été compilé en C++ ou CUDA
tandis que notre implémentation personnalisée doit être interprétée par Python.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNet(lr=0.5)
    trainer.fit(model, data)
```

## Discussion

Intuitivement, on pense que la normalisation par lots rend le paysage d'optimisation plus lisse.
Cependant, nous devons faire attention à distinguer les intuitions spéculatives des véritables explications
des phénomènes que nous observons lors de l'entraînement de modèles profonds.
Rappelez-vous que nous ne savons même pas pourquoi les réseaux de neurones profonds plus simples (MLP et CNN conventionnels)
généralisent bien en premier lieu.
Même avec le dropout et le déclin des poids (weight decay),
ils restent si flexibles que leur capacité à généraliser à des données invisibles
nécessite probablement des garanties de généralisation théorique bien plus raffinées.

L'article original proposant la normalisation par lots :cite:`Ioffe.Szegedy.2015`, en plus d'introduire un outil puissant et utile,
proposait une explication sur la raison pour laquelle elle fonctionne :
en réduisant le *décalage de covariable interne* (internal covariate shift).
Vraisemblablement, par *décalage de covariable interne*, ils entendaient quelque chose comme l'intuition exprimée ci-dessus --- la
notion que la distribution des valeurs des variables change
au cours de l'entraînement.
Cependant, deux problèmes se posaient avec cette explication :
i) Cette dérive est très différente du *décalage de covariable*, ce qui fait du nom un abus de langage. Si quoi que ce soit, elle est plus proche de la dérive de concept (concept drift). 
ii) L'explication offre une intuition sous-spécifiée mais laisse la question de *pourquoi précisément cette technique fonctionne*
en suspens, en attente d'une explication rigoureuse.
Tout au long de ce livre, nous visons à transmettre les intuitions que les praticiens
utilisent pour guider leur développement de réseaux de neurones profonds.
Cependant, nous pensons qu'il est important
de séparer ces intuitions directrices
des faits scientifiques établis.
À terme, lorsque vous maîtriserez ce sujet
et commencerez à rédiger vos propres articles de recherche,
vous voudrez délimiter clairement
entre les affirmations techniques et les intuitions.

Suite au succès de la normalisation par lots,
son explication en termes de *décalage de covariable interne*
a refait surface à plusieurs reprises dans les débats de la littérature technique
et dans le discours plus large sur la manière de présenter la recherche en apprentissage automatique.
Dans un discours mémorable prononcé lors de la remise d'un Test of Time Award
à la conférence NeurIPS 2017,
Ali Rahimi a utilisé le *décalage de covariable interne*
comme point central d'un argument comparant
la pratique moderne de l'apprentissage profond à l'alchimie.
Par la suite, l'exemple a été réexaminé en détail
dans un document de position soulignant
des tendances troublantes dans l'apprentissage automatique :cite:`Lipton.Steinhardt.2018`.
D'autres auteurs
ont proposé des explications alternatives au succès de la normalisation par lots,
certains :cite:`Santurkar.Tsipras.Ilyas.ea.2018`
affirmant que le succès de la normalisation par lots survient malgré un comportement
qui est à certains égards opposé à ceux revendiqués dans l'article original.


Nous notons que le *décalage de covariable interne*
n'est pas plus digne de critique que n'importe laquelle des
milliers d'affirmations tout aussi vagues
faites chaque année dans la littérature technique de l'apprentissage automatique.
Vraisemblablement, sa résonance en tant que point focal de ces débats
tient à sa grande reconnaissance auprès du public cible.
La normalisation par lots s'est avérée être une méthode indispensable,
appliquée dans presque tous les classificateurs d'images déployés,
valant à l'article qui a introduit la technique
des dizaines de milliers de citations. Nous conjecturons cependant que les principes directeurs
de régularisation par injection de bruit, d'accélération par redimensionnement et enfin de prétraitement
pourraient bien mener à d'autres inventions de couches et de techniques à l'avenir.

Sur une note plus pratique, il y a un certain nombre d'aspects dont il vaut la peine de se souvenir concernant la normalisation par lots :

* Pendant l'entraînement du modèle, la normalisation par lots ajuste continuellement la sortie intermédiaire du
  réseau en utilisant la moyenne et l'écart-type du mini-lot, de sorte que les
  valeurs de la sortie intermédiaire de chaque couche dans tout le réseau de neurones soient plus stables.
* La normalisation par lots est légèrement différente pour les couches entièrement connectées que pour les couches convolutives. En fait,
  pour les couches convolutives, la normalisation par couche peut parfois être utilisée comme alternative.
* Comme une couche de dropout, les couches de normalisation par lots ont des comportements différents
  en mode entraînement qu'en mode prédiction.
* La normalisation par lots est utile pour la régularisation et l'amélioration de la convergence dans l'optimisation. En revanche,
  la motivation originale de réduire le décalage de covariable interne ne semble pas être une explication valide.
* Pour des modèles plus robustes et moins sensibles aux perturbations d'entrée, envisagez de supprimer la normalisation par lots :cite:`wang2022removing`.

## Exercices

1. Devrions-nous supprimer le paramètre de biais de la couche entièrement connectée ou de la couche convolutive avant la normalisation par lots ? Pourquoi ?
1. Comparez les taux d'apprentissage pour LeNet avec et sans normalisation par lots.
    1. Tracez l'augmentation de la précision de validation.
    1. Jusqu'à quel point pouvez-vous augmenter le taux d'apprentissage avant que l'optimisation n'échoue dans les deux cas ?
1. Avons-nous besoin d'une normalisation par lots dans chaque couche ? Expérimentez-le.
1. Implémentez une version "allégée" (lite) de la normalisation par lots qui ne supprime que la moyenne, ou alternativement une qui
   ne supprime que la variance. Comment se comporte-t-elle ?
1. Fixez les paramètres `beta` et `gamma`. Observez et analysez les résultats.
1. Pouvez-vous remplacer le dropout par la normalisation par lots ? Comment le comportement change-t-il ?
1. Idées de recherche : pensez à d'autres transformations de normalisation que vous pouvez appliquer :
    1. Pouvez-vous appliquer la transformation intégrale de probabilité ?
    1. Pouvez-vous utiliser une estimation de covariance de rang plein ? Pourquoi devriez-vous probablement éviter cela ? 
    1. Pouvez-vous utiliser d'autres variantes de matrices compactes (bloc-diagonale, de rang à faible déplacement, Monarch, etc.) ?
    1. Est-ce qu'une compression par sparsification agit comme un régularisateur ?
    1. Existe-t-il d'autres projections (par exemple, cône convexe, transformations spécifiques à un groupe de symétrie) que vous pouvez utiliser ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18005)
:end_tab:
