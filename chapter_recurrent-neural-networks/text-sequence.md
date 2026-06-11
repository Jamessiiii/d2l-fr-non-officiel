# Conversion du texte brut en données séquentielles
:label:`sec_text-sequence`

Tout au long de ce livre, nous travaillerons souvent avec des données textuelles représentées comme des séquences de mots, de caractères ou de morceaux de mots. Pour commencer, nous aurons besoin de quelques outils de base pour convertir le texte brut en séquences de la forme appropriée. Les pipelines de prétraitement typiques exécutent les étapes suivantes :

1. Charger le texte sous forme de chaînes de caractères en mémoire.
1. Diviser les chaînes en jetons (par exemple, des mots ou des caractères).
1. Construire un dictionnaire de vocabulaire pour associer chaque élément du vocabulaire à un indice numérique.
1. Convertir le texte en séquences d'indices numériques.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import random
import re
```

## Lecture du jeu de données

Ici, nous travaillerons avec l'œuvre de H. G. Wells,
[*La Machine à explorer le temps*](http://www.gutenberg.org/ebooks/35),
un livre contenant un peu plus de 30 000 mots.
Bien que les applications réelles impliquent généralement
des jeux de données nettement plus volumineux,
celui-ci est suffisant pour démontrer le pipeline de prétraitement.
La méthode `_download` suivante
(**lit le texte brut dans une chaîne de caractères**).

```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    """The Time Machine dataset."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

Par simplicité, nous ignorons la ponctuation et la capitalisation lors du prétraitement du texte brut.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## Tokenisation

Les *jetons* (ou *tokens*) sont les unités atomiques (indivisibles) du texte.
Chaque pas de temps correspond à un jeton,
mais ce qui constitue précisément un jeton est un choix de conception.
Par exemple, nous pourrions représenter la phrase
"Baby needs a new pair of shoes"
comme une séquence de 7 mots,
où l'ensemble de tous les mots constitue
un large vocabulaire (généralement des dizaines
ou des centaines de milliers de mots).
Ou nous pourrions représenter la même phrase
comme une séquence beaucoup plus longue de 30 caractères,
en utilisant un vocabulaire beaucoup plus restreint
(il n'y a que 256 caractères ASCII distincts).
Ci-dessous, nous tokenisons notre texte prétraité
en une séquence de caractères.

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## Vocabulaire

Ces jetons sont toujours des chaînes de caractères.
Cependant, les entrées de nos modèles
doivent ultimement consister
en des valeurs numériques.
[**Ensuite, nous introduisons une classe
pour construire des *vocabulaires*,
c'est-à-dire des objets qui associent
chaque valeur de jeton distincte
à un indice unique.**]
Tout d'abord, nous déterminons l'ensemble des jetons uniques dans notre *corpus* d'entraînement.
Nous attribuons ensuite un indice numérique à chaque jeton unique.
Les éléments de vocabulaire rares sont souvent abandonnés par commodité.
Chaque fois que nous rencontrons un jeton lors de l'entraînement ou du test
qui n'a pas été vu auparavant ou qui a été retiré du vocabulaire,
nous le représentons par un jeton spécial "&lt;unk&gt;",
signifiant qu'il s'agit d'une valeur *inconnue* (unknown).

```{.python .input  n=8}
%%tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
```

Nous pouvons maintenant [**construire un vocabulaire**] pour notre jeu de données,
en convertissant la séquence de chaînes
en une liste d'indices numériques.
Notez que nous n'avons perdu aucune information
et que nous pouvons facilement reconvertir notre jeu de données
vers sa représentation originale (chaînes de caractères).

```{.python .input  n=9}
%%tab all
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

## Assemblage final

En utilisant les classes et méthodes ci-dessus,
nous [**regroupons tout dans la méthode suivante
`build` de la classe `TimeMachine`**],
qui renvoie `corpus`, une liste d'indices de jetons, et `vocab`,
le vocabulaire du corpus de *La Machine à explorer le temps*.
Les modifications que nous avons apportées ici sont :
(i) nous tokenisons le texte en caractères, et non en mots,
pour simplifier l'entraînement dans les sections suivantes ;
(ii) `corpus` est une liste unique, et non une liste de listes de jetons,
car chaque ligne de texte dans le jeu de données de *La Machine à explorer le temps*
n'est pas nécessairement une phrase ou un paragraphe.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## Statistiques exploratoires du langage
:label:`subsec_natural-lang-stat`

En utilisant le corpus réel et la classe `Vocab` définie sur les mots,
nous pouvons inspecter les statistiques de base concernant l'utilisation des mots dans notre corpus.
Ci-dessous, nous construisons un vocabulaire à partir des mots utilisés dans *La Machine à explorer le temps*
et affichons les dix mots les plus fréquents.

```{.python .input  n=11}
%%tab all
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

Notez que (**les dix mots les plus fréquents**)
ne sont pas très descriptifs.
On peut même imaginer que
nous verrions une liste très similaire
si nous avions choisi n'importe quel livre au hasard.
Les articles comme "the" et "a",
les pronoms comme "i" et "my",
et les prépositions comme "of", "to" et "in"
apparaissent souvent car ils remplissent des rôles syntaxiques courants.
De tels mots qui sont fréquents mais peu descriptifs
sont souvent appelés (***mots vides*** ou ***stop words***) et,
dans les générations précédentes de classificateurs de texte
basés sur les représentations dites par sac de mots (bag-of-words),
ils étaient le plus souvent filtrés.
Cependant, ils sont porteurs de sens et
il n'est pas nécessaire de les filtrer
lorsque l'on travaille avec des modèles neuronaux modernes
basés sur les RNN et les Transformers.
Si vous regardez plus loin dans la liste,
vous remarquerez
que la fréquence des mots décroît rapidement.
Le $10^{\textrm{e}}$ mot le plus fréquent
est moins d'un cinquième aussi courant que le plus populaire.
La fréquence des mots a tendance à suivre une distribution de loi de puissance
(plus précisément la loi de Zipf) à mesure que l'on descend dans les rangs.
Pour avoir une meilleure idée, nous [**tracons la figure de la fréquence des mots**].

```{.python .input  n=12}
%%tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

Après avoir traité les premiers mots comme des exceptions,
tous les mots restants suivent approximativement une ligne droite sur un graphique log-log.
Ce phénomène est capturé par la *loi de Zipf*,
qui stipule que la fréquence $n_i$
du $i^{\textrm{e}}$ mot le plus fréquent est :

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

ce qui est équivalent à

$$\log n_i = -\alpha \log i + c,$$

où $\alpha$ est l'exposant qui caractérise
la distribution et $c$ est une constante.
Cela devrait déjà nous donner matière à réflexion si nous voulons
modéliser les mots par des statistiques de comptage.
Après tout, nous surestimerons considérablement la fréquence de la traîne, également connue sous le nom de mots peu fréquents. Mais [**qu'en est-il des autres combinaisons de mots, comme deux mots consécutifs (bigrammes), trois mots consécutifs (trigrammes)**], et au-delà ?
Voyons si la fréquence des bigrammes se comporte de la même manière que la fréquence des mots uniques (unigrammes).

```{.python .input  n=13}
%%tab all
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

Une chose est notable ici. Sur les dix paires de mots les plus fréquentes, neuf sont composées uniquement de mots vides et une seule est pertinente pour le livre lui-même — "the time". De plus, voyons si la fréquence des trigrammes se comporte de la même manière.

```{.python .input  n=14}
%%tab all
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

Maintenant, [**visualisons la fréquence des jetons**] parmi ces trois modèles : unigrammes, bigrammes et trigrammes.

```{.python .input  n=15}
%%tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

Cette figure est très intéressante.
Premièrement, au-delà des mots unigrammes, les séquences de mots
semblent également suivre la loi de Zipf,
bien qu'avec un exposant $\alpha$ plus petit
dans :eqref:`eq_zipf_law`,
selon la longueur de la séquence.
Deuxièmement, le nombre de $n$-grammes distincts n'est pas si grand.
Cela nous donne l'espoir qu'il existe une structure importante dans le langage.
Troisièmement, de nombreux $n$-grammes apparaissent très rarement.
Cela rend certaines méthodes inadaptées à la modélisation du langage
et motive l'utilisation de modèles d'apprentissage profond.
Nous en discuterons dans la section suivante.


## Résumé

Le texte fait partie des formes les plus courantes de données séquentielles rencontrées en apprentissage profond.
Les choix courants de ce qui constitue un jeton sont les caractères, les mots et les morceaux de mots.
Pour prétraiter le texte, nous effectuons généralement les étapes suivantes : (i) diviser le texte en jetons ; (ii) construire un vocabulaire pour faire correspondre les chaînes de jetons à des indices numériques ; et (iii) convertir les données textuelles en indices de jetons pour que les modèles puissent les manipuler.
En pratique, la fréquence des mots a tendance à suivre la loi de Zipf. C'est vrai non seulement pour les mots individuels (unigrammes), mais aussi pour les $n$-grammes.


## Exercices

1. Dans l'expérience de cette section, tokenisez le texte en mots et faites varier la valeur de l'argument `min_freq` de l'instance `Vocab`. Caractérisez qualitativement comment les changements de `min_freq` impactent la taille du vocabulaire résultant.
1. Estimez l'exposant de la distribution de Zipf pour les unigrammes, les bigrammes et les trigrammes dans ce corpus.
1. Trouvez d'autres sources de données (téléchargez un jeu de données standard d'apprentissage automatique, choisissez un autre livre du domaine public, scrapez un site web, etc.). Pour chacune, tokenisez les données au niveau des mots et des caractères. Comment les tailles de vocabulaire se comparent-elles à celles du corpus de *La Machine à explorer le temps* pour des valeurs équivalentes de `min_freq` ? Estimez l'exposant de la distribution de Zipf correspondant aux distributions d'unigrammes et de bigrammes pour ces corpus. Comment se comparent-ils aux valeurs que vous avez observées pour le corpus de *La Machine à explorer le temps* ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18011)
:end_tab:
