```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Traduction automatique et jeu de données
:label:`sec_machine_translation`

Parmi les percées majeures qui ont suscité un intérêt généralisé pour les réseaux récurrents (RNN) modernes figure une avancée majeure dans le domaine appliqué de la *traduction automatique* statistique. Ici, le modèle reçoit une phrase dans une langue et doit prédire la phrase correspondante dans une autre. Notez qu'ici les phrases peuvent être de longueurs différentes, et que les mots correspondants dans les deux phrases peuvent ne pas apparaître dans le même ordre, en raison des différences de structure grammaticale des deux langues.


De nombreux problèmes partagent cette caractéristique de mise en correspondance entre deux séquences ainsi "non alignées". Citons par exemple le passage de messages de dialogue à des réponses, ou de questions à des réponses. Plus largement, ces problèmes sont appelés problèmes *séquence à séquence* (seq2seq) et ils constituent notre point de mire pour le reste de ce chapitre ainsi qu'une grande partie du :numref:`chap_attention-and-transformers`.

Dans cette section, nous introduisons le problème de la traduction automatique et un exemple de jeu de données que nous utiliserons dans les exemples suivants. Pendant des décennies, les formulations statistiques de la traduction entre les langues ont été populaires :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`, avant même que les chercheurs ne fassent fonctionner les approches par réseaux de neurones (les méthodes étaient souvent regroupées sous le terme de *traduction automatique neuronale*).


Nous aurons d'abord besoin d'un nouveau code pour traiter nos données. Contrairement à la modélisation du langage que nous avons vue dans la :numref:`sec_language-model`, ici chaque exemple consiste en deux séquences de texte distinctes, l'une dans la langue source et l'autre (la traduction) dans la langue cible. Les extraits de code suivants montreront comment charger les données prétraitées en mini-lots pour l'entraînement.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

```{.python .input  n=4}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
import os
```

## [**Téléchargement et prétraitement du jeu de données**]

Pour commencer, nous téléchargeons un jeu de données anglais-français composé de [paires de phrases bilingues du projet Tatoeba](http://www.manythings.org/anki/). Chaque ligne du jeu de données est une paire délimitée par une tabulation consistant en une séquence de texte en anglais (la *source*) et la séquence de texte traduite en français (la *cible*). Notez que chaque séquence de texte peut être juste une phrase, ou un paragraphe de plusieurs phrases.

```{.python .input  n=5}
%%tab all
class MTFraEng(d2l.DataModule):  #@save
    """The English-French dataset."""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
```

```{.python .input}
%%tab all
data = MTFraEng() 
raw_text = data._download()
print(raw_text[:75])
```

Après avoir téléchargé le jeu de données, nous [**effectuons plusieurs étapes de prétraitement**] pour les données textuelles brutes. Par exemple, nous remplaçons les espaces insécables par des espaces, convertissons les lettres majuscules en minuscules et insérons des espaces entre les mots et les signes de ponctuation.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)
```

```{.python .input}
%%tab all
text = data._preprocess(raw_text)
print(text[:80])
```

## [**Tokenisation**]

Contrairement à la tokenisation au niveau des caractères dans la :numref:`sec_language-model`, nous préférons ici la tokenisation au niveau des mots pour la traduction automatique (les modèles de pointe actuels utilisent des techniques de tokenisation plus complexes). La méthode `_tokenize` suivante tokenise les premières `max_examples` paires de séquences de texte, où chaque jeton (token) est soit un mot, soit un signe de ponctuation. Nous ajoutons le jeton spécial "&lt;eos&gt;" à la fin de chaque séquence pour indiquer la fin de la séquence. Lorsqu'un modèle effectue une prédiction en générant une séquence jeton après jeton, la génération du jeton "&lt;eos&gt;" peut suggérer que la séquence de sortie est complète. À la fin, la méthode ci-dessous renvoie deux listes de listes de jetons : `src` et `tgt`. Plus précisément, `src[i]` est une liste de jetons de la $i^\textrm{ème}$ séquence de texte dans la langue source (l'anglais ici) et `tgt[i]` est celle dans la langue cible (le français ici).

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt
```

```{.python .input}
%%tab all
src, tgt = data._tokenize(text)
src[:6], tgt[:6]
```

[**Traçons l'histogramme du nombre de jetons par séquence de texte.**] Dans ce jeu de données anglais-français simple, la plupart des séquences de texte ont moins de 20 jetons.

```{.python .input  n=8}
%%tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
```

```{.python .input}
%%tab all
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt);
```

## Chargement de séquences de longueur fixe
:label:`subsec_loading-seq-fixed-len`

Rappelez-vous que dans la modélisation du langage, [**chaque séquence d'exemple**], qu'il s'agisse d'un segment d'une phrase ou d'une étendue sur plusieurs phrases, (**avait une longueur fixe.**) Cela était spécifié par l'argument `num_steps` (nombre de pas de temps ou de jetons) de la :numref:`sec_language-model`. En traduction automatique, chaque exemple est une paire de séquences de texte source et cible, où les deux séquences de texte peuvent avoir des longueurs différentes.

Pour l'efficacité des calculs, nous pouvons toujours traiter un mini-lot de séquences de texte à la fois par *troncation* et *remplissage* (padding). Supposons que chaque séquence du même mini-lot doive avoir la même longueur `num_steps`. Si une séquence de texte a moins de `num_steps` jetons, nous continuerons à ajouter le jeton spécial "&lt;pad&gt;" à sa fin jusqu'à ce que sa longueur atteigne `num_steps`. Sinon, nous tronquerons la séquence de texte en ne prenant que ses `num_steps` premiers jetons et en jetant le reste. De cette façon, chaque séquence de texte aura la même longueur pour être chargée dans des mini-lots de même forme. De plus, nous enregistrons également la longueur de la séquence source à l'exclusion des jetons de remplissage. Cette information sera nécessaire pour certains modèles que nous aborderons plus tard.


Comme le jeu de données de traduction automatique se compose de paires de langues, nous pouvons construire deux vocabulaires pour la langue source et la langue cible séparément. Avec la tokenisation au niveau des mots, la taille du vocabulaire sera nettement plus grande qu'avec la tokenisation au niveau des caractères. Pour pallier cela, nous traitons ici les jetons peu fréquents qui apparaissent moins de deux fois comme le même jeton inconnu ("&lt;unk&gt;"). Comme nous l'expliquerons plus tard (:numref:`fig_seq2seq`), lors de l'entraînement avec des séquences cibles, la sortie du décodeur (jetons d'étiquettes) peut être la même que l'entrée du décodeur (jetons cibles), décalée d'un jeton ; et le jeton spécial de début de séquence "&lt;bos&gt;" sera utilisé comme premier jeton d'entrée pour prédire la séquence cible (:numref:`fig_seq2seq_predict`).

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())
```

```{.python .input}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[s] for s in sentences])
        valid_len = d2l.reduce_sum(
            d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
        return array, vocab, valid_len
    src, tgt = self._tokenize(self._preprocess(raw_text), 
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)
```

## [**Lecture du jeu de données**]

Enfin, nous définissons la méthode `get_dataloader` pour renvoyer l'itérateur de données.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

[**Lisons le premier mini-lot du jeu de données anglais-français.**]

```{.python .input  n=11}
%%tab all
data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', d2l.astype(src, d2l.int32))
print('decoder input:', d2l.astype(tgt, d2l.int32))
print('source len excluding pad:', d2l.astype(src_valid_len, d2l.int32))
print('label:', d2l.astype(label, d2l.int32))
```

Nous montrons une paire de séquences source et cible traitées par la méthode `_build_arrays` ci-dessus (au format chaîne de caractères).

```{.python .input  n=12}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays
```

```{.python .input  n=13}
%%tab all
src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

## Résumé

En traitement du langage naturel, la *traduction automatique* désigne la tâche consistant à mettre en correspondance automatiquement une séquence représentant une chaîne de texte dans une langue *source* avec une chaîne représentant une traduction plausible dans une langue *cible*. En utilisant la tokenisation au niveau des mots, la taille du vocabulaire sera nettement plus grande qu'en utilisant la tokenisation au niveau des caractères, mais les longueurs de séquence seront beaucoup plus courtes. Pour atténuer la grande taille du vocabulaire, nous pouvons traiter les jetons peu fréquents comme un jeton "inconnu". Nous pouvons tronquer et compléter les séquences de texte afin qu'elles aient toutes la même longueur pour être chargées dans des mini-lots. Les implémentations modernes regroupent souvent les séquences de longueurs similaires dans des compartiments (buckets) pour éviter de gaspiller trop de calculs sur le remplissage.


## Exercices

1. Essayez différentes valeurs de l'argument `max_examples` dans la méthode `_tokenize`. Comment cela affecte-t-il les tailles de vocabulaire de la langue source et de la langue cible ?
1. Le texte de certaines langues comme le chinois et le japonais n'a pas d'indicateurs de limite de mots (par exemple, des espaces). La tokenisation au niveau des mots est-elle toujours une bonne idée pour de tels cas ? Pourquoi ou pourquoi pas ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3863)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18020)
:end_tab:
