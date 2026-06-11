# Réseaux de neurones récurrents modernes
:label:`chap_modern_rnn`

Le chapitre précédent a introduit les idées clés des réseaux de neurones récurrents (RNN). Cependant, tout comme pour les réseaux de neurones convolutifs, il y a eu énormément d'innovations dans les architectures de RNN, aboutissant à plusieurs conceptions complexes qui ont fait leurs preuves en pratique. En particulier, les conceptions les plus populaires comportent des mécanismes pour atténuer la fameuse instabilité numérique à laquelle sont confrontés les RNN, typifiée par l'évanescence et l'explosion du gradient. Rappelons que dans le :numref:`chap_rnn`, nous avons traité l'explosion du gradient en appliquant une heuristique brute d'écrêtage du gradient. Malgré l'efficacité de ce "hack", cela laisse entier le problème de l'évanescence du gradient.

Dans ce chapitre, nous introduisons les idées clés derrière les architectures de RNN les plus réussies pour les séquences, issues de deux articles. Le premier, *Long Short-Term Memory* :cite:`Hochreiter.Schmidhuber.1997`, introduit la *cellule mémoire*, une unité de calcul qui remplace les nœuds traditionnels dans la couche cachée d'un réseau. Grâce à ces cellules mémoire, les réseaux sont capables de surmonter les difficultés d'entraînement rencontrées par les réseaux récurrents antérieurs. Intuitivement, la cellule mémoire évite le problème de l'évanescence du gradient en conservant les valeurs de l'état interne de chaque cellule mémoire en cascade le long d'une arête récurrente de poids 1 sur de nombreux pas de temps successifs. Un ensemble de portes multiplicatives aide le réseau à déterminer non seulement les entrées à autoriser dans l'état mémoire, mais aussi quand le contenu de l'état mémoire doit influencer la sortie du modèle.

Le deuxième article, *Bidirectional Recurrent Neural Networks* :cite:`Schuster.Paliwal.1997`, introduit une architecture dans laquelle l'information provenant à la fois du futur (pas de temps suivants) et du passé (pas de temps précédents) est utilisée pour déterminer la sortie à n'importe quel point de la séquence. Cela contraste avec les réseaux précédents, dans lesquels seule l'entrée passée peut affecter la sortie. Les RNN bidirectionnels sont devenus un pilier pour les tâches d'étiquetage de séquences dans le traitement du langage naturel, parmi une myriade d'autres tâches. Heureusement, les deux innovations ne s'excluent pas mutuellement et ont été combinées avec succès pour la classification des phonèmes :cite:`Graves.Schmidhuber.2005` et la reconnaissance de l'écriture manuscrite :cite:`graves2008novel`.

Les premières sections de ce chapitre expliqueront l'architecture LSTM, une version plus légère appelée l'unité récurrente à portes (GRU), les idées clés derrière les RNN bidirectionnels et une brève explication de la manière dont les couches de RNN sont empilées pour former des RNN profonds. Par la suite, nous explorerons l'application des RNN dans les tâches de séquence à séquence, en introduisant la traduction automatique ainsi que des idées clés telles que les architectures *encodeur--décodeur* et la *recherche en faisceau*.

```toc
:maxdepth: 2

lstm
gru
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```
