# Guide du constructeur
:label:`chap_computation`

Aux côtés des jeux de données géants et du matériel puissant,
d'excellents outils logiciels ont joué un rôle indispensable
dans les progrès rapides du deep learning.
À commencer par la bibliothèque pionnière Theano publiée en 2007,
des outils open source flexibles ont permis aux chercheurs
de prototyper rapidement des modèles, en évitant le travail répétitif
lors du recyclage de composants standard
tout en conservant la capacité d'effectuer des modifications de bas niveau.
Au fil du temps, les bibliothèques de deep learning ont évolué
pour proposer des abstractions de plus en plus grossières.
Tout comme les concepteurs de semi-conducteurs sont passés de la spécification de transistors
aux circuits logiques puis à l'écriture de code,
les chercheurs en réseaux de neurones sont passés d'une réflexion sur
le comportement de neurones artificiels individuels
à une conception de réseaux en termes de couches entières,
et conçoivent désormais souvent des architectures avec des *blocs* bien plus larges à l'esprit.


Jusqu'à présent, nous avons introduit quelques concepts de base du machine learning,
en montant en puissance jusqu'à des modèles de deep learning pleinement fonctionnels.
Dans le chapitre précédent,
nous avons implémenté chaque composant d'un MLP à partir de zéro
et avons même montré comment exploiter des API de haut niveau
pour déployer les mêmes modèles sans effort.
Pour vous amener aussi loin aussi vite, nous avons *fait appel* aux bibliothèques,
mais avons omis les détails plus avancés sur *leur fonctionnement*.
Dans ce chapitre, nous allons lever le voile,
en creusant plus profondément dans les composants clés du calcul en deep learning,
à savoir la construction de modèles, l'accès et l'initialisation des paramètres,
la conception de couches et de blocs personnalisés, la lecture et l'écriture de modèles sur disque,
et l'exploitation des GPU pour obtenir des accélérations spectaculaires.
Ces connaissances vous feront passer d'un *utilisateur final* à un *utilisateur expert*,
en vous donnant les outils nécessaires pour récolter les fruits
d'une bibliothèque de deep learning mature tout en conservant la flexibilité
nécessaire pour implémenter des modèles plus complexes, y compris ceux que vous inventez vous-même !
Bien que ce chapitre n'introduise aucun nouveau modèle ou jeu de données,
les chapitres de modélisation avancée qui suivent s'appuient fortement sur ces techniques.

```toc
:maxdepth: 2

model-construction
parameters
init-param
lazy-init
custom-layer
read-write
use-gpu
```
