# Réseaux de neurones convolutifs
:label:`chap_cnn`

Les données d'image sont représentées sous la forme d'une grille de pixels à deux dimensions, que l'image soit monochromatique ou en couleur. En conséquence, chaque pixel correspond respectivement à une ou plusieurs valeurs numériques. Jusqu'à présent, nous avons ignoré cette structure riche et traité les images comme des vecteurs de nombres en les *aplatissant*, indépendamment de la relation spatiale entre les pixels. Cette approche profondément insatisfaisante était nécessaire pour faire passer les vecteurs unidimensionnels résultants à travers un MLP entièrement connecté.

Parce que ces réseaux sont invariants à l'ordre des caractéristiques, nous pourrions obtenir des résultats similaires, que nous préservions un ordre correspondant à la structure spatiale des pixels ou que nous permutions les colonnes de notre matrice de conception avant d'ajuster les paramètres du MLP. Idéalement, nous exploiterions nos connaissances préalables selon lesquelles les pixels proches sont généralement liés les uns aux autres, pour construire des modèles efficaces d'apprentissage à partir de données d'image.

Ce chapitre présente les *réseaux de neurones convolutifs* (CNN)
:cite:`LeCun.Jackel.Bottou.ea.1995`, une famille puissante de réseaux de neurones conçus précisément à cette fin.
Les architectures basées sur les CNN sont désormais omniprésentes dans le domaine de la vision par ordinateur.
Par exemple, sur la collection ImageNet
:cite:`Deng.Dong.Socher.ea.2009`, c'est seulement l'utilisation des réseaux de neurones convolutifs, en abrégé Convnets, qui a permis des améliorations significatives des performances :cite:`Krizhevsky.Sutskever.Hinton.2012`.

Les CNN modernes, comme on les appelle familièrement, doivent leur conception à des inspirations issues de la biologie, de la théorie des groupes et d'une bonne dose de bricolage expérimental. En plus de leur efficacité d'échantillonnage pour obtenir des modèles précis, les CNN ont tendance à être efficaces sur le plan computationnel, à la fois parce qu'ils nécessitent moins de paramètres que les architectures entièrement connectées et parce que les convolutions sont faciles à paralléliser sur les cœurs de GPU :cite:`Chetlur.Woolley.Vandermersch.ea.2014`. Par conséquent, les praticiens appliquent souvent les CNN dès que possible, et de plus en plus, ils sont apparus comme des concurrents crédibles même sur des tâches avec une structure de séquence unidimensionnelle, telles que l'audio :cite:`Abdel-Hamid.Mohamed.Jiang.ea.2014`, le texte
:cite:`Kalchbrenner.Grefenstette.Blunsom.2014` et l'analyse de séries temporelles
:cite:`LeCun.Bengio.ea.1995`, où les réseaux de neurones récurrents sont
conventionnellement utilisés. Certaines adaptations astucieuses des CNN les ont également amenés à s'attaquer à des données structurées en graphes :cite:`Kipf.Welling.2016` et dans les systèmes de recommandation.

Tout d'abord, nous approfondirons la motivation des réseaux de neurones convolutifs. Cela sera suivi d'un tour d'horizon des opérations de base qui constituent l'épine dorsale de tous les réseaux convolutifs.
Celles-ci incluent les couches convolutives elles-mêmes, les détails pratiques tels que le remplissage (padding) et le pas (stride), les couches de regroupement (pooling) utilisées pour agréger les informations à travers des régions spatiales adjacentes, l'utilisation de plusieurs canaux à chaque couche, et une discussion approfondie sur la structure des architectures modernes.
Nous conclurons le chapitre par un exemple complet et fonctionnel de LeNet, le premier réseau convolutif déployé avec succès, bien avant l'essor de l'apprentissage profond moderne.
Dans le chapitre suivant, nous plongerons dans les implémentations complètes de certaines architectures CNN populaires et comparativement récentes dont les conceptions représentent la plupart des techniques couramment utilisées par les praticiens modernes.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```
