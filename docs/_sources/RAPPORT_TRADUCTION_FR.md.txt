# Rapport de traduction française D2L

Date : 2026-06-11

## Artefacts produits

- Sources françaises partielles : `d2l-fr/`
- Site HTML français partiel généré : `d2l-fr/_build/html`
- Site HTML préparé pour GitHub Pages : `d2l-fr/docs`
- Notebooks générés : `d2l-fr/notebooks`
- Dépôt original cloné localement : `d2l-en/`
- Dépôt GitHub personnel créé : `https://github.com/Jamessiiii/d2l-fr-non-officiel`

## Build validé

Le pipeline D2L-Book a été validé sur un smoke test anglais, puis sur la version française partielle.

Commandes utiles :

```bash
cd d2l-fr
. .venv-build/bin/activate
d2lbook build eval --tab all
d2lbook build rst --tab all
install -m 0644 frontpage.html _build/rst_all/frontpage.html
d2lbook build html --tab all
cp -RX _build/html docs
touch docs/.nojekyll
```

Le build HTML final a réussi et produit notamment :

- `_build/html/index.html`
- `_build/html/chapter_preface/index.html`
- `_build/html/chapter_preface/translation-attribution.html`
- `_build/html/chapter_installation/index.html`
- `_build/html/chapter_introduction/index.html`
- `_build/html/chapter_notation/index.html`
- `_build/html/chapter_preliminaries/index.html`
- `_build/html/chapter_linear-regression/index.html`
- `docs/chapter_linear-classification/index.html`
- `docs/chapter_multilayer-perceptrons/index.html`
- `docs/chapter_builders-guide/index.html`
- `docs/chapter_convolutional-neural-networks/index.html`
- `docs/chapter_convolutional-modern/index.html`

Vérifications locales effectuées sur `http://localhost:4173` :

- page d'accueil ouverte ;
- navigation principale présente ;
- images présentes sur les pages testées ;
- éléments MathJax présents pour les équations ;
- page d'attribution visible ;
- vérification statique des liens locaux dans `docs` : `missing 0`.

## Traduction réalisée

Traduction réalisée par agent Gemini :

- `chapter_preface/index.md` : traduit en français par `translate_preface.txt`.
- `chapter_introduction/index.md` : traduit en français par `translate_introduction.txt`.
- `chapter_installation/index.md` : traduit en français par `translate_installation.txt`.
- `chapter_notation/index.md` : traduit en français par `translate_notation.txt`.
- `chapter_preliminaries/index.md` : traduit en français par `preliminaries_index.txt`.
- `chapter_preliminaries/ndarray.md` : traduit en français par `preliminaries_ndarray.txt`.
- `chapter_preliminaries/pandas.md` : traduit en français par `preliminaries_pandas.txt`.
- `chapter_preliminaries/linear-algebra.md` : traduit en français par `preliminaries_linear_algebra.txt`.
- `chapter_preliminaries/calculus.md` : traduit en français par `preliminaries_calculus.txt`.
- `chapter_preliminaries/autograd.md` : traduit en français par `preliminaries_autograd.txt`.
- `chapter_preliminaries/probability.md` : traduit en français par `preliminaries_probability.txt`.
- `chapter_preliminaries/lookup-api.md` : traduit en français par `preliminaries_lookup_api.txt`.
- `chapter_linear-regression/index.md` : traduit en français par `linear_regression_index.txt`.
- `chapter_linear-regression/synthetic-regression-data.md` : traduit en français par `linear_regression_synthetic_regression_data.txt`.
- `chapter_linear-regression/oo-design.md` : traduit en français par `linear_regression_oo_design.txt`.
- `chapter_linear-regression/linear-regression.md` : traduit en français par `linear_regression_linear_regression.txt`.
- `chapter_linear-regression/linear-regression-scratch.md` : traduit en français par `linear_regression_linear_regression_scratch.txt`.
- `chapter_linear-regression/linear-regression-concise.md` : traduit en français par `linear_regression_linear_regression_concise.txt`.
- `chapter_linear-regression/generalization.md` : traduit en français par `linear_regression_generalization.txt`.
- `chapter_linear-regression/weight-decay.md` : traduit en français par `linear_regression_weight_decay.txt`.
- `chapter_linear-classification/index.md` : traduit en français par `linear_classification_index.txt`.
- `chapter_linear-classification/classification.md` : traduit en français par `linear_classification_classification.txt`.
- `chapter_linear-classification/image-classification-dataset.md` : traduit en français par `linear_classification_image_classification_dataset.txt`.
- `chapter_linear-classification/softmax-regression-concise.md` : traduit en français par `linear_classification_softmax_regression_concise.txt`.
- `chapter_linear-classification/softmax-regression.md` : traduit en français par `linear_classification_softmax_regression.txt`.
- `chapter_linear-classification/softmax-regression-scratch.md` : traduit en français par `linear_classification_softmax_regression_scratch.txt`.
- `chapter_linear-classification/generalization-classification.md` : traduit en français par `linear_classification_generalization_classification.txt`.
- `chapter_linear-classification/environment-and-distribution-shift.md` : traduit en français par `linear_classification_environment_and_distribution_shift.txt`.
- `chapter_multilayer-perceptrons/index.md` : traduit en français par `multilayer_perceptrons_index.txt`.
- `chapter_multilayer-perceptrons/mlp.md` : traduit en français par `multilayer_perceptrons_mlp.txt`.
- `chapter_multilayer-perceptrons/mlp-implementation.md` : traduit en français par `multilayer_perceptrons_mlp_implementation.txt`.
- `chapter_multilayer-perceptrons/numerical-stability-and-init.md` : traduit en français par `multilayer_perceptrons_numerical_stability_and_init.txt`.
- `chapter_multilayer-perceptrons/backprop.md` : traduit en français par `multilayer_perceptrons_backprop.txt`.
- `chapter_multilayer-perceptrons/dropout.md` : traduit en français par `multilayer_perceptrons_dropout.txt`.
- `chapter_multilayer-perceptrons/generalization-deep.md` : traduit en français par `multilayer_perceptrons_generalization_deep.txt`.
- `chapter_multilayer-perceptrons/kaggle-house-price.md` : traduit en français par `multilayer_perceptrons_kaggle_house_price.txt`.
- `chapter_builders-guide/index.md` : traduit en français par `builders_guide_index.txt`.
- `chapter_builders-guide/model-construction.md` : traduit en français par `builders_guide_model_construction.txt`.
- `chapter_builders-guide/parameters.md` : traduit en français par `builders_guide_parameters.txt`.
- `chapter_builders-guide/init-param.md` : traduit en français par `builders_guide_init_param.txt`.
- `chapter_builders-guide/lazy-init.md` : traduit en français par `builders_guide_lazy_init.txt`.
- `chapter_builders-guide/custom-layer.md` : traduit en français par `builders_guide_custom_layer.txt`.
- `chapter_builders-guide/read-write.md` : traduit en français par `builders_guide_read_write.txt`.
- `chapter_builders-guide/use-gpu.md` : traduit en français par `builders_guide_use_gpu.txt`.
- `chapter_convolutional-neural-networks/index.md` : traduit en français par `cnn_index.txt`.
- `chapter_convolutional-neural-networks/why-conv.md` : traduit en français par `cnn_retry_why_conv.txt`.
- `chapter_convolutional-neural-networks/conv-layer.md` : traduit en français par `cnn_retry_conv_layer.txt`.
- `chapter_convolutional-neural-networks/padding-and-strides.md` : traduit en français par `cnn_retry_padding_and_strides.txt`.
- `chapter_convolutional-neural-networks/channels.md` : traduit en français par `cnn_retry_channels.txt`.
- `chapter_convolutional-neural-networks/pooling.md` : traduit en français par `cnn_retry_pooling.txt`.
- `chapter_convolutional-neural-networks/lenet.md` : traduit en français par `cnn_retry_lenet.txt`.
- `chapter_convolutional-modern/index.md` : traduit en français par `conv_modern_index.txt`.
- `chapter_convolutional-modern/alexnet.md` : traduit en français par `conv_modern_alexnet.txt`.
- `chapter_convolutional-modern/vgg.md` : traduit en français par `conv_modern_vgg.txt`.
- `chapter_convolutional-modern/nin.md` : traduit en français par `conv_modern_nin.txt`.
- `chapter_convolutional-modern/googlenet.md` : traduit en français par `conv_modern_googlenet.txt`.
- `chapter_convolutional-modern/batch-norm.md` : traduit en français par `conv_modern_batch_norm.txt`.
- `chapter_convolutional-modern/resnet.md` : traduit en français par `conv_modern_resnet.txt`.
- `chapter_convolutional-modern/densenet.md` : traduit en français par `conv_modern_densenet.txt`.
- `chapter_convolutional-modern/cnn-design.md` : traduit en français par `conv_modern_cnn_design.txt`.

Ajouts manuels de structure française :

- `config.ini` : nom du projet, titre et liens de navigation adaptés.
- `index.md` : titre français et table des matières partielle.
- `chapter_preface/translation-attribution.md` : page d'attribution non officielle.
- `README.md` et `ATTRIBUTION.md` : avertissement de traduction non officielle, lien vers le dépôt original, crédits et licences.
- `docs/` : copie du site HTML prête pour GitHub Pages.
- `notebooks/` : notebooks générés à partir du build partiel.
- `img/polygon-circle.svg` : figure utilisée par `chapter_preliminaries/calculus.md`.
- `img/capacity-vs-error.svg`, `img/fit-linreg.svg`, `img/singleneuron.svg`, `img/neuron.svg` : figures utilisées par `chapter_linear-regression`.
- `img/softmaxreg.svg`, `img/cat-dog-train.png`, `img/cat-dog-test.png`, `img/popvssoda.png` : figures utilisées par `chapter_linear-classification`.
- `img/dropout2.svg`, `img/forward.svg`, `img/house-pricing.png`, `img/kaggle-submit2.png`, `img/kaggle.png`, `img/mlp.svg` : figures utilisées par `chapter_multilayer-perceptrons`.
- `img/blocks.svg`, `img/copyto.svg` : figures utilisées par `chapter_builders-guide`.
- `img/conv-multi-in.svg`, `img/conv-1x1.svg`, `img/correlation.svg`, `img/field-visual.png`, `img/lenet.svg`, `img/lenet-vert.svg`, `img/conv-reuse.svg`, `img/conv-pad.svg`, `img/conv-stride.svg`, `img/pooling.svg`, `img/waldo-football.jpg`, `img/waldo-mask.jpg` : figures utilisées par `chapter_convolutional-neural-networks`.
- `img/alexnet.svg`, `img/anynet.svg`, `img/densenet-block.svg`, `img/densenet.svg`, `img/filters.png`, `img/functionclasses.svg`, `img/inception-full-90.svg`, `img/inception.svg`, `img/nin.svg`, `img/regnet-fig.png`, `img/residual-block.svg`, `img/resnet-block.svg`, `img/resnet18-90.svg`, `img/resnext-block.svg`, `img/vgg.svg` : figures utilisées par `chapter_convolutional-modern`.

## Non traduit / restant à faire

Les chapitres copiés dans cette version partielle ont été traduits.
Restent à faire :

- tous les autres chapitres du dépôt officiel non encore copiés dans `d2l-fr`.

La traduction complète du livre reste à faire chapitre par chapitre en lançant de nouveaux agents Gemini sur les chapitres restants.

## Blocage rencontré puis résolu

Le premier run Gemini a réussi pour la préface, puis s'est arrêté pour les autres agents :

- `agent_02` (`translate_introduction.txt`) : erreur Gemini CLI `Error executing tool write_file: params must have required property 'file_path'`.
- `agent_03` (`translate_installation.txt`) : demande d'authentification navigateur, puis expiration.
- `translate_notation.txt` n'a pas été exécuté jusqu'au bout dans le run interrompu.

Tentative d'authentification :

- Le CLI a ouvert une page Google OAuth.
- La page est restée bloquée sur l'autorisation de `Gemini Code Assist and Gemini CLI`.
- Le CLI a terminé par : `Authentication timed out after 5 minutes. The browser tab may have gotten stuck in a loading state. Please try again or use NO_BROWSER=true for manual authentication.`
- `NO_BROWSER=true` a ensuite échoué avec : `Manual authorization is required but the current session is non-interactive.`

Après réactivation de l'authentification Gemini CLI, les agents `translate_installation.txt`, `translate_notation.txt` et `translate_introduction.txt` ont été relancés avec succès.

Lors de la traduction de `chapter_preliminaries`, deux agents ont d'abord rencontré `MODEL_CAPACITY_EXHAUSTED` (`ndarray` et `autograd`). Ils ont été relancés avec `--max-parallel 1`; `ndarray` a terminé normalement. Le run `autograd` a écrit le fichier traduit mais s'est ensuite bloqué dans Gemini; le fichier a été vérifié localement et intégré au build.

Après traduction, les blocs de code fenced de `chapter_preliminaries` ont été restaurés depuis le dépôt original anglais afin de préserver exactement le code, les commentaires de code, les directives tabulaires et les sorties techniques.

Pour `chapter_linear-regression`, huit agents Gemini ont été lancés en deux vagues. Les blocs de code fenced ont ensuite été restaurés depuis le dépôt original anglais afin de préserver le code, les commentaires de code, les directives tabulaires et les sorties techniques. Une erreur de balisage D2L produite par la traduction dans `synthetic-regression-data.md` a été corrigée avant génération RST.

Pour `chapter_linear-classification`, huit agents Gemini ont été lancés en deux vagues. Les blocs de code fenced ont ensuite été restaurés depuis le dépôt original anglais. Le build `eval_all` a réussi et a généré les notebooks du chapitre. La conversion RST/Sphinx complète s'est bloquée dans l'environnement local; le chapitre a donc été publié dans `docs/chapter_linear-classification/` via une génération HTML ciblée avec MathJax et les images locales, en attendant de stabiliser le pipeline Sphinx complet.

Pour `chapter_multilayer-perceptrons`, huit agents Gemini ont été lancés en deux vagues. Les blocs de code fenced ont ensuite été restaurés depuis le dépôt original anglais. Le build D2L-Book local étant instable dans cette session, les notebooks et pages HTML publiques ont été générés à partir des sources Markdown traduites, avec MathJax et les images locales. Cette solution garde les sources françaises et le site consultable, mais le pipeline Sphinx complet devra être stabilisé avant la version finale.

Pour `chapter_builders-guide`, huit agents Gemini ont été lancés en deux vagues. Les blocs de code fenced ont ensuite été restaurés depuis le dépôt original anglais. Les notebooks et pages HTML publiques ont été générés à partir des sources Markdown traduites, avec MathJax et les images locales, car le pipeline D2L-Book/Sphinx complet reste instable dans cette session.

Pour `chapter_convolutional-neural-networks`, les agents Gemini ont d'abord été lancés en vague parallèle. Un agent a rencontré `MODEL_CAPACITY_EXHAUSTED` et deux agents sont restés bloqués sur une invite interactive d'authentification. Les sections restantes ont donc été relancées une par une depuis le dossier parent contenant `d2l-fr` et `d2l-en`, afin de permettre à Gemini d'inspecter les sources relatives sans chemin absolu. Les blocs de code fenced ont ensuite été restaurés depuis les sources officielles brutes. Les notebooks et pages HTML publiques ont été générés à partir des sources Markdown traduites, avec MathJax et les images locales.

Pour `chapter_convolutional-modern`, les sections ont été traduites une par une avec Gemini depuis le dossier parent contenant `d2l-fr` et `d2l-en`. `cnn-design` a d'abord rencontré `MODEL_CAPACITY_EXHAUSTED`, puis a réussi lors d'une relance isolée. `batch-norm` et `resnet` ont affiché des erreurs post-écriture dans la sortie Gemini, mais les fichiers traduits ont ensuite été validés localement. Les blocs de code fenced ont été restaurés depuis les sources officielles brutes avant génération des notebooks et pages HTML publiques.

## Limites du build partiel

Le dernier build Sphinx HTML a réussi avec 11 avertissements. Un build complet précédent du même état avait listé 88 avertissements attendus pour une version partielle :

- plusieurs citations et labels pointent vers des chapitres absents du build partiel ;
- le site ne contient que les chapitres copiés/traduits à ce stade.

Pour le build complet, il faudra restaurer une copie complète ou des liens robustes vers toutes les images et tous les chapitres, puis rétablir `resources` dans `config.ini` pour se rapprocher du dépôt officiel.

Note technique : l'ancien environnement `.venv-build` placé dans la racine du dépôt a fini par bloquer `pkg_resources`/`notedown`. Les builds courants utilisent des environnements virtuels frais hors dépôt, notamment :

```bash
../d2l-fr-venv-build-fresh
../d2l-fr-venv-build-run2
```

Le HTML a été produit en lançant directement Sphinx sur `_build/rst_all` après génération des notebooks/RST, afin d'éviter la réécriture de deux notebooks `eval_all` vides observée pendant un run interrompu.

## État GitHub

Le dépôt GitHub `Jamessiiii/d2l-fr-non-officiel` a été créé via le navigateur intégré, sans modifier le dépôt original `d2l-ai/d2l-en`.

Le dépôt local est initialisé sur `main`, avec le remote :

```bash
origin  https://github.com/Jamessiiii/d2l-fr-non-officiel.git
```

GitHub CLI a ensuite été réauthentifié comme `Jamessiiii`. GitHub Pages est configuré pour servir le site depuis `main` / `docs` :

```text
https://jamessiiii.github.io/d2l-fr-non-officiel/
```

Le site public doit être revérifié après chaque nouveau push : page d'accueil, navigation, équations MathJax, figures, liens internes et attribution.
