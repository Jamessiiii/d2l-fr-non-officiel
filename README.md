# d2l-fr-non-officiel

Traduction française non officielle de *Dive into Deep Learning*.

Ce dépôt prépare une version française du site D2L à partir des sources Markdown/D2L-Book du dépôt officiel : <https://github.com/d2l-ai/d2l-en>.

## Statut

Cette version est partielle.

Traduit en français :

- Préface (`chapter_preface/index.md`)
- Page d'attribution (`chapter_preface/translation-attribution.md`)
- Introduction (`chapter_introduction/index.md`)
- Installation (`chapter_installation/index.md`)
- Notation (`chapter_notation/index.md`)
- Préliminaires (`chapter_preliminaries/*.md`)
- Régression linéaire (`chapter_linear-regression/*.md`)
- Classification linéaire (`chapter_linear-classification/*.md`)
- Perceptrons multicouches (`chapter_multilayer-perceptrons/*.md`)
- Guide du constructeur (`chapter_builders-guide/*.md`)

La traduction complète reste à poursuivre chapitre par chapitre avec des agents Gemini, en conservant le code, les équations, les labels, les références internes et la structure D2L-Book.

## Site HTML

Le site généré pour GitHub Pages se trouve dans `docs/`.
Les notebooks générés se trouvent dans `notebooks/`.

Pour reconstruire localement :

```bash
cd d2l-fr
. ../d2l-fr-venv-build-fresh/bin/activate
d2lbook build eval --tab all
d2lbook build rst --tab all
install -m 0644 frontpage.html _build/rst_all/frontpage.html
sphinx-build -b html _build/rst_all _build/html
touch docs/.nojekyll
```

GitHub Pages sert le dossier `docs/` depuis la branche `main`.

## Attribution

Cette traduction/adaptation n'est pas officielle et n'est pas approuvee par les auteurs originaux.

Les auteurs originaux de *Dive into Deep Learning* sont Aston Zhang, Zachary C. Lipton, Mu Li et Alexander J. Smola. Voir `ATTRIBUTION.md` pour les details.

## Licence

L'oeuvre originale est distribuee sous licence Creative Commons Attribution-ShareAlike 4.0 International pour le texte du livre, et le code d'exemple sous une licence MIT modifiee. Les fichiers de licence originaux sont conserves dans ce depot.
