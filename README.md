# forensic-ml

Reusable browser-based forensic classification challenge built with JupyterLite and deployed as a static GitHub Pages site.

## What this repository provides

- a real JupyterLite build served from GitHub Pages;
- a participant notebook that runs fully in the browser;
- a package diagnostics notebook for troubleshooting browser-side package installs;
- generic synthetic training and mystery CSV files;
- a deterministic data-generation script for swapping in new challenge datasets;
- validation scripts and automated tests;
- a simple landing page that links straight into the notebooks.

## Participant experience

1. Open the GitHub Pages URL for the repository.
2. Click **Launch participant notebook**.
3. Run the environment check cell to confirm the preloaded Pyodide package versions.
4. Inspect `data/training_samples.csv`.
5. Upload replacement CSV files in the JupyterLite file browser if you want to swap in a new challenge.
6. Run the model search and compare cross-validated scores.
7. Predict the class label for the mystery samples.

All computation runs in the browser. There is no Python server, backend API, cloud notebook, or participant installation step.

The JupyterLite build now bundles the compatible Pyodide runtime and a generated lockfile into the deployed site, then preloads `python-dateutil`, `pandas`, and `scikit-learn` at kernel startup so notebook imports do not depend on ad-hoc `piplite` installs or a browser fetch from a third-party CDN.

## Repository layout

- `content/notebooks/forensic_classification_challenge.ipynb` - participant-facing workflow
- `content/notebooks/package_diagnostics.ipynb` - package/environment diagnostics
- `content/data/training_samples.csv` - labelled example dataset
- `content/data/mystery_samples.csv` - unlabelled example mystery samples
- `jupyter-lite.json` - build and runtime JupyterLite/Pyodide kernel settings
- `scripts/generate_synthetic_data.py` - deterministic dataset generator
- `scripts/validate_repository.py` - repository and build validation
- `scripts/validate_notebooks.py` - notebook structure validation
- `pages/index.html` - GitHub Pages landing page
- `.github/workflows/deploy.yml` - test, build, and deploy workflow

## Data contract for swapping challenge files

The participant notebook is intentionally generic. Replacement CSV files should use:

- `sample_id` for the row identifier;
- `class_label` for the training label column;
- numeric measurement columns for all remaining features.

The committed example files use six feature columns named `measurement_01` through `measurement_06`, but you can replace them with any numeric features as long as both the training and mystery files share the same feature columns.

## Local validation

```bash
python -m pip install -r requirements.txt
python scripts/validate_repository.py --root .
python scripts/validate_notebooks.py --root .
pytest -q
rm -rf site && mkdir -p site && cp -R pages/. site/ && touch site/.nojekyll && jupyter lite build --config jupyter-lite.json --output-dir site/lite
python scripts/validate_repository.py --root . --site-dir site
```

Then open `site/index.html` with any static hosting environment, or push the repository to GitHub and enable **Pages -> Build and deployment -> GitHub Actions**.

If a browser still shows the old JupyterLite package-loading errors after a deployment, clear the site's stored browser data (or hard-refresh and reset the JupyterLite application state) so the updated bundled Pyodide runtime and lockfile are reloaded.

## GitHub Pages deployment

The workflow in `.github/workflows/deploy.yml`:

1. installs pinned dependencies;
2. validates the repository and notebooks;
3. runs the automated tests;
4. builds the landing page and JupyterLite site;
5. uploads the Pages artifact;
6. deploys automatically from the default branch.

The expected public URL shape is:

```text
https://ORG_OR_USERNAME.github.io/forensic-ml/
```
