# NDBI021 Lab 2 — Sparse representations for movie preference elicitation

This lab explores how a sparse representation of movies can be used to ask users about their preferences and generate personalized recommendations. The implementation runs inside **EasyStudy**, using its **FastCompare** plugin to administer a study and compare recommendation methods.

- **Source code:** https://github.com/lpeska/EasyStudy\_ndbi021/tree/ndbi021  
- **Poster images:** https://www.ksi.mff.cuni.cz/\~peska/vyuka/ndbi021/2026\_2027\_labs/labs2/
- **MovieLens dataset:** https://files.grouplens.org/datasets/movielens/ml-latest.zip

## 1\. Installation

### Requirements

* Git and a Python environment (Conda is recommended; Python 3.11 is a practical starting point).

Clone the course repository and select its `ndbi021` branch:

```bash
git clone --branch ndbi021 https://github.com/lpeska/EasyStudy\_ndbi021.git
cd EasyStudy\_ndbi021
```

If the checkout attempts to fetch large Git LFS objects and fails, clone with LFS downloads disabled and obtain the poster archive separately from the course page:

```bash
# Linux/macOS
GIT\_LFS\_SKIP\_SMUDGE=1 git clone --branch ndbi021 https://github.com/lpeska/EasyStudy\_ndbi021.git
```

On Windows PowerShell, set `$env:GIT\_LFS\_SKIP\_SMUDGE = '1'` before running `git clone`.

Create a dedicated environment:

```bash
conda create -n easystudy-sae python=3.11 -y
conda activate easystudy-sae
cd server
python -m pip install -r local\_requirements.txt
python -m pip install -r requirements\_sae\_lab.txt
python -m pip install jupyter ipykernel numpy pandas
python -m ipykernel install --user --name easystudy-sae --display-name "EasyStudy SAE"
```

### MovieLens data and posters

Download **MovieLens Latest** from https://files.grouplens.org/datasets/movielens/ml-latest.zip and extract its CSV files directly to:

```text
server/static/datasets/ml-latest/movies.csv
server/static/datasets/ml-latest/ratings.csv
server/static/datasets/ml-latest/links.csv
```

Obtain the movie poster archive from the [course materials](https://www.ksi.mff.cuni.cz/~peska/vyuka/ndbi021/2026_2027_labs/labs2/) and extract the required JPG files to:

```text
server/static/datasets/ml-latest/img/<movieId>.jpg
```

### Start the application

From `server/`, run:

```bash
flask --debug run
```

If your environment needs database initialization or additional environment variables, follow the original repository's setup instructions. Open the local address printed by Flask (/administration).

## 2\. Create and run a study

In EasyStudy administration, create a **FastCompare** study using the MovieLens data loader. Select **SAE choice groups** as the preference elicitation method and include **SAE profile** and optionally **EASE** as recommendation algorithms. Configure the available parameters and initialize the study.

Initialization prepares a filtered dataset, trains or loads the underlying models, and caches the results under:

```text
server/cache/fastcompare/<study-guid>/
```

Model initialization may take several minutes. After initialization, open the study's participant link. Participants first answer questions about two groups of movies, then interact with recommendation lists. If you change training or question-selection code, reinitialize the study rather than assuming existing cached objects will be updated.

## 3\. How the SAE-based method works

### Movie representations

Movies are first represented by dense collaborative embeddings (ELSA). A sparse autoencoder (SAE), trained using Compresso, transforms them into a sparse feature matrix:

\[
Z \\in \\mathbb{R}^{n \\times d}, \\qquad z\_{ij}=\\text{activation of feature }j\\text{ for movie }i.
]

Most activations are zero. In the signed SAE configuration, activations can be positive or negative; these are two directions of the **same** feature, not necessarily positive and negative sentiment. The representation size, sparsity, training duration, and SAE variant are configurable.

### Choosing preference questions

For every SAE feature `j`, define:

\[
P\_j={i:z\_{ij}>0}, \\qquad N\_j={i:z\_{ij}<0}.
]

A feature is eligible only when **both** groups have at least `group\_size` movies. Eligible features are ranked by their smaller group:

\[
\\operatorname{coverage}(j)=\\min(|P\_j|,|N\_j|).
]

The highest-scoring features are selected for questions. For each selected feature, the system randomly samples `group\_size` distinct movies from each side, using a fixed seed for reproducibility. The questions are precomputed rather than adapted to individual answers.

### Recording elicitation feedback

For each question, participants can choose **positive group**, **negative group**, **both**, or **neither**. The SAE preference profile is a vector `w\_explicit` with one entry per feature. Its update is deliberately simple:

|Choice|SAE profile update|Movies recorded as liked|
|-|-|-|
|Positive group|`w\_explicit\[j] += 1`|Positive-group movies|
|Negative group|`w\_explicit\[j] -= 1`|Negative-group movies|
|Both groups|No feature update|Movies from both groups|
|Neither group|No feature update|None|

Thus, the feature-level profile is **explicit only**. The item-level liked history is also passed into FastCompare's standard recommendation pathway, making it available to recommenders such as EASE. Movies marked as liked during elicitation are excluded from later recommendation candidates.

### Generating recommendations

The SAE recommender uses the elicited feature profile together with positive movie selections made **later**, during recommendation rounds. Let `e` be the L2-normalized explicit feature profile, and let `l` be the L2-normalized mean SAE code of the later-liked movies. When both are available:

\[
w=\\gamma e+(1-\\gamma)l, \\qquad \\operatorname{score}(i)=w^\\top z\_i.
]

The `gamma` parameter controls how much elicitation knowledge is retained relative to later selections (`gamma=1`: elicitation only; `gamma=0`: later selections only). If only one source of evidence is available, that source is used. Already displayed or selected movies are filtered from the recommendations. Standard recommenders such as EASE instead operate on the accumulated liked-item history.

## 4\. Parameters and code orientation

The main SAE implementation is under:

```text
server/plugins/fastcompare/algo/sae\_lab/
```

|Component|Responsibility|
|-|-|
|`models.py`|Learn dense item representations and SAE codes|
|`core.py`|Select feature groups, update preferences, and score movies|
|`adapter.py`|Connect the SAE methods to FastCompare and expose configuration|
|`export.py` (where available)|Write portable data for notebook inspection|
|`tests/test\_core.py`|Test the core selection and preference logic|

Typical elicitation settings include `factors` (dense embedding dimensions), `epochs` (dense-model training), `sae\_dim` (number of SAE features), `sae\_k` (SAE sparsity), `sae\_epochs`, `variant` (e.g. signed AbsTopK), `group\_size`, and `questions`. The SAE recommender exposes `gamma`. Consult the administration form's parameter tooltips for descriptions and the defaults used by your checkout.

The central functions to inspect are `select\_groups`, `update\_preferences`, and `recommend` in `core.py`. They contain the main preference-elicitation and ranking logic; the surrounding FastCompare integration handles data loading, participant sessions, presentation, and persistence.

## 5\. Inspect the learned representation

After initializing a study, portable artifacts may be available at:

```text
server/cache/fastcompare/<study-guid>/sae\_artifacts/
    sae\_codes.npz
    sae\_movies.csv
    sae\_metadata.json
```

Open the SAE representation explorer notebook and set its `ARTIFACT\_DIR` to this folder. It works with NumPy and pandas and does not need to import the EasyStudy Flask application.

The explorer allows you to inspect strongly positive and negative movies for a feature, find similar movies using cosine similarity in SAE space, check movie popularity, and count the positive and negative activations of each feature. You can also **manually label the positive and negative sides of a feature** after inspecting representative movies. Labels are stored in a CSV file (e.g. `sae\_neuron\_labels.csv`) for later analysis; they do not change the learned model or participant questions.

Be careful with IDs: `item\_index` refers to a row of `Z`, while `item\_id` is the original MovieLens ID used by the dataset and poster filenames.

## 6\. Ideas for exploration

Try to identify which SAE features have interpretable positive and negative groups. Consider whether high-coverage features are necessarily useful for elicitation, how the chosen movie examples affect a question, and whether the simple feature updates capture the information available in each answer. You can also examine how recommendations evolve after later movie selections, and what kinds of changes to selection, feedback, or ranking might be worth evaluating.

## 7\. Troubleshooting

* **Missing posters:** Check `server/static/datasets/ml-latest/img/<movieId>.jpg` and confirm that the image filenames use original MovieLens IDs.
* **No eligible signed features:** Inspect how many movies activate each SAE feature positively and negatively. The SAE variant and `group\_size` affect eligibility.
* **Old questions after changing code:** Reinitialize the study so cached question groups are recomputed.
* **Notebook import errors:** Use the exported `sae\_codes.npz` and `sae\_movies.csv` instead of loading the full EasyStudy cache pickle.
* **Missing portable artifacts:** Confirm that the initialized study's cache includes `sae\_artifacts`; if not, inspect the repository's SAE export integration.

\---

**Repository:** https://github.com/lpeska/EasyStudy\_ndbi021/tree/ndbi021  
**Course resources:** https://www.ksi.mff.cuni.cz/\~peska/vyuka/ndbi021/2026\_2027\_labs/labs2/
