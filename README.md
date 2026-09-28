<p align="center">
  <img src="onlinenet-hero.png" alt="" width="900">
</p>

<h1 align="center">OnlineNet</h1>

<p align="center"><strong>Strict Online Learning for Wild Streaming Data</strong></p>

<table align="center">
  <tr>
    <td align="center"><strong>Best aggregate performance</strong><br><sub>Four cumulative metrics across 81 streaming conditions</sub></td>
    <td align="center"><strong>0.5% parameters</strong><br><sub>On Spam, relative to the strongest competing baseline</sub></td>
    <td align="center"><strong>Lower per-instance runtime</strong><br><sub>On three representative synthetic streams versus the strongest competing baseline</sub></td>
  </tr>
</table>

## Directory structure

```text
.
|-- Code/
|   |-- main.py                         # experiment entry point
|   |-- read_results.py                 # aggregate saved results
|   |-- Config/                         # model hyperparameters
|   |-- DataCode/                       # data loading and masking
|   |-- Models/                         # OnlineNet and baseline implementations
|   |-- Results/                        # saved experiment outputs
|-- Data/                               # real and synthetic datasets
`-- requirements.txt                    # reference Python environment
```

The primary model is implemented in `Code/Models/OnlineNetTau025.py`. Its
strict test-then-train procedure is implemented in
`Code/Models/run_OnlineNetTau025.py`.

## Environment

The reference environment uses Python 3.9 and PyTorch 2.2.2. Install the
dependencies from the archive root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

All reported experiments use CPU execution by default (`use_cuda=False`).

## Running an experiment

Run commands from the `Code` directory. For example, the following command
evaluates OnlineNet on WBC with 75% feature availability:

```bash
cd Code
python main.py --methodname OnlineNetTau025 --dataname wbc --probavailable 0.75 --type noassumption
```

To run WDBC under the same setting:

```bash
python main.py --methodname OnlineNetTau025 --dataname wdbc --probavailable 0.75 --type noassumption
```

The experiment entry point supports exactly
`["olvf", "olifl", "ovfm", "orf3v", "auxdrop", "HBP", "adam_HI2", "MODL", "OnlineNetTau025"]`.
Baselines use method-specific
settings from `Code/Config/config_models.py`; Aux-Drop additionally requires a
valid base-feature, dummy-feature, or architecture-change setting. Available
dataset names can be listed with:

```bash
python main.py --help
```

The Aux-Drop configuration used for the grouped real-data experiments is:

```bash
python main.py \
  --dataname a \
  --methodname auxdrop \
  --type basefeatures \
  --ifAuxDropNoAssumpArchChange True
```

Here, `--dataname a` runs the configured collection of real-world datasets. It
can be replaced by a single dataset name for an individual run.

Model-specific hyperparameters and the number of repeated runs are defined in
`Code/Config/config_models.py`. Random seeds are set before data preparation and
at the start of each model run.

## Results

Experiment outputs are stored under:

```text
Code/Results/<setting>/<method>/<dataset>_prob_<availability>.data
```

For example, summarize the WBC result above with:

```bash
python read_results.py --methodname OnlineNetTau025 --dataname wbc --probavailable 0.75 --type noassumption
```

This command writes a CSV file containing the mean and standard deviation over
the repeated runs in the same result directory.


```bash
python -m pytest tests/test_onlinenet_tau025.py -q
```

## Data

The datasets used by the experiment scripts are included in `Data/`. Synthetic
concept-drift streams are stored in `Data/Synthetic_dataset/`. For experiments
with partially available features, the availability mask is generated at run
time using `--probavailable` and the experiment seed.

The included third-party datasets remain subject to their original terms and
licenses. Users should consult the corresponding dataset sources before reuse.

## Anonymous review

This README intentionally contains no author names, affiliations, contact
details, personal repository links, or machine-specific paths. Before upload,
the supplementary archive should exclude IDE metadata, caches, temporary
files, and local execution logs.
