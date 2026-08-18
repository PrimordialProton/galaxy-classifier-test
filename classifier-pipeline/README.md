# Classifier pipeline

Fetches newly-released JWST preview images from MAST that haven't been
processed before, classifies them with the trained model from this repo,
and writes results to `../data/jwst_results.json` for the satellite
tracker frontend to display.

## How "unsorted at time of selection" is guaranteed

`manifest.json` in this folder tracks every MAST observation ID already
processed. Each run:
1. Queries MAST for the newest public JWST observations
2. Filters out anything already in the manifest
3. Classifies only what's left (genuinely new since last run)
4. Adds those IDs to the manifest

This is the mechanism that prevents the site from just replaying the same
batch of images repeatedly.

## Running it

```bash
pip install -r requirements.txt
python fetch_and_classify.py --max_new 10
```

This requires `../outputs/best_model.pt` to exist (the trained model from
the main classifier project).

## Running it automatically (recommended)

`.github/workflows/classify.yml` (at the repo root) runs this script once
a day via GitHub Actions and commits the updated `data/jwst_results.json`
back to the repo automatically - no server to host or maintain. You can
also trigger it manually from the Actions tab in GitHub
("Run workflow" button) instead of waiting for the schedule.

## Connecting the frontend

The satellite tracker site fetches `data/jwst_results.json` directly from
this repo's raw GitHub URL (see `JWST_RESULTS_URL` near the top of the
frontend's `<script>` section). Update that URL to point at your own
GitHub username/repo once you've pushed this pipeline.

## Known limitation - read before trusting the output

The model was trained on Galaxy Zoo (SDSS, ground-based) images. Real
JWST previews look meaningfully different: different resolution, noise
characteristics, and imaging filters. The pipeline runs real inference on
real new images end-to-end, but classification accuracy on this
out-of-training-distribution data has not been separately validated.
Treat this as a working pipeline demonstration rather than a scientifically
validated classifier - and consider it a natural "v3" to actually go
measure that gap (e.g. hand-labeling a small sample of JWST previews and
checking agreement with the model's predictions).
