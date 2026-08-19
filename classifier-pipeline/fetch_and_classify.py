"""
Fetch newly-released JWST preview images from MAST that haven't been
processed yet, classify them with the trained galaxy classifier model,
and write results to a JSON file the frontend can read.

"Newly released" / "not yet processed" is tracked via manifest.json in
this same directory - a list of MAST observation IDs already seen. Each
run only classifies observations not already in that manifest, then
adds them to it. This is what guarantees the site shows freshly-sorted
data rather than replaying the same batch.

Usage:
    python fetch_and_classify.py --max_new 10

Requires: astroquery, torch, torchvision, pillow, requests
    pip install astroquery torch torchvision pillow requests

IMPORTANT CAVEAT (read before trusting the output):
The classifier was trained on Galaxy Zoo (SDSS-style, ground-based)
images. Real JWST previews differ meaningfully in resolution, noise,
and appearance. Predictions here are real model output, but their
accuracy on this genuinely different image distribution has not been
separately validated - treat confidence scores as illustrative of the
pipeline working end-to-end, not as validated scientific classifications.
This is a known, explainable limitation worth stating explicitly rather
than hiding.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests
import torch
from PIL import Image

# Prevent astropy from trying to download Earth-rotation reference data
# (IERS bulletins) from a remote server on first use of Time.now() etc.
# That download can hang indefinitely in CI/sandboxed environments with
# restricted network access - this was the actual cause of runs getting
# stuck at what looked like a stalled progress bar. We don't need
# leap-second precision here, so the bundled offline data is fine.
from astropy.utils import iers
iers.conf.auto_download = False

# Reuse the same model architecture + preprocessing as training, so
# inference exactly matches what the model was trained on.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_model, get_transforms  # noqa: E402

CLASS_NAMES = ["elliptical", "spiral", "other"]

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "manifest.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "jwst_results.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "best_model.pt")
THUMBS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "thumbnails")


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return set(json.load(f))
    return set()


def save_manifest(seen_ids):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(sorted(seen_ids), f, indent=2)


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def query_new_jwst_previews(already_seen, max_new, lookback_days=30):
    """
    Query MAST for public JWST observations released in the last
    `lookback_days` days, and return preview image URLs for ones not
    already in `already_seen`, newest first.

    The date restriction matters for more than just "newest" framing:
    without it, this queries JWST's entire multi-year public archive
    (tens of thousands of records) before filtering, which is slow
    enough to risk timing out in CI. Narrowing to a recent window keeps
    this fast and is also just the correct thing to ask for.
    """
    from astropy.time import Time
    from astroquery.mast import Observations

    mjd_now = Time.now().mjd
    mjd_min = mjd_now - lookback_days

    print(f"Querying MAST for JWST observations released in the last {lookback_days} days...")
    obs = Observations.query_criteria(
        obs_collection="JWST",
        dataproduct_type="image",
        intentType="science",
        t_obs_release=[mjd_min, mjd_now],
    )
    print(f"MAST query returned {len(obs)} observations in that window.")
    obs.sort("t_obs_release", reverse=True)

    candidates = []
    for row in obs:
        obs_id = str(row["obsid"])
        if obs_id in already_seen:
            continue
        candidates.append({
            "obs_id": obs_id,
            "target_name": row["target_name"],
            "instrument": row["instrument_name"],
            "obs_release": str(row["t_obs_release"]),
        })
        if len(candidates) >= max_new:
            break

    print(f"{len(candidates)} of those are new (not in manifest.json).")

    # Fetch preview product URLs for each candidate. MAST's exact product
    # naming varies more than expected - rather than assume one exact
    # string match, search case-insensitively across the columns that
    # commonly indicate a viewable preview image, and log what's actually
    # available when nothing matches, so a failure here is diagnosable
    # instead of silently returning nothing.
    results = []
    for cand in candidates:
        products = Observations.get_product_list(cand["obs_id"])

        preview_row = None
        for row in products:
            subgroup = str(row.get("productSubGroupDescription") or "").lower()
            ptype = str(row.get("productType") or "").lower()
            uri = str(row.get("dataURI") or "").lower()
            if "preview" in subgroup or "preview" in ptype or uri.endswith((".jpg", ".jpeg", ".png")):
                preview_row = row
                break

        if preview_row is None:
            available = sorted(set(
                str(row.get("productSubGroupDescription") or "(none)") for row in products
            ))
            print(f"  No preview found for {cand['obs_id']} ({cand['target_name']}). "
                  f"Available productSubGroupDescription values: {available}")
            continue

        preview_uri = preview_row["dataURI"]
        preview_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={preview_uri}"
        cand["preview_url"] = preview_url
        results.append(cand)

    return results


def classify_image(model, transform, device, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
    return CLASS_NAMES[pred_idx], round(probs[pred_idx].item() * 100, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new", type=int, default=10,
                         help="Max number of new observations to fetch/classify this run")
    parser.add_argument("--keep_results", type=int, default=60,
                         help="Max number of results to keep in the results file (older ones trimmed)")
    parser.add_argument("--lookback_days", type=int, default=30,
                         help="Only consider MAST observations released within this many days")
    args = parser.parse_args()

    os.makedirs(THUMBS_DIR, exist_ok=True)

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    _, eval_transform = get_transforms()

    seen_ids = load_manifest()
    print(f"{len(seen_ids)} observations already processed previously.")

    print("Querying MAST for new JWST observations...")
    new_obs = query_new_jwst_previews(seen_ids, args.max_new, lookback_days=args.lookback_days)
    print(f"Found {len(new_obs)} new, unprocessed observations.")

    results = load_results()

    for cand in new_obs:
        try:
            thumb_path = os.path.join(THUMBS_DIR, f"{cand['obs_id']}.jpg")
            resp = requests.get(cand["preview_url"], timeout=30)
            resp.raise_for_status()
            with open(thumb_path, "wb") as f:
                f.write(resp.content)

            category, confidence = classify_image(model, eval_transform, device, thumb_path)

            results.insert(0, {
                "obs_id": cand["obs_id"],
                "target_name": cand["target_name"],
                "instrument": cand["instrument"],
                "obs_release": cand["obs_release"],
                "category": category,
                "confidence": confidence,
                "thumbnail": f"thumbnails/{cand['obs_id']}.jpg",
                "processed_at": datetime.now(timezone.utc).isoformat(),
            })
            seen_ids.add(cand["obs_id"])
            print(f"  {cand['obs_id']} ({cand['target_name']}) -> {category} ({confidence}%)")

        except Exception as e:
            print(f"  Skipped {cand['obs_id']}: {e}")

    results = results[:args.keep_results]

    save_results(results)
    save_manifest(seen_ids)
    print(f"Done. {len(results)} results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
