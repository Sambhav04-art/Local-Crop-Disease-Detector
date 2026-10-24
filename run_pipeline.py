"""One command does everything:  python run_pipeline.py   (add --smoke for a 5-minute dry run)"""
import argparse, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--skip-prepare", action="store_true")
ap.add_argument("--skip-train", action="store_true"); ap.add_argument("--data-dir")
a = ap.parse_args()
run = lambda *m: subprocess.run([sys.executable, "-m", *m], check=True)
extra = ["--smoke"] if a.smoke else []
if not a.skip_prepare:
    run("cropdx.prepare_data", *(["--data-dir", a.data_dir] if a.data_dir else []))
if not a.skip_train:
    run("cropdx.train", *extra)
run("cropdx.evaluate", *extra)
run("cropdx.export")
run("cropdx.report")
print("\nAll done. Start the API:  uvicorn api.main:app --port 8000")
