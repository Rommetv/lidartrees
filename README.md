# Lidar Trees

## Quick Start
1. Create the conda env: `conda env create -f environment.yaml` (or `environment.yml` if that’s the file name).
2. If R complains about BiocManager, run once: `Rscript installpack.R`.
3. Process a tile: `Rscript process_tile.R`.
4. Extract features: `python ex_feat_all.py`.
5. Use the generated outputs in downstream notebooks (`classifier.ipynb`, `sun_exp.ipynb`, `clustering`, etc.).