set -e
PYTHONPATH=$(pwd) PROJECT_ROOT=$PYTHONPATH python src/datamodules/components/vpc25/00_prepare_anon_datasets.py base_dirs data/vpc2025_official/B3/data  data/vpc2025_official/B4/data  data/vpc2025_official/B5/data  data/vpc2025_official/T8-5/data data/vpc2025_official/T12-5/data data/vpc2025_official/T10-2/data data/vpc2025_official/T25-1/data
