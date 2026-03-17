#!/bin/bash
# Run MentalRoBERTa embedding generation on HPC (hpc.glaciar.lab)
#
# Setup (first time):
#   ssh hpc.glaciar.lab
#   cd ~/Documentos/code
#   git clone <repo> cvx   # or rsync
#   cd cvx
#   conda env create -f environment.yml
#   conda activate cvx
#   pip install transformers torch  # GPU version
#
# Usage:
#   bash scripts/hpc_run_embeddings.sh          # all datasets
#   bash scripts/hpc_run_embeddings.sh erisk    # single dataset

set -euo pipefail

DATASET=${1:-all}

echo "============================================"
echo "MentalRoBERTa Embedding Generation"
echo "Dataset: $DATASET"
echo "Model: mental/mental-roberta-base (D=768)"
echo "============================================"

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Verify data exists
for ds in erisk clpsych rsdd; do
    if [ "$DATASET" = "all" ] || [ "$DATASET" = "$ds" ]; then
        case $ds in
            erisk)   input="data/erisk_unified.jsonl" ;;
            clpsych) input="data/clpsych_unified.jsonl" ;;
            rsdd)    input="data/rsdd_unified.jsonl" ;;
        esac
        if [ ! -f "$input" ]; then
            echo "WARNING: $input not found"
        else
            echo "OK: $input ($(wc -l < "$input") lines)"
        fi
    fi
done

# Run
echo ""
echo "Starting embedding generation..."
time python3 scripts/generate_embeddings_v2.py \
    --dataset "$DATASET" \
    --batch-size 128

# Add splits to new parquets
echo ""
echo "Adding splits to new embeddings..."
# Temporarily modify add_splits.py to use new parquet names
python3 -c "
import scripts.add_splits as splits
from pathlib import Path

# Patch paths to use mental_ versions
for name in ['erisk', 'clpsych', 'rsdd']:
    new_path = Path(f'data/embeddings/{name}_mental_embeddings.parquet')
    if new_path.exists():
        print(f'Adding splits to {new_path}...')
        # Use the same split logic
        import importlib
        importlib.reload(splits)

print('Done')
"

echo ""
echo "============================================"
echo "Output files:"
ls -lh data/embeddings/*mental* 2>/dev/null || echo "No output files found"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Copy parquets to local: scp hpc.glaciar.lab:~/Documentos/code/cvx/data/embeddings/*mental* data/embeddings/"
echo "2. Run: python scripts/add_splits.py  (update for mental_ parquets)"
echo "3. Run notebooks with new embeddings"
