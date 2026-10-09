#!/bin/bash
# Polyphe backfill allocates an exclusive node; this probe uses only GPU 0.
# Upload package contents into TASK_RUN before submitting. No training source needed.
#SBATCH --job-name=core_dlfw_ci-memtracker.framework
#SBATCH --account=core_dlfw_ci
#SBATCH --partition=backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:08:00
#SBATCH --output=/lustre/fsw/core_dlfw_ci/elfieg/memtracker-framework-20261008/%j.log
set -euo pipefail
TASK_RUN=/lustre/fsw/core_dlfw_ci/elfieg/memtracker-framework-20261008
TASK_IMAGE=/lustre/fsw/core_dlfw_ci/elfieg/kimi-k3-distmuon-20260929/containers/kimi-distmuon-hybridep.sqsh
TASK_RUNTIME=/lustre/fsw/core_dlfw_ci/elfieg/memtracker-calibration-20261007/container-runtime.sh
srun --ntasks=1 --container-image="$TASK_IMAGE" \
  --container-mounts="$TASK_RUN:/probe,$TASK_RUNTIME:/runtime.sh:ro" --container-workdir=/probe bash -c '
set -euo pipefail
source /runtime.sh
export PYTHONPATH=/probe/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
for TASK_FAMILY in mlp conv; do
  for TASK_ALLOCATOR in fixed expandable; do
    TASK_ARGS=()
    if [[ "$TASK_ALLOCATOR" = expandable ]]; then TASK_ARGS+=(--expandable); fi
    for TASK_MODE in fake cuda; do
      python examples/framework_phase_demo.py --mode "$TASK_MODE" --family "$TASK_FAMILY" \
        --width 32 --batch 2 --steps 3 --full-ac "${TASK_ARGS[@]}" \
        --output "results/$TASK_FAMILY-$TASK_ALLOCATOR-$TASK_MODE.json.gz"
    done
    python -m memtracker_nccl.phase_memory "results/$TASK_FAMILY-$TASK_ALLOCATOR-cuda.json.gz" \
      --estimate "results/$TASK_FAMILY-$TASK_ALLOCATOR-fake.json.gz" \
      --output "results/$TASK_FAMILY-$TASK_ALLOCATOR-comparison.json.gz"
  done
done
'
