# LibRA Slurm Pipeline Generator

Automated pipeline generation for iterative deconvolution using RoadRunner (GPU gridding), Dale (normalization), and Hummbee (deconvolution).

## Pipeline Architecture

Each iteration consists of:
1. **3 parallel GPU jobs** (RoadRunner): Generate residual, PSF, and weight images
2. **1 CPU job** (waits for GPU jobs): Normalize images (Dale) → Deconvolve (Hummbee)
3. **Next iteration** waits for previous iteration's CPU job to complete

```
Iter 0: [GPU: residual] ─┐
        [GPU: psf]      ─┼─→ [CPU: normalize + deconvolve] ─→ Iter 1: ...
        [GPU: weight]   ─┘
```

## Quick Start

### Development/Local Testing Workflow

Generate and inspect scripts **without** needing binaries or data:

```bash
cd scripts/pipeline/slurm_generator

# 1. Set validate_paths: false in imaging_config.yaml
# 2. Fill in imaging parameters (can use placeholder paths)

# 3. Generate scripts
python generate_pipeline.py imaging_config.yaml

# 4. Inspect generated scripts
ls pipeline_work/slurm_scripts/
cat pipeline_work/submit_pipeline.sh
cat pipeline_work/slurm_scripts/libra_cfgen.sh  # if coyote enabled
cat pipeline_work/slurm_scripts/libra_iter0_residual.sh
```

### Production/Cluster Workflow

Run with validation and submit:

```bash
# 1. Set validate_paths: true in imaging_config.yaml
# 2. Fill in real paths (binaries, libraries, data)

# 3. Verify environment
python verify_binaries.py imaging_config.yaml

# 4. Generate pipeline (with validation)
python generate_pipeline.py imaging_config.yaml

# 5. Review generated scripts
cat pipeline_work/submit_pipeline.sh

# 6. Submit to cluster
./pipeline_work/submit_pipeline.sh
```

## Validation Strategy

The pipeline uses a **two-mode** approach for flexibility:

### Development Mode (`validate_paths: false`)
- **Purpose**: Generate and inspect scripts locally without cluster access
- **Skips**: Path existence checks, binary verification, data validation
- **Allows**: Placeholder/empty paths in config
- **Use when**:
  - Testing configuration changes
  - Reviewing generated scripts
  - Developing on laptop before cluster deployment
  - Committing scripts to version control

### Production Mode (`validate_paths: true`)
- **Purpose**: Validate environment before submission
- **Checks**: Binary paths, library paths, CASAPATH, data files
- **Requires**: All paths must exist and be valid
- **Use when**:
  - Running on actual cluster
  - Submitting real jobs
  - Final validation before production runs

**Example workflow:**
```bash
# Local: Development mode
vim imaging_config.yaml  # set validate_paths: false
python generate_pipeline.py imaging_config.yaml
git add pipeline_work/slurm_scripts/
git commit -m "Add pipeline configuration"

# Cluster: Production mode
vim imaging_config.yaml  # set validate_paths: true, fill real paths
python verify_binaries.py imaging_config.yaml  # ✓ All checks passed
python generate_pipeline.py imaging_config.yaml
./pipeline_work/submit_pipeline.sh
```

## Configuration File

### Pipeline Control

#### pipeline
- `n_iterations`: Number of major cycle iterations
- `output_dir`: Directory for final image products
- `work_dir`: Directory for slurm scripts and logs
- `validate_paths`: **Important!** Set to `false` for local script generation/testing, `true` for production

### Required Fields

#### environment
- `bin_dir`: Path to binaries (or use separate `bin_dir_gpu`/`bin_dir_cpu`)
- `lib_dir`: Path to libraries (or use separate `lib_dir_gpu`/`lib_dir_cpu`)
- `casapath`: Path to CASA data repository
- `bin_dir_gpu`: (Optional) GPU-specific binaries
- `lib_dir_gpu`: (Optional) GPU-specific libraries
- `bin_dir_cpu`: (Optional) CPU-specific binaries
- `lib_dir_cpu`: (Optional) CPU-specific libraries

#### slurm
Two subsections: `gpu` and `cpu`

**GPU section:**
- `partition`: Slurm partition (see Available Partitions below)
- `nodes`: Number of nodes (typically 1)
- `ntasks_per_node`: Tasks per node (typically 1)
- `gpus_per_node`: GPUs to request
- `gpu_arch`: GPU architecture (`v100`, `v100s`, `a100`, `h200`, `l40s`, `gh200`)
- `time`: Wall time (format: `HH:MM:SS` or `D-HH:MM:SS`)
- `mem`: Memory per node (e.g., `64G`)

**CPU section:**
- Same as GPU but without `gpu_arch` and `gpus_per_node`

**Common fields:**
- `account`: Slurm account for charging
- `qos`: Quality of service
- `email`: Email for job notifications
- `mail_type`: When to send email (`END,FAIL`, `ALL`, `BEGIN,END,FAIL`, etc.)

#### data
- `vis`: Path to measurement set
- `imagename_base`: Base name for output images

### Available Partitions

From your cluster (`sinfo`):

| Partition | Time Limit | GPUs | Notes |
|-----------|------------|------|-------|
| `scavenger` | 7 days | Yes | Low priority, preemptible |
| `general-short` | 4 hours | Yes | Non-preemptible, fast turnaround |
| `general-long` | 7 days | No | CPU only, long jobs |
| `general-long-gpu` | 7 days | Yes | GPU + long runtime |
| `general-long-bigmem` | 7 days | No | High memory nodes |

### GPU Architectures

| Architecture | GPUs/Node | Nodes | Example Nodes |
|--------------|-----------|-------|---------------|
| `v100` | 8 | nvl-[000-007] | Volta generation |
| `v100s` | 4 | nvf-[000-020] | Volta with more memory |
| `a100` | 4 | nal-*, nif-* | Ampere generation |
| `h200` | 4-8 | neh-*, nfh-* | Hopper generation, latest |
| `l40s` | 8 | nel-[000-001] | Ada Lovelace |
| `gh200` | 1 | nch-[000-003] | Grace-Hopper superchip |

**Binary compatibility**: Build binaries for V100 (Volta 70) or higher for forward compatibility.

### Coyote (Optional CF Generation)

Enable convolution function generation and filling before imaging:

#### coyote
- `enabled`: Set to `true` to enable CF generation and filling pipeline
- `generate`: Configuration for CF generation job (single CPU, serial)
  - `partition`: Partition for CF generation
  - `time`: Wall time for generation (e.g., `"00:30:00"`)
  - `mem`: Memory for generation job
- `fillcf`: Configuration for CF filling (parallel array job)
  - `partition`: Partition for CF filling
  - `nprocs`: Number of parallel array tasks (e.g., 40)
  - `time`: Wall time per fillcf task
  - `mem`: Memory per fillcf task
- `cfcache`: Path to CF cache directory (required if enabled)
- Other parameters: See inline comments in `imaging_config.yaml`

**Pipeline flow with coyote enabled:**
```
[CF Generation] → [CF Filling Array] → [Imaging Iterations]
  (1 CPU job)      (N parallel tasks)     (GPU + CPU jobs)
```

**Typical CF counts:**
- 256-512 CFs: Set `nprocs: 20-40`
- 1K-4K CFs: Set `nprocs: 40-80`
- 8K-32K CFs: Set `nprocs: 100-200`

## Example Configurations

### Short test run (4 hours, V100)
```yaml
slurm:
  gpu:
    partition: "general-short"
    gpu_arch: "v100"
    time: "00:30:00"
  cpu:
    partition: "general-short"
    time: "00:15:00"
```

### Production run (7 days, A100)
```yaml
slurm:
  gpu:
    partition: "general-long-gpu"
    gpu_arch: "a100"
    time: "2-00:00:00"
  cpu:
    partition: "general-long"
    time: "1-00:00:00"
```

### Scavenger (opportunistic, preemptible)
```yaml
slurm:
  gpu:
    partition: "scavenger"
    gpu_arch: "v100s"
    time: "12:00:00"
  cpu:
    partition: "scavenger"
    time: "06:00:00"
```

## Pipeline Parameters

### RoadRunner (GPU Gridding)
Key parameters in `roadrunner` section:
- `imsize`: Image size in pixels
- `cell`: Cell size in arcseconds
- `weighting`: `natural`, `uniform`, or `briggs`
- `robust`: Briggs robustness parameter (-2 to 2)
- `wprojplanes`: Number of w-projection planes
- `gridder`: `awphpg` (GPU) or `awproject` (CPU)
- `cfcache`: Path to convolution function cache

### Dale (Normalization)
- `pblimit`: Primary beam cutoff threshold
- `computepb`: Whether to compute primary beam

### Hummbee (Deconvolution)
- `deconvolver`: `hogbom`, `clark`, `multiscale`, `mtmfs`, `asp`
- `nterms`: Number of Taylor terms (for `mtmfs`)
- `gain`: Loop gain
- `threshold`: Stopping threshold (Jy)
- `nsigma`: Threshold in sigma (requires `.pb` file)
- `cycleniter`: Max iterations per major cycle
- `scales`: Scales for multiscale deconvolution (pixels)

## Output Structure

```
pipeline_work/
├── logs/
│   ├── libra_iter0_residual_<jobid>.log
│   ├── libra_iter0_psf_<jobid>.log
│   ├── libra_iter0_weight_<jobid>.log
│   ├── libra_iter0_deconv_<jobid>.log
│   └── ...
├── slurm_scripts/
│   ├── libra_iter0_residual.sh
│   ├── libra_iter0_psf.sh
│   ├── libra_iter0_weight.sh
│   ├── libra_iter0_deconv.sh
│   └── ...
└── submit_pipeline.sh
```

Image outputs go to locations specified by `imagename_base`:
- `<imagename_base>_iter0.residual`
- `<imagename_base>_iter0.psf`
- `<imagename_base>_iter0.weight`
- `<imagename_base>_iter0.model`
- etc.

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <jobid>

# View live log
tail -f pipeline_work/logs/libra_iter0_residual_<jobid>.log

# Cancel job
scancel <jobid>

# Cancel all your jobs
scancel -u $USER
```

## Troubleshooting

### Binary not found
- Check `bin_dir` paths in config
- Run `verify_binaries.py` to diagnose
- Ensure binaries are executable (`chmod +x`)

### Library loading errors
- Verify `lib_dir` contains required `.so` files
- Check `LD_LIBRARY_PATH` in generated scripts
- Build might be for wrong GPU architecture

### Job pending indefinitely
- Wrong partition for requested resources
- Requested GPU type not available in partition
- Account or QOS issues

### Out of memory
- Increase `mem` in config
- Use `general-long-bigmem` partition for large jobs
- Reduce `imsize` or other memory-intensive parameters

### Job fails with CUDA errors
- GPU architecture mismatch (binary vs hardware)
- Check you're requesting correct `gpu_arch`
- Rebuild binaries for target architecture

## Files

- `imaging_config.yaml`: Configuration file with inline documentation
- `config_parser.py`: Configuration parser and validator
- `verify_binaries.py`: Verify binary paths and environment
- `generate_pipeline.py`: Generate slurm job scripts
- `README.md`: This file
