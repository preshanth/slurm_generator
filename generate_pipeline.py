#!/usr/bin/env python3

import math
import sys
import os
import shlex
from pathlib import Path
from config_parser import ImagingConfig

class SlurmJobGenerator:
    def __init__(self, config: ImagingConfig):
        self.config = config
        self.work_dir = config.get_work_dir()
        self.log_dir = self.work_dir / "logs"
        self.scripts_dir = self.work_dir / "slurm_scripts"

    def _quote_cmd(self, cmd: list) -> str:
        """Properly quote command arguments that contain spaces or special chars"""
        return " ".join(shlex.quote(arg) for arg in cmd)

    def _containerize_cmd(self, cmd: list, job_type: str) -> str:
        """Wrap command with singularity exec if container configured, else quote normally"""
        container = self.config.config['environment'].get('container_image', '')

        if container:
            # Build singularity exec command
            binds = self.config.config['environment'].get('container_binds', [])
            sing_cmd = ['singularity', 'exec']

            # Add NVIDIA GPU support for GPU jobs
            if job_type == 'gpu':
                sing_cmd.append('--nv')

            # Add bind mounts
            for bind in binds:
                sing_cmd.extend(['--bind', bind])

            # Set working directory and container image
            sing_cmd.extend(['--pwd', '$PWD', container])

            # Append tool command with path inside container
            bin_dir = Path(self.config.config['environment']['bin_dir'])
            cmd[0] = str(bin_dir / cmd[0])
            sing_cmd.extend(cmd)

            # Quote everything except $PWD (needs to expand)
            return " ".join(shlex.quote(arg) if arg != '$PWD' else arg for arg in sing_cmd)
        else:
            # No container - use old method
            bin_dir = self._get_bin_dir(job_type)
            cmd[0] = str(bin_dir / cmd[0])
            return self._quote_cmd(cmd)

    def _get_bin_dir(self, job_type: str) -> Path:
        env = self.config.config['environment']
        if job_type == 'gpu':
            if env['bin_dir_gpu']:
                return Path(env['bin_dir_gpu'])
        elif job_type == 'cpu':
            if env['bin_dir_cpu']:
                return Path(env['bin_dir_cpu'])
        return Path(env['bin_dir'])

    def _get_lib_dir(self, job_type: str) -> Path:
        env = self.config.config['environment']
        if job_type == 'gpu':
            if env['lib_dir_gpu']:
                return Path(env['lib_dir_gpu'])
        elif job_type == 'cpu':
            if env['lib_dir_cpu']:
                return Path(env['lib_dir_cpu'])
        return Path(env['lib_dir'])

    def _generate_env_setup(self, job_type: str) -> str:
        container = self.config.config['environment'].get('container_image', '')

        if container:
            # In container mode, still export CASAPATH if specified
            casapath = self.config.config['environment'].get('casapath', '')
            if casapath:
                return f"export CASAPATH={casapath}\n"
            return ""

        # Non-container mode: set LD_LIBRARY_PATH and CASAPATH
        lib_dir = self._get_lib_dir(job_type)
        casapath = Path(self.config.config['environment']['casapath'])

        lines = [
            f"export LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH",
            f"export CASAPATH={casapath}",
            ""
        ]
        return "\n".join(lines)

    def _generate_sbatch_header(self, job_name: str, job_type: str, log_file: str) -> str:
        slurm_cfg = self.config.config['slurm']
        type_cfg = slurm_cfg[job_type]

        lines = [
            "#!/bin/bash --login",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={type_cfg['partition']}",
            f"#SBATCH --nodes={type_cfg['nodes']}",
            f"#SBATCH --ntasks-per-node={type_cfg['ntasks_per_node']}",
            f"#SBATCH --time={type_cfg['time']}",
            f"#SBATCH --mem={type_cfg['mem']}",
            f"#SBATCH --output={log_file}",
        ]

        if job_type == 'gpu':
            gpu_arch = type_cfg['gpu_arch']
            gpus = type_cfg['gpus_per_node']
            lines.append(f"#SBATCH --gres=gpu:{gpu_arch}:{gpus}")

        if slurm_cfg['account']:
            lines.append(f"#SBATCH --account={slurm_cfg['account']}")

        if slurm_cfg['qos']:
            lines.append(f"#SBATCH --qos={slurm_cfg['qos']}")

        if slurm_cfg['email']:
            lines.append(f"#SBATCH --mail-user={slurm_cfg['email']}")
            lines.append(f"#SBATCH --mail-type={slurm_cfg['mail_type']}")

        lines.extend(["", "set -e", "set -o pipefail", ""])
        return "\n".join(lines)

    def generate_coyote_cfgen_job(self) -> Path:
        job_name = "libra_cfgen"
        log_file = self.log_dir / f"{job_name}_%j.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        coy_cfg = self.config.config['coyote']['generate']
        slurm_cfg = self.config.config['slurm']

        lines = [
            "#!/bin/bash --login",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={coy_cfg['partition']}",
            f"#SBATCH --nodes=1",
            f"#SBATCH --ntasks-per-node=1",
            f"#SBATCH --time={coy_cfg['time']}",
            f"#SBATCH --mem={coy_cfg['mem']}",
            f"#SBATCH --output={log_file}",
        ]

        if slurm_cfg['account']:
            lines.append(f"#SBATCH --account={slurm_cfg['account']}")
        if slurm_cfg['qos']:
            lines.append(f"#SBATCH --qos={slurm_cfg['qos']}")
        if slurm_cfg['email']:
            lines.append(f"#SBATCH --mail-user={slurm_cfg['email']}")
            lines.append(f"#SBATCH --mail-type={slurm_cfg['mail_type']}")

        lines.extend(["", "set -e", "set -o pipefail", ""])
        header = "\n".join(lines)

        env_setup = self._generate_env_setup('cpu')

        cmd = self.config.build_coyote_cmd('dryrun')
        cmd_str = self._containerize_cmd(cmd, 'cpu')

        script_content = f"{header}{env_setup}echo \"Starting CF generation at $(date)\"\n{cmd_str}\necho \"Finished CF generation at $(date)\"\n"

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)

        return script_path

    def generate_coyote_fillcf_job(self) -> Path:
        job_name = "libra_fillcf"
        log_file = self.log_dir / f"{job_name}_%A_%a.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        coy_cfg = self.config.config['coyote']
        fillcf_cfg = coy_cfg['fillcf']
        slurm_cfg = self.config.config['slurm']
        nprocs = fillcf_cfg['nprocs']

        lines = [
            "#!/bin/bash --login",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={fillcf_cfg['partition']}",
            f"#SBATCH --array=0-{nprocs-1}",
            f"#SBATCH --nodes=1",
            f"#SBATCH --ntasks-per-node=1",
            f"#SBATCH --time={fillcf_cfg['time']}",
            f"#SBATCH --mem={fillcf_cfg['mem']}",
            f"#SBATCH --output={log_file}",
        ]

        if slurm_cfg['account']:
            lines.append(f"#SBATCH --account={slurm_cfg['account']}")
        if slurm_cfg['qos']:
            lines.append(f"#SBATCH --qos={slurm_cfg['qos']}")
        if slurm_cfg['email']:
            lines.append(f"#SBATCH --mail-user={slurm_cfg['email']}")
            lines.append(f"#SBATCH --mail-type={slurm_cfg['mail_type']}")

        lines.extend(["", "set -e", "set -o pipefail", ""])
        header = "\n".join(lines)

        env_setup = self._generate_env_setup('cpu')

        worker_script = self.scripts_dir / "fillcf_worker.py"
        cfcache_dir = coy_cfg['cfcache']

        # Build worker invocation with container support
        container = self.config.config['environment'].get('container_image', '')
        if container:
            bin_dir = Path(self.config.config['environment']['bin_dir'])
            coyote_bin = str(bin_dir / "coyote")
            binds = self.config.config['environment'].get('container_binds', [])
            binds_args = " ".join(f"--container_binds {shlex.quote(b)}" for b in binds)
            worker_call = f"python3 {worker_script} --cfcache_dir {cfcache_dir} --nprocs {nprocs} --coyote_bin {coyote_bin} --container {container} {binds_args}"
        else:
            bin_dir = self._get_bin_dir('cpu')
            coyote_bin = str(bin_dir / "coyote")
            worker_call = f"python3 {worker_script} --cfcache_dir {cfcache_dir} --nprocs {nprocs} --coyote_bin {coyote_bin}"

        script_content = f"""{header}{env_setup}echo "Starting CF filling task $SLURM_ARRAY_TASK_ID at $(date)"
{worker_call}
echo "Finished CF filling task $SLURM_ARRAY_TASK_ID at $(date)"
"""

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)

        self._generate_fillcf_worker(worker_script)

        return script_path

    def _generate_fillcf_worker(self, worker_path: Path):
        coy_cfg = self.config.config['coyote']
        vis = self.config.get_vis()

        cmd_base = self.config.build_coyote_cmd('fillcf')
        cmd_base = [arg for arg in cmd_base if not arg.startswith('cflist=')]
        # Skip only the binary name, keep help=noprompt
        cmd_params_list = cmd_base[1:]

        worker_content = f'''#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import sys

def distribute_cfs(cfcache_dir, nprocs, task_id):
    # Look for all CF directories (exclude WTCF - weighted CFs)
    pattern = os.path.join(cfcache_dir, '*CF*.im')
    cfs = sorted([os.path.basename(f) for f in glob.glob(pattern)
                  if os.path.isdir(f) and 'WTCF' not in os.path.basename(f)])

    print(f"Task {{task_id}}: Found {{len(cfs)}} total CFs in {{cfcache_dir}}")

    if not cfs:
        print(f"ERROR: No CFs found matching {{pattern}}")
        sys.exit(1)

    base, remainder = divmod(len(cfs), nprocs)
    quantities = [base + 1 if i < remainder else base for i in range(nprocs)]

    start = sum(quantities[:task_id])
    end = start + quantities[task_id]

    assigned = cfs[start:end]
    print(f"Task {{task_id}}: Assigned {{len(assigned)}} CFs ({{start}} to {{end}})")

    return assigned

def fill_cfs(cfs, cfcache_dir, coyote_bin, cmd_params, container=None, container_binds=None):
    if not cfs:
        print("No CFs assigned to this task")
        return

    cflist = ",".join(cfs)

    # Build command - wrap with singularity if container specified
    if container:
        cmd = ['singularity', 'exec']
        for bind in (container_binds or []):
            cmd.extend(['--bind', bind])
        cmd.extend(['--pwd', os.getcwd(), container, coyote_bin])
        cmd.extend(cmd_params)
        cmd.append(f"cflist={{cflist}}")
    else:
        cmd = [coyote_bin] + cmd_params + [f"cflist={{cflist}}"]

    print(f"\\nProcessing {{len(cfs)}} CFs:")
    for cf in cfs:
        print(f"  - {{cf}}")

    print(f"\\nExecuting command:")
    print(f"  {{' '.join(cmd)}}\\n")
    print("=" * 80)

    # Don't capture output - let it stream directly to log
    sys.stdout.flush()
    sys.stderr.flush()
    result = subprocess.run(cmd)

    print("=" * 80)
    if result.returncode != 0:
        print(f"\\nERROR: Command failed with return code {{result.returncode}}")
        sys.exit(result.returncode)

    print(f"\\nSuccessfully filled {{len(cfs)}} CFs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CF filling worker')
    parser.add_argument('--cfcache_dir', required=True, help='CF cache directory')
    parser.add_argument('--nprocs', type=int, required=True, help='Number of processes')
    parser.add_argument('--coyote_bin', required=True, help='Path to coyote binary')
    parser.add_argument('--container', default=None, help='Singularity container image')
    parser.add_argument('--container_binds', nargs='*', default=[], help='Container bind mount paths')
    args = parser.parse_args()

    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))

    # Command parameters passed from generator
    cmd_params = {cmd_params_list}

    print(f"\\n=== CF Filling Worker - Task {{task_id}} ===")
    print(f"CF Cache: {{args.cfcache_dir}}")
    print(f"Total processes: {{args.nprocs}}\\n")

    cfs = distribute_cfs(args.cfcache_dir, args.nprocs, task_id)
    fill_cfs(cfs, args.cfcache_dir, args.coyote_bin, cmd_params, args.container, args.container_binds)
'''

        worker_path.parent.mkdir(parents=True, exist_ok=True)
        with open(worker_path, 'w') as f:
            f.write(worker_content)
        worker_path.chmod(0o755)

    def generate_gpu_job(self, iteration: int, mode: str) -> Path:
        job_name = f"libra_iter{iteration}_{mode}"
        log_file = self.log_dir / f"{job_name}_%j.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        header = self._generate_sbatch_header(job_name, 'gpu', str(log_file))

        env_setup = self._generate_env_setup('gpu')

        cmd = self.config.build_roadrunner_cmd(iteration, mode)
        cmd_str = self._containerize_cmd(cmd, 'gpu')

        prep_commands = ""

        script_content = f"{header}{env_setup}echo \"Starting {job_name} at $(date)\"\n{prep_commands}{cmd_str}\necho \"Finished {job_name} at $(date)\"\n"

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)

        return script_path

    def generate_cpu_job(self, iteration: int) -> Path:
        job_name = f"libra_iter{iteration}_deconv"
        log_file = self.log_dir / f"{job_name}_%j.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        header = self._generate_sbatch_header(job_name, 'cpu', str(log_file))

        env_setup = self._generate_env_setup('cpu')

        imagename = self.config.get_imagename(iteration)

        dale_residual_cmd = self.config.build_dale_cmd(iteration, "residual")
        dale_residual_str = self._containerize_cmd(dale_residual_cmd, 'cpu')

        hummbee_cmd = self.config.build_hummbee_cmd(iteration)
        hummbee_str = self._containerize_cmd(hummbee_cmd, 'cpu')

        # Normalize model after deconvolution (creates .divmodel for next iteration)
        dale_model_cmd = self.config.build_dale_cmd(iteration, "model")
        dale_model_str = self._containerize_cmd(dale_model_cmd, 'cpu')

        base_name = self.config.get_imagename_base()

        # PSF normalization and pb generation (iter 0 only)
        psf_normalization = ""
        if iteration == 0:
            dale_psf_cmd = self.config.build_dale_cmd(iteration, "psf")
            dale_psf_str = self._containerize_cmd(dale_psf_cmd, 'cpu')
            psf_normalization = f"""echo "Normalizing PSF..."
{dale_psf_str}

"""

        # Tag cleanup unconditional — dale always sets SubType=Normalized
        tag_cleanup = f"""echo "Cleaning normalized tag from model..."
if [ -f {base_name}.model/table.info ]; then
    echo "Found {base_name}.model/table.info, removing normalized tag..."
    sed -i 's/SubType.*=.*//g' {base_name}.model/table.info
else
    echo "========================================================"
    echo "WARNING: {base_name}.model/table.info not found"
    echo "         Tag cleanup skipped — dale may fail next cycle"
    echo "========================================================"
fi
"""

        # Snapshots: residual/model/divmodel every iter; psf/weight/pb iter 0 only
        snapshots = f"""echo "Snapshotting iter{iteration} images..."
cp -r {base_name}.residual {base_name}_iter{iteration}.residual
cp -r {base_name}.model {base_name}_iter{iteration}.model
cp -r {base_name}.divmodel {base_name}_iter{iteration}.divmodel
"""
        if iteration == 0:
            snapshots += f"""cp -r {base_name}.psf {base_name}_iter0.psf
cp -r {base_name}.weight {base_name}_iter0.weight
cp -r {base_name}.pb {base_name}_iter0.pb
"""

        # Peak residual extraction and convergence tracking
        peak_file = self.work_dir / "peak_residuals.txt"
        sed_extract = r"sed 's/.*peakres=[0-9.Ee+-]*->\([0-9.Ee+-]*\)[, ].*/\1/'"

        convergence_check = ""
        if iteration >= 1:
            prev1 = iteration - 1
            convergence_check = f"""
PREV_PEAK=$(awk '/^iter{prev1} /{{print $2}}' {peak_file} 2>/dev/null || echo "")
if [ -n "$PREV_PEAK" ] && [ -n "$PEAK_RES" ]; then
    awk -v curr="$PEAK_RES" -v prev="$PREV_PEAK" 'BEGIN {{
        if (prev > 0) {{
            change = (prev - curr) / prev * 100
            printf "Convergence check: iter{iteration} peak=%.6f vs iter{prev1} peak=%.6f (%.1f%% improvement)\\n", curr, prev, change
            if (change < 5.0) {{
                print "========================================================"
                print "WARNING: Cycle {iteration} has similar peak residual as cycle {prev1}"
                print "         Deconvolution may have stalled"
                print "========================================================"
            }}
        }}
    }}' || true
fi"""

        peak_tracking = f"""HUMMBEE_TMP=$(mktemp)
{hummbee_str} 2>&1 | tee "$HUMMBEE_TMP"
PEAK_RES=$(grep "SDAlgorithmBase::deconvolve" "$HUMMBEE_TMP" 2>/dev/null | tail -1 | {sed_extract} || echo "")
rm -f "$HUMMBEE_TMP"

if [ -n "$PEAK_RES" ]; then
    echo "Peak residual after iter{iteration}: $PEAK_RES Jy/beam"
    echo "iter{iteration} $PEAK_RES" >> {peak_file}
else
    echo "========================================================"
    echo "WARNING: Could not extract peak residual from hummbee output"
    echo "========================================================"
fi
{convergence_check}"""

        script_content = f"""{header}{env_setup}echo "Starting {job_name} at $(date)"

{psf_normalization}echo "Normalizing residual..."
{dale_residual_str}

echo "Running deconvolution..."
{peak_tracking}

echo "Normalizing model (creates .divmodel)..."
{dale_model_str}

{tag_cleanup}
{snapshots}
echo "Finished {job_name} at $(date)"
"""

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)

        return script_path

    def generate_peeling_gpu_job(self, mode: str) -> Path:
        job_name = f"libra_peel_{mode}"
        log_file = self.log_dir / f"{job_name}_%j.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        header = self._generate_sbatch_header(job_name, 'gpu', str(log_file))
        env_setup = self._generate_env_setup('gpu')
        cmd = self.config.build_roadrunner_peeling_cmd(mode)
        cmd_str = self._containerize_cmd(cmd, 'gpu')

        script_content = f"{header}{env_setup}echo \"Starting {job_name} at $(date)\"\n{cmd_str}\necho \"Finished {job_name} at $(date)\"\n"

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        return script_path

    def generate_peeling_cpu_job(self) -> Path:
        job_name = "libra_peel_deconv"
        log_file = self.log_dir / f"{job_name}_%j.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        header = self._generate_sbatch_header(job_name, 'cpu', str(log_file))
        env_setup = self._generate_env_setup('cpu')

        base_name = self.config.get_peeling_imagename_base()

        dale_psf_str = self._containerize_cmd(
            self.config._build_dale_cmd_for_base(base_name, "psf"), 'cpu')

        dale_residual_str = self._containerize_cmd(
            self.config._build_dale_cmd_for_base(base_name, "residual"), 'cpu')

        hummbee_cmd = self.config.build_hummbee_cmd()
        hummbee_cmd[2] = f"imagename={base_name}"
        hummbee_cmd[3] = f"modelimagename={base_name}.model"
        mask = self.config.config['peeling'].get('mask', '')
        if mask:
            hummbee_cmd = [arg for arg in hummbee_cmd if not arg.startswith('mask=')]
            hummbee_cmd.append(f"mask={mask}")
        hummbee_str = self._containerize_cmd(hummbee_cmd, 'cpu')

        dale_model_str = self._containerize_cmd(
            self.config._build_dale_cmd_for_base(base_name, "model"), 'cpu')

        tag_cleanup = f"""echo "Cleaning normalized tag from peeling model..."
if [ -f {base_name}.model/table.info ]; then
    sed -i 's/SubType.*=.*//g' {base_name}.model/table.info
else
    echo "WARNING: {base_name}.model/table.info not found -- tag cleanup skipped"
fi
"""

        script_content = f"""{header}{env_setup}echo "Starting {job_name} at $(date)"

echo "Normalizing PSF..."
{dale_psf_str}

echo "Normalizing residual..."
{dale_residual_str}

echo "Running deconvolution..."
{hummbee_str}

echo "Normalizing model (creates .divmodel)..."
{dale_model_str}

{tag_cleanup}
echo "Finished {job_name} at $(date)"
"""

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        return script_path

    def generate_predict_job(self) -> Path:
        job_name = "libra_peel_predict"
        log_file = self.log_dir / f"{job_name}_%j.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        header = self._generate_sbatch_header(job_name, 'gpu', str(log_file))
        env_setup = self._generate_env_setup('gpu')
        cmd = self.config.build_roadrunner_predict_cmd()
        cmd_str = self._containerize_cmd(cmd, 'gpu')

        script_content = f"{header}{env_setup}echo \"Starting {job_name} at $(date)\"\n{cmd_str}\necho \"Finished {job_name} at $(date)\"\n"

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        return script_path

    def generate_uvsub_job(self) -> Path:
        job_name = "libra_peel_uvsub"
        log_file = self.log_dir / f"{job_name}_%j.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        header = self._generate_sbatch_header(job_name, 'cpu', str(log_file))
        env_setup = self._generate_env_setup('cpu')
        cmd = self.config.build_uvsub_cmd()
        cmd_str = self._containerize_cmd(cmd, 'cpu')

        script_content = f"{header}{env_setup}echo \"Starting {job_name} at $(date)\"\n{cmd_str}\necho \"Finished {job_name} at $(date)\"\n"

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        return script_path

    def generate_restore_job(self) -> Path:
        job_name = "libra_restore"
        log_file = self.log_dir / f"{job_name}_%j.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        header = self._generate_sbatch_header(job_name, 'cpu', str(log_file))
        env_setup = self._generate_env_setup('cpu')

        hummbee_cmd = self.config.build_hummbee_cmd(mode='restore')
        hummbee_str = self._containerize_cmd(hummbee_cmd, 'cpu')

        script_content = f"""{header}{env_setup}echo "Starting {job_name} at $(date)"

echo "Running restore..."
{hummbee_str}

echo "Finished {job_name} at $(date)"
"""

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)

        return script_path

    def _generate_cancel_script(self, job_ids_file: Path):
        """Generate a cancel script for the pipeline"""
        cancel_script_path = self.work_dir / "cancel_pipeline.sh"
        cancel_lines = [
            "#!/bin/bash --login",
            "",
            f"JOBIDS_FILE={job_ids_file}",
            "",
            "if [ ! -f \"$JOBIDS_FILE\" ]; then",
            "    echo \"No job IDs file found. Pipeline may not have been submitted yet.\"",
            "    exit 1",
            "fi",
            "",
            "echo \"Cancelling all pipeline jobs...\"",
            "while IFS= read -r jobid; do",
            "    if [ ! -z \"$jobid\" ]; then",
            "        echo \"Cancelling job $jobid\"",
            "        scancel $jobid",
            "    fi",
            "done < \"$JOBIDS_FILE\"",
            "",
            "echo \"All pipeline jobs cancelled\"",
            "rm -f \"$JOBIDS_FILE\"",
        ]

        with open(cancel_script_path, 'w') as f:
            f.write("\n".join(cancel_lines))
        cancel_script_path.chmod(0o755)

    def generate_pbmask(self) -> Path:
        """Generate a CRTF ellipse mask from a Gaussian primary beam estimate.

        FWHM (arcmin) = fwhm_coeff / freq_GHz
        Mask radius    = FWHM * sqrt(-ln(pb_level) / (4 * ln(2)))
        """
        pbmask_cfg = self.config.config['pbmask']
        rr = self.config.config['roadrunner']

        freq_ghz = float(rr['reffreq']) / 1e9
        fwhm_arcmin = pbmask_cfg['fwhm_coeff'] / freq_ghz
        pb_level = pbmask_cfg['pb_level']
        radius_arcmin = fwhm_arcmin * math.sqrt(-math.log(pb_level) / (4 * math.log(2)))

        # Phasecenter format is always "J2000 <RA> <Dec>"
        _, ra, dec = rr['phasecenter'].split()

        crtf_content = (
            "#CRTFv0 CASA Region Text Format version 0\n"
            f"ellipse[[{ra}, {dec}], [{radius_arcmin:.4f}arcmin, {radius_arcmin:.4f}arcmin], 0.0deg]\n"
        )

        mask_path = self.work_dir / "pbmask.crtf"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        mask_path.write_text(crtf_content)

        return mask_path

    def _build_imaging_loop(self, submit_lines: list, initial_dependency: str | None) -> str:
        """Build the major-cycle imaging loop. Returns the last CPU job ID variable name."""
        submit_lines.append("# Imaging iterations")
        for iteration in range(self.config.get_n_iterations()):
            gpu_job_ids = []

            modes = ["residual", "psf", "weight"] if iteration == 0 else ["residual"]

            for mode in modes:
                script = self.generate_gpu_job(iteration, mode)
                job_var = f"iter{iteration}_{mode}_id"

                dep_flag = ""
                if iteration == 0 and initial_dependency:
                    dep_flag = f"--dependency=afterok:{initial_dependency}"
                elif iteration > 0:
                    dep_flag = f"--dependency=afterok:$iter{iteration-1}_deconv_id"

                submit_lines.append(f"{job_var}=$(sbatch --parsable {dep_flag} {script})")
                submit_lines.append(f'if [ -z "${job_var}" ]; then echo "ERROR: {job_var} submission failed"; exit 1; fi')
                submit_lines.append(f"echo ${job_var} >> $JOBIDS_FILE")
                gpu_job_ids.append(f"${job_var}")

            cpu_script = self.generate_cpu_job(iteration)
            cpu_job_var = f"iter{iteration}_deconv_id"
            cpu_dep_str = ":".join(gpu_job_ids)
            submit_lines.append(f"{cpu_job_var}=$(sbatch --parsable --dependency=afterok:{cpu_dep_str} {cpu_script})")
            submit_lines.append(f'if [ -z "${cpu_job_var}" ]; then echo "ERROR: {cpu_job_var} submission failed"; exit 1; fi')
            submit_lines.append(f"echo ${cpu_job_var} >> $JOBIDS_FILE")
            submit_lines.append("")

        n_iter = self.config.get_n_iterations()
        return f"iter{n_iter-1}_deconv_id"

    def _build_peeling_loop(self, submit_lines: list, initial_dependency: str | None) -> str:
        """Build the peeling loop. Returns the last job ID variable name.

        Sequence:
          1. GPU: RoadRunner residual/psf/weight on peeling image (w-only)
          2. CPU: Dale + Hummbee (with CRTF mask) + Dale model → .divmodel
          3. GPU: RoadRunner predict (divmodel → MODEL_DATA)
          4. CPU: UVSub (DATA - MODEL_DATA → CORRECTED_DATA)
          5. Imaging loop on corrected data column
        """
        submit_lines.append("# Peeling phase")

        # --- GPU: peeling image (residual, psf, weight) ---
        peel_gpu_ids = []
        for mode in ["residual", "psf", "weight"]:
            script = self.generate_peeling_gpu_job(mode)
            job_var = f"peel_{mode}_id"
            dep_flag = f"--dependency=afterok:{initial_dependency}" if initial_dependency else ""
            submit_lines.append(f"{job_var}=$(sbatch --parsable {dep_flag} {script})")
            submit_lines.append(f'if [ -z "${job_var}" ]; then echo "ERROR: {job_var} submission failed"; exit 1; fi')
            submit_lines.append(f"echo ${job_var} >> $JOBIDS_FILE")
            peel_gpu_ids.append(f"${job_var}")

        # --- CPU: deconvolve peeling image ---
        peel_deconv_script = self.generate_peeling_cpu_job()
        peel_gpu_dep = ":".join(peel_gpu_ids)
        submit_lines.append(f"peel_deconv_id=$(sbatch --parsable --dependency=afterok:{peel_gpu_dep} {peel_deconv_script})")
        submit_lines.append('if [ -z "$peel_deconv_id" ]; then echo "ERROR: peel_deconv_id submission failed"; exit 1; fi')
        submit_lines.append("echo $peel_deconv_id >> $JOBIDS_FILE")
        submit_lines.append("")

        # --- GPU: predict divmodel into MODEL_DATA ---
        predict_script = self.generate_predict_job()
        submit_lines.append(f"peel_predict_id=$(sbatch --parsable --dependency=afterok:$peel_deconv_id {predict_script})")
        submit_lines.append('if [ -z "$peel_predict_id" ]; then echo "ERROR: peel_predict_id submission failed"; exit 1; fi')
        submit_lines.append("echo $peel_predict_id >> $JOBIDS_FILE")
        submit_lines.append("")

        # --- CPU: UVSub DATA - MODEL_DATA → CORRECTED_DATA ---
        uvsub_script = self.generate_uvsub_job()
        submit_lines.append(f"peel_uvsub_id=$(sbatch --parsable --dependency=afterok:$peel_predict_id {uvsub_script})")
        submit_lines.append('if [ -z "$peel_uvsub_id" ]; then echo "ERROR: peel_uvsub_id submission failed"; exit 1; fi')
        submit_lines.append("echo $peel_uvsub_id >> $JOBIDS_FILE")
        submit_lines.append("")

        # --- Imaging loop on corrected data column ---
        submit_lines.append("# Imaging phase (corrected data after peeling subtraction)")
        self.config.config['roadrunner']['datacolumn'] = 'corrected'
        last_job_var = self._build_imaging_loop(submit_lines, "$peel_uvsub_id")
        return last_job_var

    def generate_full_pipeline(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

        submit_script_path = self.work_dir / "submit_pipeline.sh"
        job_ids_file = self.work_dir / ".pipeline_job_ids"
        submit_lines = ["#!/bin/bash --login", "", "set -e", "set -o pipefail", ""]
        submit_lines.append(f"# Track all submitted job IDs for cancellation")
        submit_lines.append(f"JOBIDS_FILE={job_ids_file}")
        submit_lines.append(f"echo '' > $JOBIDS_FILE")
        submit_lines.append("")

        stage = self.config.get_pipeline_stage()
        initial_dependency = None

        # CF generation/filling stage
        if self.config.is_coyote_enabled() and stage in ['cf_only', 'full']:
            submit_lines.append("# Coyote CF generation and filling")
            cfgen_script = self.generate_coyote_cfgen_job()
            submit_lines.append(f"cfgen_id=$(sbatch --parsable {cfgen_script})")
            submit_lines.append('if [ -z "$cfgen_id" ]; then echo "ERROR: CF generation job submission failed"; exit 1; fi')
            submit_lines.append("echo \"Submitted CF generation job: $cfgen_id\"")
            submit_lines.append("echo $cfgen_id >> $JOBIDS_FILE")

            fillcf_script = self.generate_coyote_fillcf_job()
            submit_lines.append(f"fillcf_id=$(sbatch --parsable --dependency=afterok:$cfgen_id {fillcf_script})")
            submit_lines.append('if [ -z "$fillcf_id" ]; then echo "ERROR: CF filling job submission failed"; exit 1; fi')
            submit_lines.append("echo \"Submitted CF filling array job: $fillcf_id\"")
            submit_lines.append("echo $fillcf_id >> $JOBIDS_FILE")
            submit_lines.append("")

            initial_dependency = "$fillcf_id"

        # Stop here if cf_only
        if stage == 'cf_only':
            submit_lines.append("echo \"CF generation/filling pipeline submitted (cf_only mode)\"")
            with open(submit_script_path, 'w') as f:
                f.write("\n".join(submit_lines))
            submit_script_path.chmod(0o755)
            self._generate_cancel_script(job_ids_file)
            print(f"Generated slurm scripts in: {self.scripts_dir}")
            print(f"Log files will be in: {self.log_dir}")
            print(f"To submit pipeline: {submit_script_path}")
            print(f"To cancel pipeline: {self.work_dir}/cancel_pipeline.sh")
            print(f"Pipeline stage: {stage}")
            return

        # PB mask generation (analytical, written at pipeline-creation time)
        if self.config.is_pbmask_enabled():
            mask_path = self.generate_pbmask()
            self.config.config['hummbee']['mask'].append(str(mask_path))
            print(f"Generated PB mask: {mask_path}")

        # Dispatch to pipeline-type-specific loop builder
        if stage in ['imaging_only', 'full']:
            pipeline_type = self.config.config['pipeline'].get('pipeline_type', 'imaging')
            if pipeline_type == 'imaging':
                last_job_var = self._build_imaging_loop(submit_lines, initial_dependency)
            elif pipeline_type == 'peeling':
                last_job_var = self._build_peeling_loop(submit_lines, initial_dependency)
            else:
                raise ValueError(f"Unknown pipeline_type: '{pipeline_type}'. Valid values: imaging, peeling")

            restore_script = self.generate_restore_job()
            submit_lines.append(f"restore_id=$(sbatch --parsable --dependency=afterok:${last_job_var} {restore_script})")
            submit_lines.append('if [ -z "$restore_id" ]; then echo "ERROR: restore job submission failed"; exit 1; fi')
            submit_lines.append("echo $restore_id >> $JOBIDS_FILE")
            submit_lines.append("")

        submit_lines.append("echo \"Pipeline submitted successfully\"")

        with open(submit_script_path, 'w') as f:
            f.write("\n".join(submit_lines))
        submit_script_path.chmod(0o755)

        self._generate_cancel_script(job_ids_file)

        print(f"Generated slurm scripts in: {self.scripts_dir}")
        print(f"Log files will be in: {self.log_dir}")
        print(f"To submit pipeline: {submit_script_path}")
        print(f"To cancel pipeline: {self.work_dir}/cancel_pipeline.sh")
        print(f"Pipeline stage: {stage}")
        if self.config.is_coyote_enabled():
            print(f"Coyote CF generation and filling enabled")


def print_config_summary(config: ImagingConfig):
    """Print concise summary of what will be generated"""
    stage = config.get_pipeline_stage()
    slurm = config.config['slurm']

    print("Pipeline Configuration:")
    print(f"  Stage: {stage}")
    print(f"  Iterations: {config.get_n_iterations()}")
    print(f"  Data: {config.get_vis()}")
    print(f"  Output: {config.get_imagename_base()}")

    print("\nResources:")
    print(f"  GPU: {slurm['gpu']['partition']}, {slurm['gpu']['gpu_arch']}, {slurm['gpu']['gpus_per_node']} GPUs")
    print(f"  CPU: {slurm['cpu']['partition']}")

    if config.is_coyote_enabled():
        coy = config.config['coyote']
        print(f"\nCoyote CF:")
        print(f"  Cache: {coy['cfcache']}")
        print(f"  Parallel tasks: {coy['fillcf']['nprocs']}")

    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_pipeline.py <config_file>")
        sys.exit(1)

    config = ImagingConfig(sys.argv[1])
    print_config_summary(config)
    generator = SlurmJobGenerator(config)
    generator.generate_full_pipeline()
