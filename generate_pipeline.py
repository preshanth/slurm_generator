#!/usr/bin/env python3

import sys
import os
from pathlib import Path
from config_parser import ImagingConfig

class SlurmJobGenerator:
    def __init__(self, config: ImagingConfig):
        self.config = config
        self.work_dir = config.get_work_dir()
        self.log_dir = self.work_dir / "logs"
        self.scripts_dir = self.work_dir / "slurm_scripts"

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
            "#!/bin/bash",
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

        lines.append("")
        return "\n".join(lines)

    def generate_coyote_cfgen_job(self) -> Path:
        job_name = "libra_cfgen"
        log_file = self.log_dir / f"{job_name}_%j.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        coy_cfg = self.config.config['coyote']['generate']
        slurm_cfg = self.config.config['slurm']

        lines = [
            "#!/bin/bash",
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

        lines.append("")
        header = "\n".join(lines)

        env_setup = self._generate_env_setup('cpu')
        bin_dir = self._get_bin_dir('cpu')

        cmd = self.config.build_coyote_cmd('dryrun')
        cmd[0] = str(bin_dir / cmd[0])
        cmd_str = " ".join(cmd)

        script_content = f"{header}{env_setup}echo \"Starting CF generation at $(date)\"\n{cmd_str}\necho \"Finished CF generation at $(date)\"\n"

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)

        return script_path

    def generate_coyote_fillcf_job(self, dependency: str = None) -> Path:
        job_name = "libra_fillcf"
        log_file = self.log_dir / f"{job_name}_%A_%a.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        coy_cfg = self.config.config['coyote']
        fillcf_cfg = coy_cfg['fillcf']
        slurm_cfg = self.config.config['slurm']
        nprocs = fillcf_cfg['nprocs']

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={fillcf_cfg['partition']}",
            f"#SBATCH --array=0-{nprocs-1}",
            f"#SBATCH --nodes=1",
            f"#SBATCH --ntasks-per-node=1",
            f"#SBATCH --time={fillcf_cfg['time']}",
            f"#SBATCH --mem={fillcf_cfg['mem']}",
            f"#SBATCH --output={log_file}",
        ]

        if dependency:
            lines.append(f"#SBATCH --dependency=afterok:{dependency}")

        if slurm_cfg['account']:
            lines.append(f"#SBATCH --account={slurm_cfg['account']}")
        if slurm_cfg['qos']:
            lines.append(f"#SBATCH --qos={slurm_cfg['qos']}")
        if slurm_cfg['email']:
            lines.append(f"#SBATCH --mail-user={slurm_cfg['email']}")
            lines.append(f"#SBATCH --mail-type={slurm_cfg['mail_type']}")

        lines.append("")
        header = "\n".join(lines)

        env_setup = self._generate_env_setup('cpu')

        worker_script = self.scripts_dir / "fillcf_worker.py"
        cfcache_dir = coy_cfg['cfcache']
        bin_dir = self._get_bin_dir('cpu')
        coyote_bin = str(bin_dir / "coyote")

        script_content = f"""{header}{env_setup}echo "Starting CF filling task $SLURM_ARRAY_TASK_ID at $(date)"
python3 {worker_script} --cfcache_dir {cfcache_dir} --nprocs {nprocs} --coyote_bin {coyote_bin}
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
        cmd_params = ' '.join([f'"{arg}"' for arg in cmd_base[2:]])

        worker_content = f'''#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import sys

def distribute_cfs(cfcache_dir, nprocs, task_id):
    cfs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(cfcache_dir, 'CF*.im')) if os.path.isdir(f)])

    if not cfs:
        print(f"No CFs found in {{cfcache_dir}}")
        sys.exit(1)

    base, remainder = divmod(len(cfs), nprocs)
    quantities = [base + 1 if i < remainder else base for i in range(nprocs)]

    start = sum(quantities[:task_id])
    end = start + quantities[task_id]

    return cfs[start:end]

def fill_cfs(cfs, cfcache_dir, coyote_bin):
    if not cfs:
        print("No CFs assigned to this task")
        return

    cflist = ",".join(cfs)
    cmd = [coyote_bin] + {cmd_params}.split() + [f"cflist={{cflist}}"]

    print(f"Processing {{len(cfs)}} CFs: {{', '.join(cfs[:3])}}{{' ...' if len(cfs) > 3 else ''}}")
    print(f"Command: {{' '.join(cmd)}}")

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CF filling worker')
    parser.add_argument('--cfcache_dir', required=True, help='CF cache directory')
    parser.add_argument('--nprocs', type=int, required=True, help='Number of processes')
    parser.add_argument('--coyote_bin', required=True, help='Path to coyote binary')
    args = parser.parse_args()

    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))

    cfs = distribute_cfs(args.cfcache_dir, args.nprocs, task_id)
    fill_cfs(cfs, args.cfcache_dir, args.coyote_bin)
'''

        worker_path.parent.mkdir(parents=True, exist_ok=True)
        with open(worker_path, 'w') as f:
            f.write(worker_content)
        worker_path.chmod(0o755)

    def generate_gpu_job(self, iteration: int, mode: str, dependencies: list = None) -> Path:
        job_name = f"libra_iter{iteration}_{mode}"
        log_file = self.log_dir / f"{job_name}_%j.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        header = self._generate_sbatch_header(job_name, 'gpu', str(log_file))

        if dependencies:
            dep_str = ":".join(dependencies)
            header += f"#SBATCH --dependency=afterok:{dep_str}\n\n"

        env_setup = self._generate_env_setup('gpu')
        bin_dir = self._get_bin_dir('gpu')

        cmd = self.config.build_roadrunner_cmd(iteration, mode, mode)
        cmd[0] = str(bin_dir / cmd[0])
        cmd_str = " ".join(cmd)

        script_content = f"{header}{env_setup}echo \"Starting {job_name} at $(date)\"\n{cmd_str}\necho \"Finished {job_name} at $(date)\"\n"

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)

        return script_path

    def generate_cpu_job(self, iteration: int, dependencies: list) -> Path:
        job_name = f"libra_iter{iteration}_deconv"
        log_file = self.log_dir / f"{job_name}_%j.log"
        script_path = self.scripts_dir / f"{job_name}.sh"

        header = self._generate_sbatch_header(job_name, 'cpu', str(log_file))

        dep_str = ":".join(dependencies)
        header += f"#SBATCH --dependency=afterok:{dep_str}\n\n"

        env_setup = self._generate_env_setup('cpu')
        bin_dir = self._get_bin_dir('cpu')

        imagename = self.config.get_imagename(iteration)

        dale_psf_cmd = self.config.build_dale_cmd(iteration, "psf")
        dale_psf_cmd[0] = str(bin_dir / dale_psf_cmd[0])

        dale_residual_cmd = self.config.build_dale_cmd(iteration, "residual")
        dale_residual_cmd[0] = str(bin_dir / dale_residual_cmd[0])

        hummbee_cmd = self.config.build_hummbee_cmd(iteration)
        hummbee_cmd[0] = str(bin_dir / hummbee_cmd[0])

        script_content = f"""{header}{env_setup}echo "Starting {job_name} at $(date)"

echo "Normalizing PSF..."
{' '.join(dale_psf_cmd)}

echo "Normalizing residual..."
{' '.join(dale_residual_cmd)}

echo "Running deconvolution..."
{' '.join(hummbee_cmd)}

echo "Finished {job_name} at $(date)"
"""

        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)

        return script_path

    def generate_full_pipeline(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

        submit_script_path = self.work_dir / "submit_pipeline.sh"
        submit_lines = ["#!/bin/bash", ""]

        initial_dependency = None

        if self.config.is_coyote_enabled():
            submit_lines.append("# Coyote CF generation and filling")
            cfgen_script = self.generate_coyote_cfgen_job()
            submit_lines.append(f"cfgen_id=$(sbatch --parsable {cfgen_script})")
            submit_lines.append("echo \"Submitted CF generation job: $cfgen_id\"")

            fillcf_script = self.generate_coyote_fillcf_job(dependency="$cfgen_id")
            submit_lines.append(f"fillcf_id=$(sbatch --parsable {fillcf_script})")
            submit_lines.append("echo \"Submitted CF filling array job: $fillcf_id\"")
            submit_lines.append("")

            initial_dependency = "$fillcf_id"

        submit_lines.append("# Imaging iterations")
        for iteration in range(self.config.get_n_iterations()):
            gpu_job_ids = []

            for mode in ["residual", "psf", "weight"]:
                deps = None
                if iteration == 0 and initial_dependency:
                    deps = [initial_dependency]
                elif iteration > 0:
                    deps = [f"$iter{iteration-1}_deconv_id"]

                script = self.generate_gpu_job(iteration, mode, deps)
                job_var = f"iter{iteration}_{mode}_id"
                submit_lines.append(f"{job_var}=$(sbatch --parsable {script})")
                gpu_job_ids.append(f"${job_var}")

            cpu_script = self.generate_cpu_job(iteration, gpu_job_ids)
            cpu_job_var = f"iter{iteration}_deconv_id"
            submit_lines.append(f"{cpu_job_var}=$(sbatch --parsable {cpu_script})")
            submit_lines.append("")

        submit_lines.append("echo \"Pipeline submitted successfully\"")

        with open(submit_script_path, 'w') as f:
            f.write("\n".join(submit_lines))
        submit_script_path.chmod(0o755)

        print(f"Generated slurm scripts in: {self.scripts_dir}")
        print(f"Log files will be in: {self.log_dir}")
        print(f"To submit pipeline: {submit_script_path}")
        if self.config.is_coyote_enabled():
            print(f"Coyote CF generation and filling enabled")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_pipeline.py <config_file>")
        sys.exit(1)

    config = ImagingConfig(sys.argv[1])
    generator = SlurmJobGenerator(config)
    generator.generate_full_pipeline()
