import sys
import textwrap
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_parser import ImagingConfig
from generate_pipeline import SlurmJobGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "pipeline": {
        "n_iterations": 2,
        "output_dir": "./output",
        "work_dir": "",        # filled in per test via tmp_path
        "validate_paths": False,
        "stage": "imaging_only",
    },
    "environment": {
        "container_image": "",
        "container_binds": [],
        "bin_dir": "/opt/libra/bin",
        "bin_dir_gpu": "",
        "bin_dir_cpu": "",
        "lib_dir": "/opt/libra/lib",
        "lib_dir_gpu": "",
        "lib_dir_cpu": "",
        "casapath": "/opt/casa/data",
    },
    "slurm": {
        "gpu": {
            "partition": "general-short",
            "nodes": 1,
            "ntasks_per_node": 1,
            "gpus_per_node": 1,
            "gpu_arch": "v100",
            "time": "02:00:00",
            "mem": "64G",
        },
        "cpu": {
            "partition": "general-short",
            "nodes": 1,
            "ntasks_per_node": 1,
            "time": "01:00:00",
            "mem": "32G",
        },
        "account": "",
        "qos": "",
        "email": "",
        "mail_type": "END,FAIL",
    },
    "data": {
        "vis": "/data/test.ms",
        "imagename_base": "test_image",
    },
    "roadrunner": {
        "datacolumn": "data",
        "field": "",
        "spw": "*",
        "uvrange": "",
        "imsize": 1024,
        "cell": 1.0,
        "stokes": "I",
        "phasecenter": "J2000 19h59m28.5s +40d44m01.5s",
        "reffreq": "3.0e9",
        "weighting": "briggs",
        "rmode": "norm",
        "robust": 0.0,
        "gridder": "awphpg",
        "wprojplanes": 1,
        "cfcache": "/data/cfcache",
        "wbawp": True,
        "sowimageext": "",
        "complexgrid": "",
        "pbcor": True,
        "conjbeams": True,
        "pblimit": 0.001,
        "usepointing": False,
        "pointingoffsetsigdev": [300, 300],
    },
    "dale": {
        "pblimit": 0.2,
        "computepb": False,
    },
    "hummbee": {
        "deconvolver": "asp",
        "nterms": 1,
        "gain": 0.1,
        "threshold": 0.0,
        "nsigma": 0.0,
        "cycleniter": 1000,
        "cyclefactor": 1.0,
        "scales": [],
        "largestscale": -1,
        "fusedthreshold": 0,
        "mask": [],
        "specmode": "mfs",
        "pbcor": False,
        "mode": "deconvolve",
    },
}


def make_generator(tmp_path, overrides=None):
    """Write a config YAML to tmp_path and return a SlurmJobGenerator."""
    cfg = BASE_CONFIG.copy()
    cfg = yaml.safe_load(yaml.dump(cfg))   # deep copy via round-trip
    cfg["pipeline"]["work_dir"] = str(tmp_path)

    if overrides:
        for key_path, value in overrides.items():
            parts = key_path.split(".")
            target = cfg
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg))

    config = ImagingConfig(str(config_path))
    return SlurmJobGenerator(config)


def read_script(path: Path) -> str:
    return path.read_text()


# ---------------------------------------------------------------------------
# CPU job tests
# ---------------------------------------------------------------------------

def test_cpu_iter0_has_psf_normalization(tmp_path):
    gen = make_generator(tmp_path)
    script = read_script(gen.generate_cpu_job(0))
    assert "Normalizing PSF" in script
    assert "imtype=psf" in script


def test_cpu_iter0_no_tag_cleanup(tmp_path):
    gen = make_generator(tmp_path)
    script = read_script(gen.generate_cpu_job(0))
    assert "SubType" not in script
    assert "Cleaning normalized tag" not in script


def test_cpu_iter1_has_tag_cleanup_with_correct_imagename(tmp_path):
    gen = make_generator(tmp_path)
    script = read_script(gen.generate_cpu_job(1))
    assert "test_image_iter1.model/table.info" in script
    assert "Cleaning normalized tag" in script


def test_cpu_iter1_copies_prev_model_before_hummbee(tmp_path):
    gen = make_generator(tmp_path)
    script = read_script(gen.generate_cpu_job(1))
    assert "cp -r test_image_iter0.model test_image_iter1.model" in script
    # Copy must appear before hummbee invocation
    copy_pos = script.index("cp -r test_image_iter0.model")
    hummbee_pos = script.index("hummbee")
    assert copy_pos < hummbee_pos

def test_cpu_iter0_no_model_copy(tmp_path):
    gen = make_generator(tmp_path)
    script = read_script(gen.generate_cpu_job(0))
    assert "cp -r" not in script or "iter" not in script.split("cp -r")[1].split("\n")[0]

def test_cpu_iter1_no_psf_normalization(tmp_path):
    gen = make_generator(tmp_path)
    script = read_script(gen.generate_cpu_job(1))
    assert "Normalizing PSF" not in script
    assert "imtype=psf" not in script


def test_no_undefined_shell_vars_in_cpu_scripts(tmp_path):
    gen = make_generator(tmp_path)
    for iteration in range(2):
        script = read_script(gen.generate_cpu_job(iteration))
        assert "$iter" not in script
        assert "$imagename" not in script


# ---------------------------------------------------------------------------
# GPU job tests
# ---------------------------------------------------------------------------

def test_gpu_iter0_generates_three_modes(tmp_path):
    gen = make_generator(tmp_path)
    gen.generate_full_pipeline()
    scripts_dir = tmp_path / "slurm_scripts"
    assert (scripts_dir / "libra_iter0_residual.sh").exists()
    assert (scripts_dir / "libra_iter0_psf.sh").exists()
    assert (scripts_dir / "libra_iter0_weight.sh").exists()


def test_gpu_iter1_generates_only_residual(tmp_path):
    gen = make_generator(tmp_path)
    gen.generate_full_pipeline()
    scripts_dir = tmp_path / "slurm_scripts"
    assert (scripts_dir / "libra_iter1_residual.sh").exists()
    assert not (scripts_dir / "libra_iter1_psf.sh").exists()
    assert not (scripts_dir / "libra_iter1_weight.sh").exists()


def test_gpu_iter1_has_existence_checks_before_copy(tmp_path):
    gen = make_generator(tmp_path)
    script = read_script(gen.generate_gpu_job(1, "residual"))
    assert "if [ ! -d" in script
    assert "exit 1" in script
    assert "cp -r" in script
    # Existence check must appear before the copy
    check_pos = script.index("if [ ! -d")
    copy_pos = script.index("cp -r")
    assert check_pos < copy_pos


# ---------------------------------------------------------------------------
# Dependency chain test
# ---------------------------------------------------------------------------

def test_submit_script_dependency_chain(tmp_path):
    gen = make_generator(tmp_path)
    gen.generate_full_pipeline()
    submit = (tmp_path / "submit_pipeline.sh").read_text()

    # iter0 CPU depends on iter0 GPU jobs
    assert "dependency=afterok:$iter0_residual_id:$iter0_psf_id:$iter0_weight_id" in submit

    # iter1 GPU depends on iter0 CPU
    assert "dependency=afterok:$iter0_deconv_id" in submit

    # iter1 CPU depends on iter1 GPU
    assert "dependency=afterok:$iter1_residual_id" in submit


# ---------------------------------------------------------------------------
# Container binds test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PB mask tests
# ---------------------------------------------------------------------------

def test_pbmask_crtf_radius_lband(tmp_path):
    # VLA L-band: 1.5 GHz, fwhm_coeff=45.0 → FWHM=30 arcmin, p=0.2 → r≈22.86 arcmin
    gen = make_generator(tmp_path, overrides={
        "roadrunner.reffreq": "1.5e9",
        "pbmask": {"enabled": True, "pb_level": 0.2, "fwhm_coeff": 45.0},
    })
    mask_path = gen.generate_pbmask()
    content = mask_path.read_text()
    assert "CRTFv0" in content
    assert "ellipse" in content
    # Radius should be ~22.86 arcmin — check it's in the right ballpark
    import re
    match = re.search(r"([\d.]+)arcmin", content)
    assert match, "No arcmin radius found in CRTF"
    radius = float(match.group(1))
    assert 22.0 < radius < 24.0


def test_pbmask_crtf_contains_phasecenter(tmp_path):
    gen = make_generator(tmp_path, overrides={
        "pbmask": {"enabled": True, "pb_level": 0.2, "fwhm_coeff": 45.0},
    })
    mask_path = gen.generate_pbmask()
    content = mask_path.read_text()
    assert "19h59m28.5s" in content
    assert "+40d44m01.5s" in content


def test_pbmask_disabled_not_in_hummbee_cmd(tmp_path):
    gen = make_generator(tmp_path)   # pbmask not in base config → disabled
    gen.generate_full_pipeline()
    script = read_script(tmp_path / "slurm_scripts" / "libra_iter0_deconv.sh")
    assert "pbmask.crtf" not in script


def test_pbmask_enabled_injected_into_hummbee_cmd(tmp_path):
    gen = make_generator(tmp_path, overrides={
        "pbmask": {"enabled": True, "pb_level": 0.2, "fwhm_coeff": 45.0},
    })
    gen.generate_full_pipeline()
    script = read_script(tmp_path / "slurm_scripts" / "libra_iter0_deconv.sh")
    assert "pbmask.crtf" in script


def test_container_binds_are_separate_flags(tmp_path):
    gen = make_generator(tmp_path, overrides={
        "environment.container_image": "/opt/containers/libra.sif",
        "environment.bin_dir": "/opt/libra/bin",
        "coyote": {
            "enabled": True,
            "cfcache": "/data/cfcache",
            "generate": {"partition": "general-short", "time": "00:30:00", "mem": "8G"},
            "fillcf": {"partition": "general-short", "nprocs": 4, "time": "01:00:00", "mem": "4G"},
            "telescope": "EVLA", "imsize": 0, "cell": 0.0, "stokes": "I",
            "reffreq": "", "phasecenter": "", "wplanes": 1, "wbawp": True,
            "aterm": True, "psterm": False, "conjbeams": True,
            "muellertype": "diagonal", "dpa": 360, "field": "", "spw": "*",
            "buffersize": 0, "oversampling": 20,
        },
        "pipeline.stage": "full",
        "environment.container_binds": ["/mnt/scratch", "/mnt/home"],
    })
    script = read_script(gen.generate_coyote_fillcf_job())
    # Each path must appear as its own --container_binds flag, not joined
    assert "--container_binds /mnt/scratch" in script
    assert "--container_binds /mnt/home" in script
    assert "--container_binds '/mnt/scratch /mnt/home'" not in script
