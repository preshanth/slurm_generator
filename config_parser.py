#!/usr/bin/env python3

import yaml
import os
from typing import Dict, Any, List
from pathlib import Path

class ImagingConfig:
    def __init__(self, config_file: str):
        self.config_file = Path(config_file)
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self._validate_config()

    def _validate_config(self):
        required_sections = ['pipeline', 'slurm', 'data', 'roadrunner', 'dale', 'hummbee', 'environment']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")

        validate_paths = self.config['pipeline'].get('validate_paths', True)

        if validate_paths:
            env = self.config['environment']
            using_container = bool(env.get('container_image'))

            # If using container, validate the container file exists
            if using_container:
                container_path = Path(env['container_image'])
                if not container_path.exists():
                    raise ValueError(f"Container image not found: {env['container_image']}")
                if not container_path.is_file():
                    raise ValueError(f"Container image is not a file: {env['container_image']}")

            # Always require bin_dir (path inside container or on host)
            if not env['bin_dir'] and not (env.get('bin_dir_gpu') and env.get('bin_dir_cpu')):
                raise ValueError("environment.bin_dir or both bin_dir_gpu/bin_dir_cpu must be specified")

            # lib_dir and casapath only required for non-container mode
            if not using_container:
                if not env['lib_dir'] and not (env.get('lib_dir_gpu') and env.get('lib_dir_cpu')):
                    raise ValueError("environment.lib_dir or both lib_dir_gpu/lib_dir_cpu must be specified")

                if not env['casapath']:
                    raise ValueError("environment.casapath must be specified")

            if not self.config['data']['vis']:
                raise ValueError("data.vis must be specified")

            if not self.config['roadrunner']['cfcache']:
                raise ValueError("roadrunner.cfcache must be specified")

            if not self.config['roadrunner']['phasecenter']:
                raise ValueError("roadrunner.phasecenter must be specified")

    def should_validate_paths(self) -> bool:
        return self.config['pipeline'].get('validate_paths', True)

    def get_pipeline_stage(self) -> str:
        stage = self.config['pipeline'].get('stage', 'full')
        if stage not in ['cf_only', 'imaging_only', 'full']:
            raise ValueError(f"Invalid pipeline stage: {stage}. Must be 'cf_only', 'imaging_only', or 'full'")

        # Validate stage compatibility
        if stage == 'cf_only' and not self.is_coyote_enabled():
            raise ValueError("stage='cf_only' requires coyote.enabled=true")

        return stage

    def get_n_iterations(self) -> int:
        return self.config['pipeline']['n_iterations']

    def get_output_dir(self) -> Path:
        return Path(self.config['pipeline']['output_dir'])

    def get_work_dir(self) -> Path:
        return Path(self.config['pipeline']['work_dir'])

    def get_vis(self) -> str:
        return self.config['data']['vis']

    def get_imagename_base(self) -> str:
        return self.config['data']['imagename_base']

    def get_imagename(self, iteration: int, suffix: str = "") -> str:
        base = self.get_imagename_base()
        if suffix:
            return f"{base}_iter{iteration}_{suffix}"
        return f"{base}_iter{iteration}"

    def get_modelimagename(self, iteration: int) -> str:
        if iteration == 0:
            return ""
        # Use .divmodel (normalized model) from previous iteration
        return f"{self.get_imagename_base()}_iter{iteration-1}.divmodel"

    def build_roadrunner_cmd(self, iteration: int, mode: str) -> List[str]:
        rr = self.config['roadrunner']
        imagename_base = self.get_imagename(iteration)
        # Roadrunner needs full imagename with extension for each mode
        imagename = f"{imagename_base}.{mode}"
        modelimagename = self.get_modelimagename(iteration)

        cmd = [
            "roadrunner",
            "help=noprompt",
            f"vis={self.get_vis()}",
            f"imagename={imagename}",
            f"modelimagename={modelimagename}",
            f"datacolumn={rr['datacolumn']}",
            f"sowimageext={rr['sowimageext']}",
            f"complexgrid={rr['complexgrid']}",
            f"imsize={rr['imsize']}",
            f"cell={rr['cell']}",
            f"stokes={rr['stokes']}",
            f"reffreq={rr['reffreq']}",
            f"phasecenter={rr['phasecenter']}",
            f"weighting={rr['weighting']}",
            f"rmode={rr['rmode']}",
            f"robust={rr['robust']}",
            f"wprojplanes={rr['wprojplanes']}",
            f"gridder={rr['gridder']}",
            f"cfcache={rr['cfcache']}",
            f"mode={mode}",
            f"wbawp={1 if rr['wbawp'] else 0}",
            f"field={rr['field']}",
            f"spw={rr['spw']}",
            f"uvrange={rr['uvrange']}",
            f"pbcor={1 if rr['pbcor'] else 0}",
            f"conjbeams={1 if rr['conjbeams'] else 0}",
            f"pblimit={rr['pblimit']}",
            f"usepointing={1 if rr['usepointing'] else 0}",
            f"pointingoffsetsigdev={','.join(map(str, rr['pointingoffsetsigdev']))}"
        ]
        return cmd

    def build_dale_cmd(self, iteration: int, imtype: str) -> List[str]:
        dale = self.config['dale']
        imagename = self.get_imagename(iteration)

        # Always use iter0 weight for normalization
        weightimage = f"{self.get_imagename_base()}_iter0.weight"

        cmd = [
            "dale",
            "help=noprompt",
            f"imagename={imagename}",
            f"imtype={imtype}",
            f"pblimit={dale['pblimit']}",
            f"computepb={1 if dale['computepb'] else 0}"
        ]

        if imtype in ['residual', 'model']:
            cmd.append(f"weightimage={weightimage}")
            cmd.append(f"sowimage={self.get_imagename(0)}.sumwt")

        return cmd

    def build_hummbee_cmd(self, iteration: int) -> List[str]:
        hb = self.config['hummbee']
        imagename = self.get_imagename(iteration)
        modelimagename = f"{imagename}.model"

        cmd = [
            "hummbee",
            "help=noprompt",
            f"imagename={imagename}",
            f"modelimagename={modelimagename}",
            f"deconvolver={hb['deconvolver']}",
            f"nterms={hb['nterms']}",
            f"gain={hb['gain']}",
            f"nsigma={hb['nsigma']}",
            f"threshold={hb['threshold']}",
            f"cycleniter={hb['cycleniter']}",
            f"cyclefactor={hb['cyclefactor']}",
            f"specmode={hb['specmode']}",
            f"pbcor={1 if hb['pbcor'] else 0}",
            f"mode={hb['mode']}"
        ]

        if hb['scales']:
            cmd.append(f"scales={','.join(map(str, hb['scales']))}")

        if hb['largestscale'] > 0:
            cmd.append(f"largestscale={hb['largestscale']}")

        if hb['fusedthreshold'] > 0:
            cmd.append(f"fusedthreshold={hb['fusedthreshold']}")

        if hb['mask']:
            cmd.append(f"mask={','.join(hb['mask'])}")

        return cmd

    def get_slurm_config(self) -> Dict[str, Any]:
        return self.config['slurm'].copy()

    def is_coyote_enabled(self) -> bool:
        return self.config.get('coyote', {}).get('enabled', False)

    def build_coyote_cmd(self, mode: str) -> List[str]:
        coy = self.config['coyote']
        rr = self.config['roadrunner']

        imsize = coy['imsize'] if coy['imsize'] > 0 else rr['imsize']
        cell = coy['cell'] if coy['cell'] > 0 else rr['cell']
        reffreq = coy['reffreq'] if coy['reffreq'] else rr['reffreq']
        phasecenter = coy['phasecenter'] if coy['phasecenter'] else rr['phasecenter']
        field = coy['field'] if coy['field'] else rr['field']
        spw = coy['spw'] if coy['spw'] else rr['spw']

        cmd = [
            "coyote",
            "help=noprompt",
            f"vis={self.get_vis()}",
            f"telescope={coy['telescope']}",
            f"imsize={imsize}",
            f"cell={cell}",
            f"stokes={coy['stokes']}",
            f"reffreq={reffreq}",
            f"phasecenter={phasecenter}",
            f"wplanes={coy['wplanes']}",
            f"cfcache={coy['cfcache']}",
            f"wbawp={1 if coy['wbawp'] else 0}",
            f"aterm={1 if coy['aterm'] else 0}",
            f"psterm={1 if coy['psterm'] else 0}",
            f"conjbeams={1 if coy['conjbeams'] else 0}",
            f"muellertype={coy['muellertype']}",
            f"dpa={coy['dpa']}",
            f"field={field}",
            f"spw={spw}",
            f"buffersize={coy['buffersize']}",
            f"oversampling={coy['oversampling']}",
            f"mode={mode}",
            "cflist="
        ]
        return cmd


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python config_parser.py <config_file>")
        sys.exit(1)

    config = ImagingConfig(sys.argv[1])

    print("=== Configuration Validation Successful ===\n")
    print(f"Pipeline iterations: {config.get_n_iterations()}")
    print(f"Output directory: {config.get_output_dir()}")
    print(f"Measurement set: {config.get_vis()}")
    print(f"\nExample commands for iteration 0:")
    print("\nRoadRunner (residual):")
    print(" ".join(config.build_roadrunner_cmd(0, "residual", "residual")))
    print("\nDale (PSF normalization):")
    print(" ".join(config.build_dale_cmd(0, "psf")))
    print("\nHummbee (deconvolution):")
    print(" ".join(config.build_hummbee_cmd(0)))
