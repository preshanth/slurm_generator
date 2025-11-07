#!/usr/bin/env python3

import sys
from pathlib import Path
from config_parser import ImagingConfig

def check_binary(bin_path: Path) -> bool:
    return bin_path.exists() and bin_path.is_file()

def verify_environment(config: ImagingConfig) -> bool:
    env = config.config['environment']

    bin_dir = Path(env['bin_dir'])
    lib_dir = Path(env['lib_dir'])
    casapath = Path(env['casapath'])

    required_bins = ['roadrunner', 'dale', 'hummbee']
    if config.is_coyote_enabled():
        required_bins.append('coyote')

    all_good = True

    print("Checking directories...")
    if not bin_dir.exists():
        print(f"❌ Binary directory not found: {bin_dir}")
        all_good = False
    else:
        print(f"✓ Binary directory: {bin_dir}")

    if not lib_dir.exists():
        print(f"❌ Library directory not found: {lib_dir}")
        all_good = False
    else:
        print(f"✓ Library directory: {lib_dir}")

    if not casapath.exists():
        print(f"❌ CASAPATH not found: {casapath}")
        all_good = False
    else:
        print(f"✓ CASAPATH: {casapath}")

    print("\nChecking binaries...")
    for binary in required_bins:
        bin_path = bin_dir / binary
        if check_binary(bin_path):
            print(f"✓ {binary}")
        else:
            print(f"❌ {binary} not found at {bin_path}")
            all_good = False

    return all_good

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_binaries.py <config_file>")
        sys.exit(1)

    config = ImagingConfig(sys.argv[1])

    if not config.should_validate_paths():
        print("⚠️  Path validation is disabled in config (validate_paths: false)")
        print("Skipping binary verification - script generation only mode")
        sys.exit(0)

    if verify_environment(config):
        print("\n✓ All checks passed")
        sys.exit(0)
    else:
        print("\n❌ Verification failed")
        sys.exit(1)
