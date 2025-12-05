#!/usr/bin/env python3

import sys
from pathlib import Path
from config_parser import ImagingConfig

def check_casarc(casapath: Path) -> tuple[bool, str]:
    """Check if ~/.casarc exists and has correct measures.directory"""
    casarc = Path.home() / ".casarc"

    if not casarc.exists():
        return False, f"~/.casarc not found. Create with:\n  echo 'measures.directory: {casapath}' > ~/.casarc"

    with open(casarc) as f:
        content = f.read()

    if 'measures.directory:' not in content:
        return False, f"~/.casarc missing 'measures.directory:'. Add:\n  measures.directory: {casapath}"

    # Extract the configured path
    for line in content.split('\n'):
        if 'measures.directory:' in line:
            configured = line.split(':', 1)[1].strip()
            if Path(configured) != casapath:
                return False, f"~/.casarc has measures.directory: {configured}\n  Expected: {casapath}"

    return True, str(casarc)

def verify_environment(config: ImagingConfig) -> bool:
    """Verify environment with clear reporting of what's being checked"""
    env = config.config['environment']

    bin_dir = Path(env['bin_dir'])
    lib_dir = Path(env['lib_dir'])
    casapath = Path(env['casapath'])

    required_bins = ['roadrunner', 'dale', 'hummbee']
    if config.is_coyote_enabled():
        required_bins.append('coyote')

    all_good = True

    print("Environment paths:")
    print(f"  bin_dir: {bin_dir}")
    print(f"  lib_dir: {lib_dir}")
    print(f"  casapath: {casapath}")

    print("\nDirectory checks:")
    for name, path in [("bin_dir", bin_dir), ("lib_dir", lib_dir), ("casapath", casapath)]:
        if not path.exists():
            print(f"  ❌ {name}: {path}")
            all_good = False
        else:
            print(f"  ✓ {name}: {path}")

    print("\nBinary checks:")
    for binary in required_bins:
        bin_path = bin_dir / binary
        if bin_path.exists() and bin_path.is_file():
            print(f"  ✓ {binary}")
        else:
            print(f"  ❌ {binary} not found")
            all_good = False

    print("\nCASA configuration:")
    casarc_ok, casarc_msg = check_casarc(casapath)
    if casarc_ok:
        print(f"  ✓ {casarc_msg}")
    else:
        print(f"  ⚠️  {casarc_msg}")
        # Warning only, not fatal

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
