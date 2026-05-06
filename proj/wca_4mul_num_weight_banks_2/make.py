#!/usr/bin/env python3

#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path
import sys
import os

# --- PATHS ---
REMAP_PACKED_SRC = Path('../../../tiny/benchmark/training/image_classification/flatc_output/pretrainedResnet_clustered_quant_remap_packed.tflite')
REMAP_SRC        = Path('../../../tiny/benchmark/training/image_classification/flatc_output/pretrainedResnet_clustered_quant_remap.tflite')

DST1 = Path('../../third_party/mlcommons/tiny/v0.1/training/image_classification/trained_models/pretrainedResnet_quant.tflite')
DST2 = Path('build/src/tiny/v0.1/training/image_classification/trained_models/pretrainedResnet_quant.tflite')

MAKEFILE_PATH = "Makefile"   # path to your main Makefile

def do_copy(src, dsts):
    for dst in dsts:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f'\033[1;35m[INFO] Copied {src} -> {dst}\033[0m')

def ensure_conv_accelerate(makefile_path):
    with open(makefile_path, "r") as f:
        lines = f.readlines()
    # Remove all previous lines with exactly "DEFINES += CONV_ACCELERATE"
    lines = [l for l in lines if l.strip() != "DEFINES += CONV_ACCELERATE"]
    # Find first line with "export DEFINES :="
    idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("export DEFINES :="):
            idx = i
            break
    if idx is not None:
        # Insert after export DEFINES :=
        lines.insert(idx+1, "DEFINES += CONV_ACCELERATE\n")
        with open(makefile_path, "w") as f:
            f.writelines(lines)
        print("\033[1;32m[INFO] Inserted 'DEFINES += CONV_ACCELERATE' after export DEFINES :=\033[0m")
    else:
        print("\033[1;31m[ERROR] export DEFINES := not found in Makefile!\033[0m")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accelerate', action='store_true', help='Use packed model and enable CONV_ACCELERATE in Makefile')
    parser.add_argument('make_target', nargs='+', help='Make target to build/run (e.g. load, renode, renode-headless)')
    args = parser.parse_args()

    if args.accelerate:
        print('\033[1;32m[INFO] Using packed tflite & CONV_ACCELERATE\033[0m')
        do_copy(REMAP_PACKED_SRC, [DST1, DST2])
        ensure_conv_accelerate(MAKEFILE_PATH)
    else:
        print('\033[1;33m[INFO] Using remap tflite\033[0m')
        do_copy(REMAP_SRC, [DST1, DST2])
        # Optional: could remove the line from Makefile here if you want strict "no accelerate" hygiene

    print(f'\033[1;36m[INFO] Running make {" ".join(args.make_target)}\033[0m')
    subprocess.check_call(['make'] + args.make_target, env=os.environ)

if __name__ == '__main__':
    main()



