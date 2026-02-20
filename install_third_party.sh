#!/bin/bash

set -e

source ~/.bashrc
conda activate diffogenv

mkdir -p third_party
cd third_party
git clone https://github.com/real-stanford/diffusion_policy.git
cd diffusion_policy
touch diffusion_policy/__init__.py
pip install -e .
cd ..
cd ..
pip install -e .
