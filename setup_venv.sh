#!/bin/bash

set -e

PYTHON_VER="3"
SUFFIX=""

if [[ "${1}" == "--dev" ]]; then
    SUFFIX="_dev"
    PYTHON_VER="3.7"
fi

python${PYTHON_VER} -m venv ".venv${SUFFIX}"
source ".venv${SUFFIX}/bin/activate"
echo "$(python --version)"
pip install -U pip
pip install -r "requirements${SUFFIX}.txt"
