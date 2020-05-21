#!/bin/bash
set -eux
PYTHON=python3
$PYTHON training/train.py
$PYTHON training/cross_validate.py
