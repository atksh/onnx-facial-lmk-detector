#!/bin/bash
set -eux

sh download.sh
python prepare.py
python main.py

# check
python inference.py

