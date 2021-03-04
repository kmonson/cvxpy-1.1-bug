#!/usr/bin/env bash

pip install cvxpy==1.0.31
CVXPY_TEST_FILE="1.0.31.txt" python solver.py
pip install cvxpy==1.1.0
CVXPY_TEST_FILE="1.1.0.txt" python solver.py