#!/usr/bin/env bash

pip install cvxpy==1.0.31
CVXPY_TEST_FILE="1.0.31.txt" python solver.py
CVXPY_TEST_FILE="1.0.31_fix.txt" python solver_fix.py
pip install cvxpy==1.1.0
CVXPY_TEST_FILE="1.1.0.txt" python solver.py
CVXPY_TEST_FILE="1.1.0_fix.txt" python solver_fix.py