#!/bin/bash

python3 JLAL_runner.py >> time_results_LJAL.log
python3 DCOP_runner.py >> time_results_DCOP.log
python3 CG_runner.py >> time_results_CG.log
