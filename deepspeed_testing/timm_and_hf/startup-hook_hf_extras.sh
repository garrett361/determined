#!/bin/bash
pip install datasets==2.7.0 evaluate==0.3.0 accelerate==0.14.0 setuptools==59.5.0 scikit-learn==1.1.3
# pip install transformers from the cloned repo. Needed for run_glue
cd /run/determined/workdir/shared_fs/transformers
pip install .
# cd back to the workdir
cd /run/determined/workdir/
