#!/bin/bash

pip install -r requirements.txt
python -c "import determined; print(determined.core._checkpoint.__file__)"
# sed -i 's/for fname in conflicts:/for fname in sorted(conflicts):/' /run/determined/pythonuserbase/lib/python3.10/site-packages/determined/core/_checkpoint.py
