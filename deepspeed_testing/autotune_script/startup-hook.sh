# Install less and vim
apt-get --yes install less vim
# Update deepspeed
pip install "deepspeed[autotuning_ml]"==0.7.5 timm==0.6.7 torchmetrics==0.9.2 pyarrow datasets==2.7.0 evaluate==0.3.0 accelerate==0.14.0 setuptools==59.5.0 scikit-learn==1.1.3
# Hack for seeing DEBUG logs from deepspeed
sed -i 's/level=logging.INFO/level=logging.DEBUG/g' /opt/conda/lib/python3.8/site-packages/deepspeed/utils/logging.py
# Set up the environment
export PDSH_SSH_ARGS="-o PasswordAuthentication=no -o StrictHostKeyChecking=no -p 12350 -2 -a -x %h"
export PDSH_SSH_ARGS_APPEND="-i /run/determined/ssh/id_rsa"
# Start the sshd server in the background
/usr/sbin/sshd -p 12350 -f /run/determined/ssh/sshd_config -D&
# Start in the workdir
echo "cd /run/determined/workdir" >> $HOME/.bash_login
# pip install transformers from the cloned repo. Needed for run_glue
cd /run/determined/workdir/shared_fs/transformers
pip install .
# cd back to the workdir
cd /run/determined/workdir/
