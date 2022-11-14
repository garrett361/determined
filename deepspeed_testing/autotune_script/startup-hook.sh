# Install less and vim
apt-get --yes install less vim
# Update deepspeed
pip install deepspeed==0.7.5
# Hack for seeing DEBUG logs from deepspeed
sed -i 's/level=logging.INFO/level=logging.DEBUG/g' /opt/conda/lib/python3.8/site-packages/deepspeed/utils/logging.py
# Set up the environment
export PDSH_SSH_ARGS="-o PasswordAuthentication=no -o StrictHostKeyChecking=no -p 12350 -2 -a -x %h"
export PDSH_SSH_ARGS_APPEND="-i /run/determined/ssh/id_rsa"
# Start the sshd server in the background
f
# Start in the workdir
echo "cd /run/determined/workdir" >> $HOME/.bash_login