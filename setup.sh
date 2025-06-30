#!/usr/bin/env sh

# disable tmux
touch ~/.no_auto_tmux

# set the shell prompt to a standard format.
echo '' >> ~/.profile
echo '# Set the shell prompt to a standard format for tramp in emacs' >> ~/.profile

# Option 1 could be too slow? use Option 2
#echo 'PS1="\u@\h:\w\$ "' >> ~/.bashrc

# Option 2
cat << 'EOF' >> ~/.profile
if [[ $TERM == "dumb" ]]; then
  PS1='$ '
fi
EOF

# keep vterm alive
echo '' >> /etc/ssh/sshd_config
echo '# keep vterm alive' >> /etc/ssh/sshd_config
cat << 'EOF' >> /etc/ssh/sshd_config
ClientAliveInterval 60
ClientAliveCountMax 3
EOF
sudo systemctl restart sshd

# add nvcc in .profile
echo '' >> ~/.profile
echo '# add nvcc ' >> ~/.profile
cat << 'EOF' >> ~/.profile
export PATH=/usr/local/cuda/bin:$PATH
EOF
source ~/.profile

# setup git
git config --global user.email "hyovin.kwak@tu-dortmund.de"
git config --global user.name "Hyovin Kwak"

# install clangd to enable lsp-mode
# sudo apt install clangd-14
# sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-14 100
sudo apt install clangd

# install cuda tool kit if not installed.
if ! command -v nvcc >/dev/null 2>&1 && [ ! -d "/usr/local/cuda" ]; then
    apt-get update && apt-get install -y cuda-toolkit-12-6
else
    echo "CUDA is already installed or present in /usr/local/cuda."
fi
