#!/usr/bin/env sh

# disable tmux
touch ~/.no_auto_tmux

# set the shell prompt to a standard format.
echo '' >> ~/.profile
echo '# Set the shell prompt to a standard format for tramp in emacs' >> ~/.profile

# Option 1 too slow?
#echo 'PS1="\u@\h:\w\$ "' >> ~/.bashrc

# Option 2
cat << 'EOF' >> ~/.profile
if [[ $TERM == "dumb" ]]; then
  PS1='$ '
fi
EOF


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
sudo apt install clangd
#sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-17 100
