#!/usr/bin/env sh

# disable tmux
touch ~/.no_auto_tmux

# set the shell prompt to a standard format.
echo '# Set the shell prompt to a standard format for tramp in emacs' >> ~/.bashrc

# Option 1
# echo 'PS1="\u@\h:\w\$ "' >> ~/.bashrc

# Option 2
cat << 'EOF' >> ~/.bashrc
if [[ $TERM == "dumb" ]]; then
  PS1='$ '
fi
EOF

# setup git
git config --global user.email "hyovin.kwak@tu-dortmund.de"
git config --global user.name "Hyovin Kwak"

# install clangd to enable lsp-mode
sudo apt install clangd
