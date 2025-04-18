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

# install clangd to enable lsp-mode
sudo apt install clangd
