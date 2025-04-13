#!/usr/bin/env sh

# disable tmux
# touch ...

# set the shell prompt to a standard format.
echo 'PS1="\u@\h:\w\$ "' >> ~/.bashrc

# install clangd to enable lsp-mode
sudo apt install clangd
