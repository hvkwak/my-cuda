#!/usr/bin/env sh

# disable tmux
# touch ...

# set the shell prompt to a standard format.
echo 'PS1="\u@\h:\w\$ "' >> ~/.bashrc

#TODO: see if this code can work
#if [[ $TERM == "dumb" ]]; then
#    export PS1="$ "
#fi

# install clangd to enable lsp-mode
sudo apt install clangd
