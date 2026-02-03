# Bash completion for unreflect
# Source this file or add to .bashrc: source /path/to/cli/unreflect-completion.bash

_unreflect() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local subcommands="train test sweep agent"
    local options="-h --help"
    COMPREPLY=($(compgen -W "$subcommands $options" -- "$cur"))
}

complete -F _unreflect unreflect
