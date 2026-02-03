# Zsh completion for unreflect
# Add to .zshrc: fpath=(/path/to/cli $fpath) then run compinit, or:
# source /path/to/cli/unreflect-completion.zsh

_unreflect() {
    local -a subcommands
    subcommands=(train test sweep agent)
    _arguments -C \
        '(-h --help)'{-h,--help}'[Show help and exit]' \
        '1:subcommand:($subcommands)' \
        '*::args: _normal'
}

(( $+functions[_unreflect] )) && compdef _unreflect unreflect
