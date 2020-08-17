set -e

export CLANG_FORMAT=clang-format

[ $(which $CLANG_FORMAT) ] || python3 -m pip install clang-format

format_dir() {
    find $1 -regex '.*\.\(cpp\|hpp\|cc\|cxx\)' -exec $CLANG_FORMAT -style=file -i {} \;
}

cd $(dirname $0)/..

format_dir ./examples
format_dir ./include
format_dir ./src
