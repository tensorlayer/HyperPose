set -e

export CLANG_FORMAT=clang-format-9

find ./include -regex '.*\.\(cpp\|hpp\|cc\|cxx\)' -exec $CLANG_FORMAT -style=file -i {} \;
find ./src -regex '.*\.\(cpp\|hpp\|cc\|cxx\)' -exec $CLANG_FORMAT -style=file -i {} \;
find ./examples -regex '.*\.\(cpp\|hpp\|cc\|cxx\)' -exec $CLANG_FORMAT -style=file -i {} \;