#!/bin/sh
set -e

apt list --installed 2>/dev/null | awk -F '/' '{print $1}' >installed.txt

for p in $(cat deps/apt.txt); do
    if [ $(grep "^$p$" installed.txt | wc -l) -le 0 ]; then
        echo "NOT installed: $p"
    else
        echo "found: $p"
    fi
done

rm installed.txt
