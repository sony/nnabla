#! /bin/sh

python -m pytest $@ \
    || ( ret=$?; [ $ret -eq 5 ] && (echo "No test is executed." ; exit 0) || exit $ret )
