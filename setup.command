#!/bin/sh
PROJECT_ROOT=$(CDPATH= cd -P "$(dirname "$0")" && pwd) || exit 1
exec sh "$PROJECT_ROOT/setup.sh" "$@"
