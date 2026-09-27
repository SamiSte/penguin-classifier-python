#!/bin/sh
# Bereits vorbereitetes Image ohne Download oder Neubau starten.
PROJECT_ROOT=$(CDPATH= cd -P "$(dirname "$0")" && pwd) || exit 1
exec sh "$PROJECT_ROOT/scripts/docker-launcher.sh" start "$@"
