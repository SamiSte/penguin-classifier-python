#!/bin/sh
# Anwendung anhalten; gespeicherte Daten bleiben erhalten.
PROJECT_ROOT=$(CDPATH= cd -P "$(dirname "$0")" && pwd) || exit 1
exec sh "$PROJECT_ROOT/scripts/docker-launcher.sh" stop "$@"
