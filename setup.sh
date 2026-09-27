#!/bin/sh
# Einmalige Vorbereitung; Anwendung danach direkt starten.
PROJECT_ROOT=$(CDPATH= cd -P "$(dirname "$0")" && pwd) || exit 1
exec sh "$PROJECT_ROOT/scripts/docker-launcher.sh" setup "$@"
