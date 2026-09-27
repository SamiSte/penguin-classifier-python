#!/bin/sh
# Gemeinsamer Startablauf fuer Linux und macOS, ohne Shell-Erweiterungen.
PROJECT_ROOT=$(CDPATH= cd -P "$(dirname "$0")/.." && pwd) || exit 1
ACTION=${1:-}
if [ "$#" -gt 0 ]; then
    shift
fi
OPEN_BROWSER=1
PAUSE=1

finish() {
    result=$1
    if [ "$PAUSE" -eq 1 ] && [ -t 0 ]; then
        printf '\nZum Schliessen die Eingabetaste druecken ... '
        IFS= read -r answer || :
    fi
    exit "$result"
}

fail() {
    printf '\nFehler: %s\n' "$1" >&2
    finish 1
}

usage() {
    printf '%s\n' \
        'Aufruf: sh setup.sh|start.sh|stop.sh [--no-browser] [--no-pause]' \
        'setup: Image bauen und starten (einmalig mit Internetzugang).' \
        'start: Vorbereitetes Image lokal und ohne Download starten.' \
        'stop: Anwendung anhalten; Beobachtungen und Modelle behalten.' \
        '--no-browser: Browser nicht automatisch oeffnen.' \
        '--no-pause: Am Ende nicht auf eine Eingabe warten.'
}

for option in "$@"; do
    case "$option" in
        --no-browser) OPEN_BROWSER=0 ;;
        --no-pause) PAUSE=0 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unbekannte Option: %s\n' "$option" >&2; usage >&2; exit 2 ;;
    esac
done

case "$ACTION" in
    setup|start|stop) ;;
    *) usage >&2; exit 2 ;;
esac

PENGUIN_PORT=${PENGUIN_PORT:-8050}
case "$PENGUIN_PORT" in
    ''|*[!0-9]*) fail 'PENGUIN_PORT muss eine ganze Zahl zwischen 1 und 65535 sein.' ;;
esac
if ! [ "$PENGUIN_PORT" -ge 1 ] 2>/dev/null || ! [ "$PENGUIN_PORT" -le 65535 ] 2>/dev/null; then
    fail 'PENGUIN_PORT muss eine ganze Zahl zwischen 1 und 65535 sein.'
fi
export PENGUIN_PORT

[ -f "$PROJECT_ROOT/compose.yaml" ] || fail 'Die Datei compose.yaml fehlt. Bitte den vollstaendigen Projektordner verwenden.'

DOCKER=$(command -v docker 2>/dev/null) || DOCKER=''
if [ -z "$DOCKER" ] && [ -x /Applications/Docker.app/Contents/Resources/bin/docker ]; then
    DOCKER=/Applications/Docker.app/Contents/Resources/bin/docker
fi
[ -n "$DOCKER" ] || fail 'Docker wurde nicht gefunden. Bitte Docker gemaess der Anleitung einrichten und danach erneut starten.'

"$DOCKER" info >/dev/null 2>&1 || fail 'Docker ist nicht erreichbar. Bitte die Docker-Laufzeit (auf macOS Docker Desktop) starten und warten, bis sie bereit ist. Unter Linux muss Ihr Benutzer auf Docker zugreifen koennen.'
"$DOCKER" compose version >/dev/null 2>&1 || fail 'Docker Compose ist nicht verfuegbar. Bitte Docker Compose v2 gemaess der Anleitung einrichten.'

compose() {
    "$DOCKER" compose --project-directory "$PROJECT_ROOT" -f "$PROJECT_ROOT/compose.yaml" "$@"
}

if [ "$ACTION" = stop ]; then
    printf 'Pinguin-Klassifikator wird angehalten ...\n'
    compose stop app || fail 'Die Anwendung konnte nicht angehalten werden. Bitte die Docker-Fehlermeldung oben beachten.'
    printf 'Anwendung angehalten. Beobachtungen und Modellversionen bleiben gespeichert.\n'
    finish 0
fi

if [ "$ACTION" = setup ]; then
    printf 'Anwendung wird vorbereitet. Beim ersten Mal ist Internetzugang erforderlich.\n'
    compose build app || fail 'Das Image konnte nicht gebaut werden. Bitte die Docker-Fehlermeldung oben beachten und die Einrichtung erneut ausfuehren.'
else
    "$DOCKER" image inspect penguin-classifier:local >/dev/null 2>&1 || fail 'Das vorbereitete Image fehlt. Bitte zuerst mit Internetzugang setup.command (macOS) oder sh setup.sh (Linux) ausfuehren.'
fi

printf 'Pinguin-Klassifikator wird gestartet ...\n'
compose up -d --no-build --pull never --wait --wait-timeout 60 app || fail 'Die Anwendung ist nicht startbereit. Bitte die Docker-Fehlermeldung oben beachten. Moegliche Ursachen sind ein belegter Port oder ein Startfehler. Docker Compose muss --wait unterstuetzen.'

APP_URL="http://127.0.0.1:$PENGUIN_PORT"
printf '\nAnwendung bereit: %s\n' "$APP_URL"
if [ "$OPEN_BROWSER" -eq 1 ]; then
    if [ "$(uname -s)" = Darwin ] && command -v open >/dev/null 2>&1; then
        open "$APP_URL" >/dev/null 2>&1 || printf 'Bitte die obige Adresse im Browser oeffnen.\n'
    elif command -v xdg-open >/dev/null 2>&1; then
        # xdg-open kann auf den Browserprozess warten; der Starter bleibt frei.
        xdg-open "$APP_URL" >/dev/null 2>&1 &
        printf 'Falls sich kein Browser oeffnet, bitte die obige Adresse aufrufen.\n'
    else
        printf 'Bitte die obige Adresse im Browser oeffnen.\n'
    fi
fi
finish 0
