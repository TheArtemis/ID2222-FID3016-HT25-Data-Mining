#!/usr/bin/env bash
set -euo pipefail

# zips project root excluding files/dirs in .gitignore
# Usage: ./zipignore.sh [output.zip]
# If no git repo, falls back to rsync --exclude-from=.gitignore

cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"

if command -v git >/dev/null 2>&1; then
    PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
fi

if [ -z "${PROJECT_ROOT:-}" ]; then
    if [ -d "$SCRIPT_DIR/dumps" ]; then
        PROJECT_ROOT="$SCRIPT_DIR"
    else
        PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    fi
fi
DUMPS_DIR="$PROJECT_ROOT/dumps"
mkdir -p "$DUMPS_DIR"

timestamp="$(date +%Y%m%d%H%M%S)"
default_name="$(basename "$PROJECT_ROOT")-$timestamp.zip"

if [ $# -gt 0 ]; then
    out_name="$(basename -- "$1")"
    case "$out_name" in
        *.zip) ;;
        *) out_name="${out_name}.zip" ;;
    esac
else
    out_name="$default_name"
fi

# always place archive inside dumps/
OUT="$DUMPS_DIR/$out_name"

# operate from project root so git/rsync cover the full tree
cd "$PROJECT_ROOT"

command -v zip >/dev/null 2>&1 || { echo "zip is required"; exit 1; }

tmpfile="$(mktemp)"
trap 'rm -f "$tmpfile"' EXIT

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    # include tracked + untracked-but-not-ignored files
    git ls-files -z --cached --others --exclude-standard >"$tmpfile"
    # convert NUL -> newline for zip -@
    tr '\0' '\n' <"$tmpfile" >"${tmpfile}.nl"
    mv "${tmpfile}.nl" "$tmpfile"
    if [ ! -s "$tmpfile" ]; then
        echo "No files to archive."
        exit 1
    fi
    # create zip (place output outside project to avoid self-inclusion)
    zip -@ "$OUT" <"$tmpfile"
    echo "Created $OUT"
    exit 0
fi

# fallback: use .gitignore with rsync if available
if [ -f .gitignore ] && command -v rsync >/dev/null 2>&1; then
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"; rm -f "$tmpfile"' EXIT
    mkdir -p "$tmpdir/project"
    # copy everything except patterns in .gitignore and .git
    rsync -a --exclude-from='.gitignore' --exclude '.git' ./ "$tmpdir/project/"
    (cd "$tmpdir" && zip -r "$OUT" project)
    echo "Created $OUT"
    exit 0
fi

# last resort: zip everything except .git and the output zip (placed outside project)
echo "No git repo detected and no .gitignore/rsync fallback available. Zipping all files except .git."
zip -r "$OUT" . -x .git/\* 2>/dev/null
echo "Created $OUT"