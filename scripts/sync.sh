#!/usr/bin/env bash
# Quick sync: stage changed files, commit, push to origin.
# Usage:
#   ./scripts/sync.sh                  # auto-generates commit message
#   ./scripts/sync.sh "my message"     # custom commit message

set -e

BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Check for changes (tracked files only)
if git diff --quiet && git diff --cached --quiet; then
    echo "Nothing to sync — no changes to tracked files."
    exit 0
fi

# Stage all modified tracked files (not untracked)
git add -u

# Build commit message
if [ -n "$1" ]; then
    MSG="$1"
else
    # Auto-generate from changed file names
    FILES=$(git diff --cached --name-only | head -5)
    COUNT=$(git diff --cached --name-only | wc -l | tr -d ' ')
    if [ "$COUNT" -le 5 ]; then
        MSG="Sync: $(echo $FILES | tr '\n' ', ' | sed 's/,$//')"
    else
        MSG="Sync: ${COUNT} files updated"
    fi
fi

git commit -m "$MSG"
git push origin "$BRANCH"

echo ""
echo "Pushed to origin/$BRANCH"
echo "On Windows: git pull"
