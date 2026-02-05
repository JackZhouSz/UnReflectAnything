#!/usr/bin/env bash
# Sync docs/api_cli_reference.md to the GitHub wiki as "API-CLI-Reference".
#
# Usage:
#   ./scripts/sync_wiki.sh              # copy only (you commit/push in wiki repo)
#   ./scripts/sync_wiki.sh --push      # copy, commit, and push (requires wiki clone)
#
# Setup (one-time):
#   Clone the wiki repo next to this repo or set WIKI_DIR:
#     git clone https://github.com/alberto-rota/UnReflectAnything.wiki.git
#   If cloned as sibling:  ../UnReflectAnything.wiki
#   Or set:  export WIKI_DIR=/path/to/UnReflectAnything.wiki

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${REPO_ROOT}/docs/api_cli_reference.md"
WIKI_PAGE="API-CLI-Reference.md"

# Wiki directory: env > sibling > default
if [[ -n "${WIKI_DIR}" ]]; then
  WIKI_ROOT="${WIKI_DIR}"
elif [[ -d "${REPO_ROOT}/../UnReflectAnything.wiki" ]]; then
  WIKI_ROOT="${REPO_ROOT}/../UnReflectAnything.wiki"
else
  echo "Error: Wiki repo not found."
  echo "  Set WIKI_DIR to the wiki repo path, or clone it as a sibling:"
  echo "  git clone https://github.com/alberto-rota/UnReflectAnything.wiki.git"
  exit 1
fi

if [[ ! -f "${SOURCE}" ]]; then
  echo "Error: Source file not found: ${SOURCE}"
  exit 1
fi

cp "${SOURCE}" "${WIKI_ROOT}/${WIKI_PAGE}"
echo "Copied ${SOURCE} -> ${WIKI_ROOT}/${WIKI_PAGE}"

if [[ "${1:-}" == "--push" ]]; then
  cd "${WIKI_ROOT}"
  git add "${WIKI_PAGE}"
  if git diff --staged --quiet; then
    echo "No changes to commit."
  else
    git commit -m "Update API & CLI Reference"
    git push origin master
    echo "Committed and pushed to wiki."
  fi
else
  echo "To commit and push, run from the wiki repo:"
  echo "  cd ${WIKI_ROOT}"
  echo "  git add ${WIKI_PAGE}"
  echo "  git commit -m 'Update API & CLI Reference'"
  echo "  git push origin master"
  echo "Or run this script with --push:  ./scripts/sync_wiki.sh --push"
fi
