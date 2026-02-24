#!/usr/bin/env bash
# cleanup.sh — Consolidate the legacy DATA/ directory into data/raw/DATA/
#
# The original repository uses an uppercase DATA/ directory at the repo root.
# Our Python package expects data under data/raw/DATA/ (which is git-ignored).
# This script copies the contents, preserving the originals until you confirm.
#
# Usage:
#   bash cleanup.sh          # copies DATA/ -> data/raw/DATA/
#   bash cleanup.sh --delete  # copies then removes the original DATA/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/DATA"
DST_DIR="${SCRIPT_DIR}/data/raw/DATA"

DELETE_AFTER=false
if [[ "${1:-}" == "--delete" ]]; then
    DELETE_AFTER=true
fi

if [[ ! -d "${SRC_DIR}" ]]; then
    echo "[INFO] No DATA/ directory found at repo root — nothing to do."
    exit 0
fi

echo "[INFO] Consolidating DATA/ -> data/raw/DATA/"

# Create destination tree
mkdir -p "${DST_DIR}"

# Copy chapter directories, preserving structure and permissions
for chapter_dir in "${SRC_DIR}"/*/; do
    chapter_name="$(basename "${chapter_dir}")"
    dst_chapter="${DST_DIR}/${chapter_name}"

    if [[ -d "${dst_chapter}" ]]; then
        echo "[SKIP] ${chapter_name} already exists in ${DST_DIR}"
    else
        echo "[COPY] ${chapter_name} -> ${dst_chapter}"
        cp -r "${chapter_dir}" "${dst_chapter}"
    fi
done

# Copy any loose files at the DATA/ root
for file in "${SRC_DIR}"/*; do
    [[ -d "${file}" ]] && continue
    fname="$(basename "${file}")"
    if [[ -f "${DST_DIR}/${fname}" ]]; then
        echo "[SKIP] ${fname} already exists in ${DST_DIR}"
    else
        echo "[COPY] ${fname} -> ${DST_DIR}/${fname}"
        cp "${file}" "${DST_DIR}/${fname}"
    fi
done

echo "[INFO] Data consolidation complete."
echo "       Source: ${SRC_DIR}"
echo "       Dest:   ${DST_DIR}"

if [[ "${DELETE_AFTER}" == true ]]; then
    echo "[INFO] Removing original DATA/ directory..."
    rm -rf "${SRC_DIR}"
    echo "[DONE] DATA/ removed."
else
    echo ""
    echo "[NOTE] Original DATA/ directory is still intact."
    echo "       Run 'bash cleanup.sh --delete' to remove it after verifying."
fi
