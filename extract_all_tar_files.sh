#!/usr/bin/env bash

MAIN_DIR="/media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE"

set -u

echo "============================================================"
echo " FAST TAR EXTRACTION SCRIPT"
echo "============================================================"
echo "Main dataset directory:"
echo "$MAIN_DIR"
echo

if [ ! -d "$MAIN_DIR" ]; then
    echo "ERROR: Main directory does not exist:"
    echo "$MAIN_DIR"
    exit 1
fi

archives_found=0
archives_skipped=0
archives_extracted=0
archives_failed=0

echo "Searching for tar files..."
echo

while IFS= read -r -d '' file; do
    archives_found=$((archives_found + 1))

    dir="$(dirname "$file")"
    filename="$(basename "$file")"

    # Remove common tar extensions to get expected output folder name
    folder_name="$filename"
    folder_name="${folder_name%.tar.gz}"
    folder_name="${folder_name%.tgz}"
    folder_name="${folder_name%.tar.bz2}"
    folder_name="${folder_name%.tbz2}"
    folder_name="${folder_name%.tbz}"
    folder_name="${folder_name%.tar.xz}"
    folder_name="${folder_name%.txz}"
    folder_name="${folder_name%.tar}"

    expected_folder="$dir/$folder_name"

    echo "------------------------------------------------------------"
    echo "Archive #$archives_found:"
    echo "$file"
    echo "Expected extracted folder:"
    echo "$expected_folder"

    if [ -d "$expected_folder" ]; then
        echo "Status: SKIPPED"
        echo "Reason: Extracted folder already exists."
        archives_skipped=$((archives_skipped + 1))
        echo
        continue
    fi

    echo "Status: NOT EXTRACTED"
    echo "Extracting in same directory..."
    echo "Command: tar -xvf \"$filename\""
    echo

    if (cd "$dir" && tar -xvf "$filename"); then
        echo
        echo "Status: EXTRACTION SUCCESSFUL"
        archives_extracted=$((archives_extracted + 1))
    else
        echo
        echo "Status: EXTRACTION FAILED"
        archives_failed=$((archives_failed + 1))
    fi

    echo
done < <(
    find "$MAIN_DIR" -type f \( \
        -iname "*.tar" -o \
        -iname "*.tar.gz" -o \
        -iname "*.tgz" -o \
        -iname "*.tar.bz2" -o \
        -iname "*.tbz" -o \
        -iname "*.tbz2" -o \
        -iname "*.tar.xz" -o \
        -iname "*.txz" \
    \) -print0
)

echo "============================================================"
echo " SUMMARY"
echo "============================================================"
echo "Archives found:        $archives_found"
echo "Skipped existing:      $archives_skipped"
echo "Extracted now:         $archives_extracted"
echo "Failed extractions:    $archives_failed"
echo "============================================================"
echo "Done."