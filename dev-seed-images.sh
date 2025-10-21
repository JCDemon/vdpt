#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <session-id>" >&2
  exit 1
fi

session_id="$1"
source_dir="samples/images"
target_dir="artifacts/uploads/${session_id}/images"

if [[ ! -d "$source_dir" ]]; then
  echo "Source directory '$source_dir' not found" >&2
  exit 1
fi

shopt -s nullglob
mkdir -p "$target_dir"
files=("$source_dir"/*.png "$source_dir"/*.jpg "$source_dir"/*.jpeg)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .png or .jpg files found in '$source_dir'" >&2
  exit 1
fi

cp -v "${files[@]}" "$target_dir"/

echo "Seeded $((${#files[@]})) image(s) into $target_dir"
