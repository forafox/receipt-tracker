#!/usr/bin/env bash
set -euo pipefail

CORU_REPO="abdoelsayed/CORU"
RU_REPO="cdek-ocr/receipt-ocr-ru"

is_valid() {
    local file="$1"
    [[ -s "$file" ]] || return 1

    case "$file" in
        *.zip)
            unzip -tq "$file" >/dev/null 2>&1
            ;;
        *.json)
            python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$file" >/dev/null 2>&1
            ;;
        *)
            return 0
            ;;
    esac
}

download() {
    local repo="$1"
    local repo_path="$2"
    local dir="$3"
    local name
    name="$(basename "$repo_path")"

    mkdir -p "$dir"

    if [[ -f "$dir/$name" ]] && is_valid "$dir/$name"; then
        echo "Skipping $dir/$name (already downloaded and valid)"
        return 0
    fi

    echo "Downloading $dir/$name"
    hf download "$repo" \
        --repo-type dataset \
        --include "$repo_path" \
        --local-dir "$dir"

    if [[ "$dir/$repo_path" != "$dir/$name" ]]; then
        mv -f "$dir/$repo_path" "$dir/$name"
        rmdir -p --ignore-fail-on-non-empty "$dir/$(dirname "$repo_path")" 2>/dev/null || true
    fi
}

unzip_if_needed() {
    local file="$1"
    local dir="${file%.zip}"

    if [[ -d "$dir" ]]; then
        echo "Skipping unzip $file ($dir already exists)"
        return 0
    fi

    echo "Unzipping $file"
    unzip -q "$file" -d "$dir"
}

download_recognition() {
    local name="$1"
    local dir="$2"

    download "$CORU_REPO" "OCR/$name.zip" "$dir"
    unzip_if_needed "$dir/$name.zip"
}

download_receipt() {
    local name="$1"
    local dir="$2"

    download "$CORU_REPO" "Receipt/$name.zip" "$dir"
    download "$CORU_REPO" "Receipt/$name.json" "$dir"
    unzip_if_needed "$dir/$name.zip"
}

download_ru() {
    local name="$1"
    local dir="$2"

    mkdir -p "$dir"
    hf download "$RU_REPO" \
        --repo-type dataset \
        --include "images/$name/*" \
        --include "annotations/$name.jsonl" \
        --local-dir "$dir"
}

mkdir -p recognition/eng_arab
for split in test train val; do
    download_recognition "$split" "recognition/eng_arab"
done

mkdir -p detection/eng_arab
download "$CORU_REPO" "Receipt/labels.txt" "detection/eng_arab"
for split in test val; do
    download_receipt "$split" "detection/eng_arab"
done

mkdir -p detection/ru
for split in test train validation; do
    download_ru "$split" "detection/ru"
done
