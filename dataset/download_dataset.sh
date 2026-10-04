#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://huggingface.co/datasets/abdoelsayed/CORU/resolve/main"

is_valid() {
    local file="$1"
    [[ -s "$file" ]] || return 1

    case "$file" in
        *.zip)
            unzip -tq "$file" >/dev/null 2>&1
            ;;
        *.json)
            python -c "import json,sys; json.load(open(sys.argv[1]))" "$file" >/dev/null 2>&1
            ;;
        *)
            return 0
            ;;
    esac
}

download() {
    local url="$1"
    local file="$2"

    if [[ -f "$file" ]]; then
        if is_valid "$file"; then
            echo "Skipping $file (already downloaded and valid)"
            return 0
        fi
        echo "Existing $file is invalid, re-downloading"
        rm -f "$file"
    fi

    echo "Downloading $file"
    wget -q --show-progress -O "$file" "$url"
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

    mkdir -p "$dir"
    cd "$dir"
    download "$BASE_URL/OCR/$name.zip" "$name.zip"
    unzip_if_needed "$name.zip"
    cd - >/dev/null
}

download_receipt() {
    local name="$1"
    local dir="$2"

    mkdir -p "$dir"
    cd "$dir"
    download "$BASE_URL/Receipt/$name.zip" "$name.zip"
    download "$BASE_URL/Receipt/$name.json" "$name.json"
    unzip_if_needed "$name.zip"
    cd - >/dev/null
}

mkdir -p recognition/eng_arab
for split in test train val; do
    download_recognition "$split" "recognition/eng_arab"
done

mkdir -p detection/eng_arab
download "$BASE_URL/Receipt/labels.txt" "detection/eng_arab/labels.txt"
for split in test val; do
    download_receipt "$split" "detection/eng_arab"
done
