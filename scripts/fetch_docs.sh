#!/usr/bin/env bash
# Download the Kubernetes docs (concepts + tasks) into data/raw/k8s.
# Source: github.com/kubernetes/website, licensed CC BY 4.0. The docs change over time, so the
# commit that was fetched is recorded in data/raw/k8s/SOURCE.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 --filter=blob:none --sparse https://github.com/kubernetes/website.git "$TMP/site"
git -C "$TMP/site" sparse-checkout set content/en/docs/concepts content/en/docs/tasks

rm -rf data/raw/k8s && mkdir -p data/raw/k8s
cp -r "$TMP/site/content/en/docs/concepts" "$TMP/site/content/en/docs/tasks" data/raw/k8s/
echo "https://github.com/kubernetes/website @ $(git -C "$TMP/site" rev-parse HEAD)" > data/raw/k8s/SOURCE.txt

echo "Pages: $(find data/raw/k8s -name '*.md' | wc -l)   $(cat data/raw/k8s/SOURCE.txt)"
