#!/usr/bin/env bash
# Pulls the RAG corpus (LangGraph/LangChain + FastAPI docs) via sparse clones
# and records the exact commits in corpus/SOURCES.txt for reproducible re-indexing.
set -euo pipefail
cd "$(dirname "$0")/.."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
rm -rf corpus && mkdir -p corpus

git clone -q --depth 1 --filter=blob:none --sparse https://github.com/langchain-ai/docs "$tmp/lc"
git -C "$tmp/lc" sparse-checkout set src/oss/langgraph src/oss/langchain src/oss/concepts
for d in langgraph langchain concepts; do cp -R "$tmp/lc/src/oss/$d" "corpus/$d"; done

git clone -q --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi "$tmp/fa"
git -C "$tmp/fa" sparse-checkout set docs/en/docs
cp -R "$tmp/fa/docs/en/docs" corpus/fastapi

find corpus -type f ! -name '*.md' ! -name '*.mdx' -delete
find corpus -type d -empty -delete

{
  echo "langchain-ai/docs $(git -C "$tmp/lc" rev-parse HEAD)"
  echo "fastapi/fastapi   $(git -C "$tmp/fa" rev-parse HEAD)"
} > corpus/SOURCES.txt

echo "files: $(find corpus -type f \( -name '*.md' -o -name '*.mdx' \) | wc -l)"
cat corpus/SOURCES.txt
