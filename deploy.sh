#!/usr/bin/env bash
# Deploy to Cloud Run. Requires: gcloud auth, a billed GCP project, and
# GOOGLE_API_KEY / NVIDIA_API_KEY / TAVILY_API_KEY in the local .env.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${GCP_PROJECT_ID:?set GCP_PROJECT_ID in .env or the shell}"
: "${GCP_REGION:=us-west1}"
: "${CLOUD_RUN_SERVICE:=multi-agent-rag}"

if ! command -v gcloud >/dev/null; then
  echo "gcloud not found. Install the Google Cloud SDK, then re-run." >&2
  exit 127
fi

gcloud config set project "$GCP_PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud builds submit --tag "gcr.io/${GCP_PROJECT_ID}/${CLOUD_RUN_SERVICE}"
gcloud run deploy "$CLOUD_RUN_SERVICE" \
  --image "gcr.io/${GCP_PROJECT_ID}/${CLOUD_RUN_SERVICE}" \
  --platform managed \
  --region "$GCP_REGION" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 10 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "GOOGLE_API_KEY=${GOOGLE_API_KEY},GEMINI_MODEL=${GEMINI_MODEL:-gemini-3.8-flash},NVIDIA_API_KEY=${NVIDIA_API_KEY},NVIDIA_MODEL=${NVIDIA_MODEL:-nvidia/nemotron-3-ultra-550b-a55b},NVIDIA_BASE_URL=${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1},TAVILY_API_KEY=${TAVILY_API_KEY},STATE_BACKEND=sqlite"
