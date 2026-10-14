# Process log

## AI tools used

Cursor (Claude / Grok) to scaffold the FastAPI + LangGraph supervisor, RAG ingest, SSE frontend, and deploy script. Local Python 3.14 for ingestion and pytest. Runtime models: Gemini 3.8 Flash and NVIDIA Nemotron 3 Ultra via LangChain SDKs. Tavily Remote MCP for live search.

## Development narrative

Assignment 2 started inside the Assignment 1 folder. I moved it to its own repo so A1’s files, git history, and Cloud Run service `csci599-a1` were never edited. NVIDIA’s hosted embedding models on this key are retired (410 Gone), so ingest uses local `fastembed` (`BAAI/bge-small-en-v1.5`) into Chroma — about 4,620 chunks, ~19 minutes. The first `./deploy.sh` failed with `command not found`: a leading space after `GOOGLE_API_KEY=` made bash treat the key as a command. The next upload was 57 KB because Cloud Build follows `.gitignore` and dropped `chroma_index/`. A `.gcloudignore` that keeps the index fixed that.

## Verification narrative

Live `/health` returns the three agents and Tavily tools in 0.27 s warm. A RAG question (“What does a LangGraph checkpointer store?”) streamed 209 token events, cited `checkpointers.mdx` / `persistence.mdx`, and finished in 48.7 s (TTFT 4.5 s) after Gemini 429s and an NVIDIA fallback. Assignment 1’s `/health` still lists filesystem, Tavily, and weather. Gemini’s free tier is 20 requests/day; this pass exhausted it, so the measured path is the fallback.

## What I learned

The tools did not catch three things I had to debug myself: NVIDIA embeddings being retired, bash `source .env` treating an unquoted key as a command, and Cloud Build silently omitting the vector index because it follows `.gitignore`. I also learned the Gemini free-tier daily cap is real — a router that always “picks Gemini” is not a router once quota is gone; the inspectable fallback is what kept the deploy answering.
