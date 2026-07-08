"""Debug tool: see exactly what the vector store retrieves (or contains).

No API key needed — this only exercises the retrieval half of the pipeline.

Usage:
    python eval/inspect_retrieval.py "your question here" [--k 10]
    python eval/inspect_retrieval.py --grep "25x25"        # find chunks containing a string
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from rag_assistant.config import get_settings
from rag_assistant.vectorstore import VectorStore

# PDFs contain characters Windows' default console codepage can't print.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="question to retrieve for")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--grep", help="list stored chunks containing this substring instead")
    parser.add_argument("--full", action="store_true", help="print full chunk text, not a snippet")
    args = parser.parse_args()

    store = VectorStore(get_settings().chroma_dir)
    print(f"Store: {store.count()} chunks from {list(store.list_sources())}\n")

    if args.grep:
        needle = args.grep.lower()
        hits = [
            (meta, text)
            for _, text, meta in store.iter_chunks()
            if needle in text.lower()
        ]
        print(f"{len(hits)} chunk(s) contain {args.grep!r}:\n")
        for meta, text in hits:
            body = text if args.full else textwrap.shorten(text, 300)
            print(f"--- {meta['source']} p.{meta['page']} ---\n{body}\n")
        return

    if not args.query:
        parser.error("provide a query or --grep")

    for rank, r in enumerate(store.search(args.query, k=args.k), start=1):
        body = r.text if args.full else textwrap.shorten(r.text, 240)
        print(f"#{rank}  score={r.score:.3f}  {r.source} p.{r.page}\n    {body}\n")


if __name__ == "__main__":
    main()
