#!/usr/bin/env python3
"""
TechCorp AI Assistant — Flask application

New API endpoints (Part I implementation):
  POST /api/chat              — standard RAG chat (architecture + strategy selectable)
  POST /api/chat/stream       — streaming SSE version of /api/chat
  POST /api/ingest            — ingest a document from an uploaded file or raw text/HTML
  GET  /api/status            — system status + vector store stats
  POST /api/evaluate          — run baseline evaluation over the golden dataset
  GET  /api/architectures     — list available architectures
  GET  /api/strategies        — list available retrieval strategies
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import sys
import json
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "core"))

from core.telemetry import init_telemetry
from core.vector_engine import VectorEngine
from core.chat_engine import ChatEngine
from core.document_processor import DocumentProcessor
from core.retrieval import BM25Index
from core.evaluation import GoldenDataset, RAGEvaluator

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Initialise RAG stack
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  TechCorp AI Assistant — RAG Part I")
print("=" * 60)
print("\n[INIT] Loading RAG components...")

init_telemetry(service_name="techcorp-rag", console=True)
print("[INIT] Telemetry ready (OpenTelemetry → stdout)")

vector_engine = VectorEngine()
print("[INIT] Vector engine ready")

chat_engine = ChatEngine(vector_engine, architecture="simple", retrieval_strategy="top_k")
print("[INIT] Chat engine ready")

doc_processor = DocumentProcessor(vector_engine, chunk_strategy="recursive")
print("[INIT] Document processor ready")

# ---------------------------------------------------------------------------
# Available options
# ---------------------------------------------------------------------------

ARCHITECTURES = [
    {"id": "simple",         "label": "Simple RAG",        "description": "Top-K retrieve → prompt → generate"},
    {"id": "multihop",       "label": "Multi-Hop RAG",      "description": "Iterative sub-question decomposition"},
    {"id": "conversational", "label": "Conversational RAG", "description": "Follow-up questions with memory"},
    {"id": "agentic",        "label": "Agentic RAG",        "description": "ReAct tool-use loop"},
    {"id": "crag",           "label": "CRAG",               "description": "Self-correcting with relevance grading"},
    {"id": "graph",          "label": "Graph RAG",          "description": "Knowledge-graph enriched retrieval"},
]

STRATEGIES = [
    {"id": "top_k",       "label": "Top-K",         "description": "Dense vector similarity + score threshold"},
    {"id": "hybrid",      "label": "Hybrid (BM25+)", "description": "BM25 + vector search fused with RRF"},
    {"id": "multi_query", "label": "Multi-Query",   "description": "LLM-generated query variants"},
    {"id": "hyde",        "label": "HyDE",           "description": "Hypothetical Document Embeddings"},
    {"id": "filtered",    "label": "Filtered",       "description": "Date/ACL/doc-type post-filtering"},
]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        architecture = data.get("architecture", "simple")
        strategy = data.get("retrieval_strategy", "top_k")
        access_level = data.get("access_level", "public")

        response = chat_engine.get_response(
            user_message,
            architecture=architecture,
            retrieval_strategy=strategy,
            user_access_level=access_level,
        )

        return jsonify({
            "response": response["answer"],
            "sources": response["sources"],
            "confidence": response["confidence"],
            "provider": response.get("provider", "fallback"),
            "architecture": response.get("architecture", architecture),
            "retrieval_strategy": response.get("retrieval_strategy", strategy),
            "metadata": response.get("metadata", {}),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    def generate():
        try:
            data = request.json or {}
            user_message = data.get("message", "").strip()
            if not user_message:
                yield f"data: {json.dumps({'error': 'No message provided'})}\n\n"
                return

            architecture = data.get("architecture", "simple")
            strategy = data.get("retrieval_strategy", "top_k")

            yield f"data: {json.dumps({'event': 'start'})}\n\n"

            response = chat_engine.get_response(
                user_message,
                architecture=architecture,
                retrieval_strategy=strategy,
            )

            for word in response["answer"].split():
                time.sleep(0.04)
                yield f"data: {json.dumps({'event': 'token', 'content': word + ' '})}\n\n"

            yield f"data: {json.dumps({'event': 'sources', 'sources': response['sources'], 'confidence': response['confidence']})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Ingest content into the vector store.
    Accepts JSON: {"type": "text"|"html"|"pdf_path", "content": "...", "source": "...", "access_level": "public"}
    """
    try:
        data = request.json or {}
        content_type = data.get("type", "text")
        content = data.get("content", "")
        source = data.get("source", f"manual_{datetime.now().isoformat()}")
        access_level = data.get("access_level", "public")

        if not content:
            return jsonify({"error": "No content provided"}), 400

        if content_type == "html":
            chunks_data = doc_processor.process_html_content(content, source=source, access_level=access_level)
        else:
            # Treat as plain text — use recursive chunking
            from core.document_processor import recursive_chunk, build_metadata
            from core.foundations import make_chunk_id, count_tokens
            raw = recursive_chunk(content)
            total = len(raw)
            chunks_data = []
            for i, chunk in enumerate(raw):
                if chunk.strip():
                    cid = make_chunk_id(source, i, chunk)
                    meta = build_metadata(
                        source=source, doc_type="text",
                        chunk_index=i, total_chunks=total,
                        access_level=access_level,
                        extra={"token_count": count_tokens(chunk)},
                    )
                    chunks_data.append({"id": cid, "text": chunk, "metadata": meta})

        texts = [c["text"] for c in chunks_data]
        metas = [c["metadata"] for c in chunks_data]
        ids = [c["id"] for c in chunks_data]
        vector_engine.add_documents(texts, metas, ids=ids)

        # Rebuild BM25 after new documents
        chat_engine.rebuild_bm25()

        return jsonify({
            "status": "success",
            "chunks_ingested": len(chunks_data),
            "source": source,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Run baseline evaluation.
    Body (optional): {"subset": 5, "architecture": "simple", "strategy": "top_k"}
    """
    try:
        data = request.json or {}
        subset = data.get("subset", 5)
        arch = data.get("architecture", "simple")
        strategy = data.get("strategy", "top_k")

        def rag_fn(question: str):
            return chat_engine.get_response(
                question, architecture=arch, retrieval_strategy=strategy
            )

        dataset = GoldenDataset()
        dataset.load()
        evaluator = RAGEvaluator(rag_fn=rag_fn)
        report = evaluator.evaluate(
            dataset, architecture=arch, retrieval_strategy=strategy, subset=subset
        )
        evaluator.print_report(report)

        from dataclasses import asdict
        return jsonify(asdict(report))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    try:
        stats = vector_engine.get_stats()
        return jsonify({
            "status": "operational",
            "documents": stats["total_documents"],
            "chunks": stats["total_chunks"],
            "last_updated": stats["last_updated"],
            "provider": chat_engine.provider,
            "embedding_backend": stats.get("embedding_backend", "sentence_transformers"),
            "embedding_model": stats.get("embedding_model", "all-MiniLM-L6-v2"),
            "embedding_dim": stats.get("embedding_dim", 384),
            "faiss_index_size": stats.get("faiss_index_size", 0),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/architectures", methods=["GET"])
def list_architectures():
    return jsonify({"architectures": ARCHITECTURES})


@app.route("/api/strategies", methods=["GET"])
def list_strategies():
    return jsonify({"strategies": STRATEGIES})


@app.route("/api/conversation/reset", methods=["POST"])
def reset_conversation():
    """Clear the conversational RAG chat history."""
    chat_engine.reset_conversation()
    return jsonify({"status": "ok", "message": "Conversation memory cleared"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not vector_engine.is_initialized():
        print("\n[INIT] First run — processing TechCorp documents...")
        doc_processor.process_all_documents()
        chat_engine.rebuild_bm25()
        print("[INIT] Document processing complete!")

    app.run(host="0.0.0.0", port=5252, debug=True)
