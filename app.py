#!/usr/bin/env python3
"""
TechCorp AI Assistant - Interactive RAG Chat Interface
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import sys
from datetime import datetime
import json
import time

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.telemetry import init_telemetry
from core.vector_engine import VectorEngine
from core.chat_engine import ChatEngine
from core.document_processor import DocumentProcessor

app = Flask(__name__)

# Initialize RAG components
print("\n" + "="*60)
print("🚀 Starting TechCorp AI Assistant")
print("="*60)
print("\n[INIT] Loading RAG components...")

# Initialise OpenTelemetry before any component (console=True prints spans to stdout).
# Switch console=False and set OTEL_EXPORTER_OTLP_ENDPOINT to send to Jaeger/Tempo.
init_telemetry(service_name="techcorp-rag", console=True)
print("[INIT] Telemetry ready (OpenTelemetry → stdout)")

vector_engine = VectorEngine()
print("[INIT] Vector engine ready")

chat_engine = ChatEngine(vector_engine)
print("[INIT] Chat engine ready")

doc_processor = DocumentProcessor(vector_engine)
print("[INIT] Document processor ready")

@app.route('/')
def index():
    """Render the chat interface"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get response from RAG system
        response = chat_engine.get_response(user_message)
        
        return jsonify({
            'response': response['answer'],
            'sources': response['sources'],
            'confidence': response['confidence'],
            'provider': response.get('provider', 'fallback'),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Handle chat messages with streaming response"""
    def generate():
        try:
            data = request.json
            user_message = data.get('message', '')
            
            if not user_message:
                yield f"data: {json.dumps({'error': 'No message provided'})}\n\n"
                return
            
            # Send initial event
            yield f"data: {json.dumps({'event': 'start'})}\n\n"
            
            # Get response from RAG system
            response = chat_engine.get_response(user_message)
            
            # Stream the response word by word
            words = response['answer'].split()
            for i, word in enumerate(words):
                time.sleep(0.05)  # Small delay for streaming effect
                yield f"data: {json.dumps({'event': 'token', 'content': word + ' '})}\n\n"
            
            # Send sources at the end
            yield f"data: {json.dumps({'event': 'sources', 'sources': response['sources'], 'confidence': response['confidence']})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status"""
    try:
        stats = vector_engine.get_stats()
        return jsonify({
            'status': 'operational',
            'documents': stats['total_documents'],
            'chunks': stats['total_chunks'],
            'last_updated': stats['last_updated'],
            'provider': chat_engine.provider,
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    # Initialize database with documents on first run
    if not vector_engine.is_initialized():
        print("First run detected. Processing TechCorp documents...")
        doc_processor.process_all_documents()
        print("Document processing complete!")
    
    # Run the app
    app.run(host='0.0.0.0', port=5252, debug=True)