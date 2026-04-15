"""
Telemetry setup for TechCorp RAG application.

Two telemetry layers:
  1. ChromaDB product telemetry  — anonymized usage stats sent to Chroma's
     PostHog analytics (opt-in/out via `anonymized_telemetry` in Settings).
  2. OpenTelemetry (OTEL) tracing — structured spans capturing latency,
     query text, result counts, and similarity scores for every RAG operation.
     Exportable to any OTLP backend (Jaeger, Grafana Tempo, Honeycomb, etc.).
     In this setup we use ConsoleSpanExporter so spans are visible in stdout.
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    ConsoleSpanExporter,
    BatchSpanProcessor,
)

_initialized = False
_tracer: trace.Tracer | None = None


def init_telemetry(service_name: str = "techcorp-rag", console: bool = True) -> trace.Tracer:
    """
    Initialise OpenTelemetry tracing.

    Args:
        service_name: Identifies this service in trace backends.
        console:      True  → print spans to stdout (dev/demo mode).
                      False → send spans to OTLP endpoint configured via
                              OTEL_EXPORTER_OTLP_ENDPOINT env variable.
    Returns:
        A tracer instance ready for use.
    """
    global _initialized, _tracer
    if _initialized:
        return _tracer  # type: ignore[return-value]

    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    if console:
        # SimpleSpanProcessor exports each span immediately — good for demos.
        # In production use BatchSpanProcessor to avoid blocking the main thread.
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        # Production: send to any OTLP-compatible backend (Jaeger, Tempo, …)
        # Set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 (gRPC)
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        import os
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    _initialized = True
    return _tracer


def get_tracer() -> trace.Tracer:
    """Return the module-level tracer (initialise with defaults if needed)."""
    global _tracer
    if _tracer is None:
        return init_telemetry()
    return _tracer
