# DataBridge — Real-Time Data Integration Platform

**Version:** 1.5
**Last Updated:** March 2024
**Product Owner:** Priya Nair
**Category:** Data Engineering / ETL

---

## Overview

DataBridge is TechCorp's managed data integration platform that connects cloud applications, databases, and data warehouses without writing custom ETL code. It supports real-time streaming and scheduled batch pipelines with a no-code visual builder.

---

## Key Features

### Connectors
- **300+ pre-built connectors** including:
  - Databases: PostgreSQL, MySQL, MongoDB, Snowflake, BigQuery, Redshift
  - SaaS apps: Salesforce, HubSpot, Shopify, Stripe, Zendesk
  - Cloud storage: S3, GCS, Azure Blob
  - Streaming: Kafka, Kinesis, Pub/Sub
- Custom connector SDK available for proprietary systems.

### Pipeline Modes
| Mode | Latency | Use Case |
|------|---------|----------|
| Real-time streaming | < 1 second | Event-driven apps, fraud detection |
| Micro-batch | 1–5 minutes | Near-real-time dashboards |
| Scheduled batch | Hourly / daily | Data warehouse loads, reporting |

### Transformation Engine
- Visual drag-and-drop transformation builder.
- SQL-based transformations for advanced users.
- Built-in data quality checks: null validation, type enforcement, deduplication.
- Alerting on schema changes in source systems.

### Monitoring
- Pipeline health dashboard with row counts, error rates, and latency metrics.
- Automatic retry on transient failures (configurable retry policy).
- Dead-letter queue for failed records.
- PagerDuty and Slack alerting integrations.

---

## Pricing

| Plan | Price | Data Volume | Pipelines |
|------|-------|-------------|-----------|
| Starter | $199 / month | 10 GB / month | 5 |
| Professional | $799 / month | 100 GB / month | 25 |
| Enterprise | Custom | Unlimited | Unlimited |

Additional data billed at $0.50 per GB (Starter), $0.30 per GB (Professional).

---

## Security

- In-transit encryption: TLS 1.3.
- At-rest encryption: AES-256.
- Credentials stored in HashiCorp Vault (never in plain text).
- IP allowlisting for connector sources.
- Audit log of all pipeline changes and executions.

Support: databridge-support@techcorp.com | Docs: docs.techcorp.com/databridge
