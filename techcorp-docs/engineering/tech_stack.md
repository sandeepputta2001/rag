# TechCorp Engineering Tech Stack

**Owner:** Engineering Leadership
**Last Updated:** Q1 2024
**Review Cycle:** Quarterly

---

## Backend

### Languages
| Language | Use Cases | Version |
|----------|-----------|---------|
| Python | API services (FastAPI, Django), ML pipelines, data engineering | 3.11+ |
| Go | High-performance microservices, CLI tooling, infrastructure agents | 1.22+ |
| TypeScript (Node.js) | Serverless functions, BFF (Backend for Frontend) layers | Node 20 LTS |

### Frameworks
- **FastAPI** — primary framework for new Python APIs. Async-first, OpenAPI spec auto-generated.
- **Django + DRF** — used for legacy monolith services and admin dashboards.
- **Gin** — Go HTTP framework for high-throughput microservices.

### Databases
| Database | Role | Managed Service |
|----------|------|-----------------|
| PostgreSQL 16 | Primary relational store | AWS RDS |
| MongoDB Atlas | Document store for unstructured data | Atlas M30+ |
| Redis 7 | Caching, session management, rate limiting | AWS ElastiCache |
| Snowflake | Data warehouse, analytics | Snowflake Enterprise |
| ClickHouse | Real-time OLAP, event analytics | Self-hosted on EKS |

### Message Queue / Streaming
- **Apache Kafka** (Confluent Cloud) — event streaming, service decoupling.
- **AWS SQS / SNS** — simpler async jobs and notifications.
- **Celery + Redis** — background task processing for Python services.

---

## Frontend

### Core Stack
- **React 18** with **TypeScript** — all new UI development.
- **Next.js 14** — server-side rendering, app router, React Server Components.
- **Tailwind CSS** — utility-first styling. No CSS-in-JS.
- **Radix UI** — accessible headless component primitives.

### State & Data Fetching
- **Zustand** — client-side state management (replaces Redux).
- **TanStack Query** (React Query) — server state, caching, and data fetching.
- **Zod** — runtime schema validation and TypeScript type inference.

### Testing (Frontend)
- **Vitest** — unit tests.
- **Playwright** — end-to-end tests (critical user journeys only).
- **Storybook** — component development and visual regression.

---

## Infrastructure & Cloud

### Cloud Provider
- **AWS** is the primary cloud provider (US-East-1 primary, US-West-2 DR).
- GCP is used for BigQuery and certain ML workloads.

### Container Orchestration
- **Kubernetes (EKS)** — all production services run on Kubernetes.
- **Helm** — chart management for service deployments.
- **ArgoCD** — GitOps continuous delivery to Kubernetes.

### Infrastructure as Code
- **Terraform** — all AWS and GCP resources. State stored in S3 with DynamoDB locking.
- **Terragrunt** — DRY Terraform configurations across environments.
- Modules are version-pinned and stored in the internal Terraform Registry.

### Observability
| Tool | Purpose |
|------|---------|
| Datadog | APM, infrastructure metrics, dashboards |
| OpenTelemetry | Distributed tracing standard |
| PagerDuty | On-call alerting and incident management |
| Sentry | Error tracking and release health |
| Grafana | Custom dashboards for business metrics |

### Networking
- **AWS ALB** — HTTP/HTTPS load balancing.
- **Cloudflare** — CDN, DDoS protection, WAF.
- **Istio** — service mesh for inter-service mTLS and traffic management.
- All internal services communicate over private VPC. No direct internet access from pods.

---

## CI/CD

- **GitHub Actions** — all build, test, and deploy pipelines.
- **Trunk-based development** — short-lived feature branches, merged to main within 2 days.
- Required checks before merge:
  - Unit + integration tests pass.
  - Code coverage ≥ 80%.
  - SAST scan passes (Semgrep).
  - Docker image vulnerability scan (Trivy).
  - At least 2 engineer approvals.

### Deployment Strategy
- **Blue/green deployments** for stateless services.
- **Rolling updates** for stateful services with careful migration management.
- Feature flags managed via **LaunchDarkly** to separate deploy from release.
- Automated rollback triggered if error rate exceeds 1% within 10 minutes of deploy.

---

## Security Engineering

- **Secrets management:** HashiCorp Vault for all service credentials. No secrets in environment variables or code.
- **SAST:** Semgrep runs on every PR.
- **DAST:** OWASP ZAP runs weekly against staging environments.
- **Dependency scanning:** Dependabot + Snyk for automated vulnerability PRs.
- **Penetration testing:** Annual external pentest + quarterly internal red team exercises.

---

## Code Review Standards

- All PRs require **2 engineer approvals** (senior engineer required for architecture changes).
- PRs should be scoped to a single concern and be reviewable in under 30 minutes.
- Inline comments must be resolved or explicitly deferred before merge.
- No `// TODO` comments merged without an associated Jira ticket.
- Code coverage minimum: **80%** for new files; existing files must not regress.

---

Contact Engineering Leadership: eng-leadership@techcorp.internal
Architecture decisions tracked in: Confluence > Engineering > ADRs
