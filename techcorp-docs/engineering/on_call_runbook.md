# On-Call Runbook

**Owner:** Platform Engineering
**Last Updated:** January 2024
**Review Cycle:** Quarterly

---

## Overview

This runbook defines the on-call responsibilities, escalation paths, and standard operating procedures for TechCorp's engineering on-call rotation.

---

## On-Call Schedule

- Rotation frequency: **weekly** (Monday 9 AM to next Monday 9 AM, Pacific Time).
- Each rotation requires a **primary** and a **secondary** on-call engineer.
- Schedule is managed in PagerDuty: pagerduty.techcorp.internal.
- Swap requests must be approved by the team lead and updated in PagerDuty at least 48 hours in advance.

### On-Call Compensation

| Time | Compensation |
|------|-------------|
| Weekday on-call availability | $200 / week stipend |
| Weekend on-call availability | $350 / weekend stipend |
| P1 incident response (nights/weekends) | 1.5× hourly rate for time worked |
| Response required at > 2 incidents / night | Comp day off the following week |

---

## Severity Levels

| Severity | Definition | Response Time | Example |
|----------|-----------|---------------|---------|
| P1 — Critical | Complete service outage or data loss | **15 minutes** | CloudSync down globally |
| P2 — High | Major feature broken, significant user impact | **1 hour** | File sync failing for >10% of users |
| P3 — Medium | Degraded performance or minor feature broken | **4 hours (business hours)** | Slow upload speeds |
| P4 — Low | Cosmetic issue or minor UX bug | **Next sprint** | Incorrect icon displayed |

---

## Escalation Path

```
Alert fires in PagerDuty
        │
        ▼
  Primary On-Call (5-min response window)
        │ no response
        ▼
  Secondary On-Call (escalated automatically)
        │ no response
        ▼
  Engineering Manager on duty
        │ if P1 > 30 mins unresolved
        ▼
  VP of Engineering + CTO bridge call
```

---

## Common Incident Playbooks

### CloudSync API Latency Spike
1. Check Datadog dashboard: `techcorp.cloudsync.api.p99_latency`.
2. Identify affected endpoint from APM traces.
3. Check RDS CPU and connection pool metrics.
4. If DB is the bottleneck: enable read replica routing via feature flag `db.readonly_routing`.
5. If Kubernetes pod count is low: `kubectl scale deployment cloudsync-api --replicas=20 -n production`.
6. Post incident update to #incidents Slack channel every 30 minutes.

### Authentication Service Down
1. Check Okta status page: status.okta.com.
2. If Okta is up, check TechCorp auth-service pods: `kubectl get pods -n auth -l app=auth-service`.
3. Check Vault seal status: `vault status` (unseal if sealed).
4. Roll back last deployment if it occurred within the past 2 hours: `argocd app rollback auth-service`.

### Database (RDS) High CPU
1. Run slow query report: `SELECT * FROM pg_stat_activity WHERE state = 'active' ORDER BY duration DESC LIMIT 20;`
2. Kill blocking queries if they have been running for > 5 minutes.
3. Check if a migration is running: `SELECT * FROM schema_migrations WHERE status = 'running';`
4. If CPU > 90% for > 10 minutes, promote read replica to primary via AWS RDS console.

---

## Post-Incident Process

All P1 and P2 incidents require a **Post-Incident Review (PIR)**:

1. PIR document created within **24 hours** of incident resolution.
2. Template: Confluence > Engineering > Incident Reviews > [PIR Template].
3. Timeline, root cause, contributing factors, and action items documented.
4. PIR reviewed in the next weekly engineering sync.
5. Action items tracked as Jira tickets with due dates.

Blameless culture: PIRs focus on systems and processes, not individual fault.

---

## Useful Commands

```bash
# Check all production pods health
kubectl get pods -n production | grep -v Running

# View recent deployments
argocd app list --output wide

# Check Kafka consumer lag
kafka-consumer-groups.sh --bootstrap-server kafka.internal:9092 --describe --group cloudsync-consumers

# Tail application logs
kubectl logs -f deployment/cloudsync-api -n production --tail=100
```

---

Contact Platform Engineering: platform-eng@techcorp.internal | PagerDuty: pagerduty.techcorp.internal
