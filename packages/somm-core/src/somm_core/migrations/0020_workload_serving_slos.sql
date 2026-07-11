-- Migration 0020 — serving-performance SLOs on workloads.
--
-- Extends workload adequacy constraints beyond end-to-end latency so goodput
-- and frontier fitness can evaluate first-token and decode-token behavior.

ALTER TABLE workloads ADD COLUMN max_p95_ttft_ms INTEGER;
ALTER TABLE workloads ADD COLUMN max_tpot_ms REAL;
