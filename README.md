---
title: Kyc Env Environment Server
emoji: 🖥️
colorFrom: purple
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Kyc Env Environment
# KYC Data Auditor Environment (disha-kyc-v1)

### 🚀 Motivation
The **KYC-Audit** environment evaluates an agent's ability to perform multi-step data validation under strict priority constraints. In real-world finance, data integrity is binary; a small formatting error is less critical than a security flag. This environment tests if an LLM can navigate a **Priority Waterfall** (Drop > Flag > Normalize > Impute) without getting distracted by lower-priority issues.

### 🛠️ Environment Definition
- **Observation Space**: A `KYCObservation` object containing a user's `name`, `age`, `email`, `phone`, and `city`.
- **Action Space**: 
    - `3 (DROP)`: High-risk/Unusable data.
    - `4 (FLAG)`: Security/Age anomalies.
    - `2 (NORMALIZE)`: Formatting/Case issues.
    - `1 (IMPUTE)`: Missing secondary fields.
    - `0 (KEEP)`: Perfect record.

### 📝 Task Description
The agent acts as a **High-Security Auditor**. It must process 10 records. 
- **Difficulty**: Medium. Requires strict adherence to priority logic (e.g., ignoring a space if the identity is 'Unknown').
- **Reward**: `1.0` for correct priority action, `-0.5` for any mismatch.