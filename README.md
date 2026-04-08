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

# 🛡️ KYC Data Cleaning RL Environment

This repository contains a **Reinforcement Learning (RL) Environment** designed to automate the cleaning and validation of KYC (Know Your Customer) records. The agent must learn to categorize data issues and apply the correct processing action to maximize data integrity.

---

## 🚀 Environment Description
The `KYCEnv` is a custom RL environment built for high-scale data processing. It simulates the real-world challenge of handling messy, incomplete, and suspicious user data. 

- **Problem Statement**: Automate the transition of raw, unverified data into a "Golden Record" state.
- **Dataset**: Real-time processing of over 200,000 records involving names, ages, emails, and contact details.

---

## 🎮 Markov Decision Process (MDP)

### 1. Observation Space (`KYCObservation`)
The agent perceives the world through a Pydantic-validated state:
* `record_id` (int): Unique identifier for the current record.
* `name` (str): User's full name (targets: trailing spaces, casing issues).
* `age` (int): User's age (targets: negative values, outliers > 120).
* `email` (str): Contact email (targets: test domains like `example.com`).
* `phone` (str): Contact number.
* `city` (str): Residential city.
* `step_count` (int): Current step in the episode.
* `episode_id` (str): Unique ID for the tracking session.

### 2. Action Space (`KYCAction`)
The agent can choose from 5 discrete actions:

| ID | Action | Description |
| :--- | :--- | :--- |
| **0** | **KEEP** | Data is clean; no changes required. |
| **1** | **IMPUTE** | Fill in a single missing field (Age, Email, Phone, or City). |
| **2** | **NORMALIZE** | Fix formatting (trim spaces, lowercase emails, fix casing). |
| **3** | **DROP** | Remove record (Invalid Age, Missing Name, or "Unknown" identity). |
| **4** | **FLAG** | Mark as suspicious (Age > 120 or Disposable/Test Email domains). |

### 3. Reward Function
- **`+1.0`**: For selecting the correct cleaning action based on internal priority logic.
- **`-0.5`**: For incorrect categorization or applying the wrong cleaning priority.

---

## ⚖️ Internal Priority Logic
To ensure consistency, the environment follows a strict hierarchy for determining the "Correct Action":
1. **DROP** (Highest Priority: Invalid Identity/Age)
2. **FLAG** (Suspicious: Outliers/Test domains)
3. **NORMALIZE** (Formatting: Trailing spaces/Casing)
4. **IMPUTE** (Missing Data: Null/NaN values)
5. **KEEP** (Clean: No issues detected)

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Pandas
- Pydantic v2
- OpenEnv Core

### Local Setup
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/disha9/disha-kyc-env.git](https://github.com/disha9/disha-kyc-env.git)
   cd disha-kyc-env