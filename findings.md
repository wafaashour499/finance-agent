# Findings

## 1. Overall Financial Position
- **Total Income:** 90000.0
- **Total Expenses:** 202230.03
- **Net Savings:** -112230.03 (deficit)

## 2. Spending Increase – Transport
- **Period:** 2026-01 → 2026-02
- **Previous Amount:** 6178.5
- **Current Amount:** 8689.0
- **Increase:** 40.63%
- **Category Trend:**
  - 2026-01 total 6178.5 (42 transactions)
  - 2026-02 total 8689.0 (55 transactions) – 40.63% increase
  - Subsequent months: 2026-03 8912.6, 2026-04 10908.34, 2026-05 6321.65, 2026-06 6124.8
- **Top February Transactions (first 20 shown):**
  - 2026-02-03: 251.57
  - 2026-02-15: 224.63
  - 2026-02-16: 208.14
  - 2026-02-06: 206.21
  - 2026-02-12: 205.69
  - 2026-02-17: 204.56
  - 2026-02-10: 201.98
  - 2026-02-02: 187.62
  - 2026-02-23: 185.15
  - 2026-02-04: 183.9
  - *(additional similar amounts omitted for brevity)*

## 3. Spending Increase – Shopping
- **Period:** 2026-04 → 2026-05
- **Previous Amount:** 2521.25
- **Current Amount:** 33009.93
- **Increase:** 1209.27%
- **Category Trend:**
  - 2026-04 total 2521.25 (4 transactions)
  - 2026-05 total 33009.93 (11 transactions) – 1209.27% increase
  - Subsequent month 2026-06 total 3268.48 (6 transactions)
- **May Transactions (all 11):**
  - 2026-05-24: 6252.89
  - 2026-05-22: 6196.26
  - 2026-05-21: 5971.81
  - 2026-05-20: 5766.48
  - 2026-05-23: 4745.55
  - 2026-05-07: 863.9
  - 2026-05-16: 827.18
  - 2026-05-08: 738.33
  - 2026-05-08: 643.89
  - 2026-05-02: 620.08
  - 2026-05-11: 383.56

## 4. Statistical & ML Anomalies – Transport (Large Single Transaction)
- **Date:** 2026-04-10
- **Description:** Transport
- **Amount:** 3000.0
- **Z‑Score:** 16.47 (statistical)
- **Isolation Forest:** flagged (rolling_count 6, rolling_sum 11327.43)

## 5. Statistical & ML Anomalies – Food (Restaurant)
- **Date:** 2026-03-15
- **Description:** Restaurant
- **Amount:** 8500.0
- **Z‑Score:** 11.28 (statistical)
- **Isolation Forest:** flagged (rolling_count 6, rolling_sum 11327.43)

## 6. Statistical & ML Anomalies – Shopping (May High‑Value Transactions)
- **Dates & Amounts:**
  - 2026-05-20: 5766.48 (Z‑Score 2.61)
  - 2026-05-21: 5971.81 (Z‑Score 2.73)
  - 2026-05-22: 6196.26 (Z‑Score 2.85)
  - 2026-05-23: 4745.55 (Z‑Score 2.02)
  - 2026-05-24: 6252.89 (Z‑Score 2.89)
- **Isolation Forest:** each flagged with increasing rolling counts (2‑5) and cumulative sums shown in ML anomalies output.

# Recommendations

1. **Overall Deficit:** Review all expense categories to identify major contributors and verify the necessity of high‑cost items.
2. **Transport Increase:** Verify the February Transport transactions for necessity, especially given the 40.63% month‑to‑month rise.
3. **Shopping Surge:** Verify the large May Shopping purchases, as they drove a 1209.27% increase.
4. **Transport 3000 Transaction:** Confirm the purpose of the 3000.0 transport expense on 2026‑04‑10.
5. **Food Restaurant 8500 Transaction:** Confirm the reason for the 8500.0 restaurant expense on 2026‑03‑15.
6. **May Shopping Anomalies:** Review each high‑value Shopping transaction in May (amounts between 4745.55 and 6252.89) for legitimacy.
