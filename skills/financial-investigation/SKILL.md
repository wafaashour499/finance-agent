---
name: financial-investigation
description: Investigates financial transactions, unusual spending, monthly spending increases, and provides evidence-based recommendations.
---

# Financial Investigation

## Objective
Investigate personal and small-business financial transactions.

## Guidelines
- Start from the user's question; choose only the tools it needs.
- When something notable appears, drill down with get_category_trend and list_transactions before concluding.
- Compare statistical and ML findings when both were used.

## Rules
- Never invent financial figures.
- Use only the results returned by the tools.
- Distinguish spending increases from unusual transactions.
- A statistical or ML anomaly does not prove fraud.
- Do not double-count related findings.
- Never expose personal information.
- Never use or reveal the label field.

## Report Format
1. Financial Summary.
2. Main Findings.
3. Exactly one practical recommendation per finding.