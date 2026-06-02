---
name: tqsdk-trading-and-data
description: "Guide TqSdk Python market data, trading, margin, simulation, backtest, and debugging workflows."
risk: safe
source: "https://github.com/shinny-xuyida/tqsdk-agent-skills/tree/main/skills/tqsdk-trading-and-data"
---

# TqSdk Trading and Data

Use this skill to answer TqSdk questions with the repository's real APIs, docs, and examples. Prefer minimal runnable snippets, keep futures and stock behavior separate, and explain the update loop explicitly whenever the user's issue depends on `wait_update()`.

## When to Use This Skill

- Use when the user asks about TqSdk, TqApi, TqAuth, TqAccount, TqKq, TqKqStock, TqSim, TqSimStock, TqMultiAccount, TqBacktest, TqScenario, TargetPosTask, TargetPosScheduler, or DataDownloader.
- Use when the user needs TqSdk market data, K-line, tick, historical data, account, position, order, trade, margin, simulation, backtest, or error-debugging help.
- Use when the user asks TqSdk questions in Chinese, especially about quotes, K-lines, historical data, margin, risk ratio, scenario trials, positions, fills, orders, accounts, order placement, cancellation, rebalancing, field meanings, or errors.

## Route The Request First

Read only the references needed for the user's question.

1. Read [references/wait-update-and-update-loop.md](references/wait-update-and-update-loop.md) for `wait_update`, `is_changing`, `deadline`, async update notifications, Jupyter caveats, or backtest progression questions.
2. Read [references/market-data.md](references/market-data.md) for `get_quote`, `get_kline_serial`, `get_tick_serial`, contract discovery, symbol metadata, and `DataDownloader`.
3. Read [references/account-type-matrix.md](references/account-type-matrix.md) for `TqAccount`, `TqKq`, `TqKqStock`, `TqSim`, `TqSimStock`, OTG account classes, and `TqMultiAccount`.
4. Read [references/accounts-and-trading.md](references/accounts-and-trading.md) for account, position, order, and trade getters plus multi-account getter patterns.
5. Read [references/scenario-and-margin.md](references/scenario-and-margin.md) for `TqScenario`, real-account margin-rate lookup, margin occupancy calculation, risk-ratio what-if analysis, and limited built-in margin discount rules.
6. Read [references/order-functions-and-position-tools.md](references/order-functions-and-position-tools.md) for `insert_order`, `cancel_order`, `TargetPosTask`, `support_open_min_volume`, `TargetPosScheduler`, and advanced execution helpers.
7. Read [references/object-fields.md](references/object-fields.md) when the user asks what fields mean on `Quote`, K-line or tick rows, `Account`, `Position`, `Order`, `Trade`, or their stock variants.
8. Read [references/simulation-and-backtest.md](references/simulation-and-backtest.md) for local sim, Quick sim, stock sim, backtest, and cross-account backtest limits.
9. Read [references/error-faq.md](references/error-faq.md) when the user asks about common TqSdk failures, confusing behavior, or exception messages.
10. Read [references/example-map.md](references/example-map.md) when you want a repository-backed example or doc page to imitate.

## Core Rules

1. Treat `get_*` results as live references, not snapshots. Explain that they refresh during `wait_update()`.
2. Explain `wait_update()` whenever the user is confused by missing data, stale fields, orders not leaving the client, or `TargetPosTask` not acting.
3. Distinguish futures and stock workflows:
   - Futures accounts and objects use `Account`, `Position`, `Order`, `Trade`.
   - Stock accounts and objects use `SecurityAccount`, `SecurityPosition`, `SecurityOrder`, `SecurityTrade`.
   - Stock trading does not use `offset`, and `TargetPosTask` is not for stock trading.
4. Choose account type before writing code. Do not default to `TqKq` or `TqAccount` unless the user really needs that account mode.
5. In multi-account mode, pass `account=` for getters and trading calls, or use the account object's own `get_account`, `get_position`, `get_order`, and `get_trade`.
6. For current market examples, avoid expired contracts. Prefer contract discovery APIs or main-contract symbols.
7. When the user asks for field meanings, explain the smallest relevant field set first and say whether the object is futures or stock.
8. When the user asks for long historical ranges, prefer `DataDownloader` over pretending `get_kline_serial` is an unlimited history API.
9. When the user asks for advanced execution, prefer public helpers first:
   - `TargetPosTask` for target net position.
   - `TargetPosTask(..., support_open_min_volume=True)` only for contracts with exchange minimum opening size rules when approximate completion is acceptable.
   - `TargetPosScheduler` plus `twap_table` or `vwap_table` from `tqsdk.algorithm` for scheduled execution.
   - Mention `InsertOrderTask` and `InsertOrderUntilAllTradedTask` as internal or advanced helpers, not the default answer.
10. Use `TqScenario` for synchronous what-if margin and risk trials, not for live order placement. It is futures-only, single-account, and updates the trial snapshot immediately after each call.
11. Explain the margin-rate source in every `TqScenario` answer:
   - `account=None` or `TqSim()` uses `Quote` margin data.
   - `TqAccount(...)` or `TqKq()` queries account-specific rates synchronously and may fall back to `Quote` margin if lookup fails.
12. Treat margin discounts conservatively. Reuse one `TqScenario` object for step-by-step trial actions, and do not promise broker-specific preferential rules beyond the limited built-in rules modeled by TqSdk.
13. Preserve exchange-specific close semantics in `TqScenario`. For SHFE or INE futures, keep `CLOSE` versus `CLOSETODAY` consistent with the imported position snapshot.

## Answering Style

- Prefer imports from `tqsdk.__init__` for top-level APIs. When an API is documented under a submodule, use that official submodule path such as `tqsdk.tools` or `tqsdk.algorithm`.
- Prefer short, correct code blocks over broad pseudo-code.
- Name the exact API the user should call next.
- If behavior differs in live trading, Quick sim, local sim, stock sim, or backtest, say so explicitly.
- If the answer depends on a common pitfall, state the pitfall directly instead of burying it in examples.
- If the answer uses `TqScenario`, say what snapshot goes into `positions`, what balance goes into `init_balance`, and whether the result comes from quote margin or account-specific margin rates.

## Examples

Ask:

```text
Use this skill to write a TqSdk example that downloads 1-minute K-lines and avoids expired contracts.
```

Ask:

```text
Use this skill to debug why my TqSdk insert_order call does not appear to send an order.
```

## Limitations

- This skill is guidance for agents. It does not connect to broker accounts or place trades by itself.
- Generated trading code must be reviewed before live use.
- Broker-specific margin discounts, preferential rules, and account permissions may differ from generic examples.
- Public examples should avoid expired contracts and should not expose account credentials.
