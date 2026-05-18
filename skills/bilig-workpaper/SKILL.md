---
name: bilig-workpaper
description: Formula WorkPaper runtime and MCP server for AI agents and Node.js services. Use when an agent needs spreadsheet-style formulas, cell edits, recalculation, readback verification, or persisted WorkPaper JSON without driving Excel UI.
risk: safe
source: proompteng/bilig (MIT)
---

# Bilig WorkPaper

## Overview

Bilig WorkPaper gives agents a reviewable spreadsheet model without requiring browser or Excel UI automation. Use it to edit cells through an API, recalculate formulas, verify display-value readback, and persist WorkPaper JSON for audit or restore flows.

## When to Use This Skill

- Use when an agent needs spreadsheet-style formulas in a backend service or local workflow.
- Use when the task involves editing input cells, recalculating formulas, and proving the output with readback.
- Use when Excel, Google Sheets, or screenshot-driven spreadsheet automation would be too brittle.
- Use when a workflow needs file-backed WorkPaper JSON that can be reviewed, restored, or versioned.

## Step-by-Step Guide

### 1. Install or run the package

Use the package directly with npm exec when you only need the MCP server:

```bash
npm exec --package @bilig/headless -- bilig-workpaper-mcp --workpaper ./pricing.workpaper.json --init-demo-workpaper --writable
```

For TypeScript code, install the package in the project:

```bash
npm install @bilig/headless
```

### 2. Prefer WorkPaper state over UI automation

Keep workbook state in a WorkPaper JSON document. Write inputs through the WorkPaper API or MCP tools, recalculate, then read the computed display value back from the model.

### 3. Use the MCP tools for agent workflows

A file-backed MCP session should expose tools for listing sheets, reading cells or ranges, setting cell contents, validating formulas, exporting the document, and reading display values after recalculation.

### 4. Verify after every mutation

After each write, read the changed input cell and the dependent output cell. Keep the exported WorkPaper JSON when the task needs proof, rollback, or a reproducible bug report.

## Examples

### Example 1: File-backed MCP server

```bash
npm exec --package @bilig/headless -- bilig-workpaper-mcp --workpaper ./pricing.workpaper.json --init-demo-workpaper --writable
```

An agent can then update `Inputs!B3`, read `Summary!B3`, and export the updated WorkPaper JSON.

### Example 2: TypeScript service workflow

```ts
import { WorkPaper } from "@bilig/headless";

const paper = new WorkPaper();
paper.setCellContents("Inputs", "B2", 1200);
paper.setCellContents("Inputs", "B3", "=B2*1.2");
paper.recalculate();

const result = paper.getCellDisplayValue("Inputs", "B3");
console.log(result);
```

## Best Practices

- Do use Bilig for formulas, readback, and persisted spreadsheet state when the workflow can live outside Excel UI.
- Do keep formula inputs, computed outputs, and exported WorkPaper JSON together when reporting bugs or validating agent changes.
- Do validate formulas before writing them into cells when the agent is generating formulas.
- Don't use screenshot or browser automation for spreadsheet logic if a WorkPaper API call can express the same operation directly.
- Don't treat a write as complete until a readback proves the computed output changed as expected.

## Troubleshooting

**Problem:** The agent wrote a formula but the output did not change.  
**Solution:** Recalculate the WorkPaper, then read both the input cell and the dependent display-value cell.

**Problem:** The workflow needs to preserve state across agent turns.  
**Solution:** Run the MCP server with `--workpaper ./file.workpaper.json --writable` and export the document after important changes.
