# CFS Initialization

This directory was initialized using the Cursor File Structure (CFS) CLI.

## Categories

- **rules/** - Rules used by Cursor (automatically read by Cursor agents)
- **research/** - Research-related documents
- **bugs/** - Bug investigation and fix instructions
- **features/** - Feature development documents
- **refactors/** - Refactoring-related documents
- **docs/** - Documentation creation instructions
- **progress/** - Progress tracking and handoff documents
- **qa/** - Testing and QA documents
- **tmp/** - Temporary files for Cursor agent use

## Usage

Use the `cfs` CLI tool to manage documents in these categories.

*NOTE: The command `cfs instructions` has two aliases: `cfs i` and `cfs instr`.*

### Quick Start

```bash
# Create a new bug investigation document
cfs instructions bugs create

# Edit a document
cfs instructions bugs edit 1

# View all documents
cfs instructions view

# Create a rules document
cfs rules create
```

For help: `cfs --help`
