---
github_issue: 7
---
# When Running Dry Run For Cfs Sync It Should Not Prompt You For Resolving Github Vs Cfs Conflicts

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

When you run `cfs gh sync --dry-run`, if there are any conflicts between the GitHub issues version and the CFS version of an issue, it asks you to resolve the conflict using four options. This should not happen in a dry run because a dry run should just be to preview what would happen if you were to actually sync. This feature should be removed from the dry run but kept in the real sync operation.

## Acceptance criteria

- When you run the above command using dry run, you should not be prompted to resolve any conflicts that might emerge between a GitHub issue and a CFS issue.
