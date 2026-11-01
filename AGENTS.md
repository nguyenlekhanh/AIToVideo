Project Context Maintenance

The file docs/summary.md is the primary project context and AI handoff document.

Whenever a major task is completed or an important project change occurs, update docs/summary.md accordingly.

The update should include:

Changes to completed features.
Changes to ongoing or unfinished features.
Architecture changes.
Database and API changes.
New technical decisions and their reasoning.
Newly discovered bugs or resolved issues.
Updated development priorities and next steps.
Any new important files or dependencies.
Important Rules
Keep the summary consistent with the actual source code.
Do not document planned functionality as implemented.
Do not unnecessarily rewrite the entire document for minor changes.
Do not include API keys, passwords, tokens, or other sensitive information.
Preserve important historical technical decisions.
Keep the document useful for AI model handoffs.

The goal is to ensure that any new AI model can continue working on this project without losing important context.