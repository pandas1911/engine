# Onboarding

- **Read `docs/codebase-structure.md` before writing or modifying code.**
  It contains a detailed breakdown of the project's module layout, data flow, key design patterns, and class-level documentation. Familiarizing yourself with it upfront will save significant time when navigating the codebase.

# Rules

1. **Do not modify core code under `engine/` without explicit approval.**
   If a change is necessary, you must first explain the **reason** and **scope** of the modification. Proceed only after receiving approval.

2. **Do not commit or push changes to the Git repository without permission.**
   All code changes must be submitted for review first. Commits are made only after explicit approval is granted.

3. **All code comments must be written in English.**
   Do not use Chinese or any other language in inline comments, docstrings, or documentation within the codebase.

4. **Keep `docs/codebase-structure.md` in sync with the actual codebase.**
   After any code change — and again before committing — verify that the structure document still accurately reflects the current code. If anything has changed (modules added/removed, class signatures updated, data flow shifted, etc.), update the document accordingly.

# Notes

- **Runtime logs are stored in the `logs/` directory.**
  Before starting any debugging work, review the relevant log files first to inform your analysis and optimization.