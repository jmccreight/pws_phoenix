#!/bin/sh
# .github/scripts/ruff_check.sh
# Pre-commit hook to run ruff check and format on staged Python and Jupyter files

# Get list of staged Python and Jupyter notebook files
STAGED_PY_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
STAGED_IPYNB_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.ipynb$')

# Combine both lists
STAGED_FILES="$STAGED_PY_FILES $STAGED_IPYNB_FILES"

if [ -z "$STAGED_FILES" ]; then
    echo "No Python or Jupyter notebook files staged, skipping ruff checks."
    exit 0
fi

echo "Running ruff check on staged files..."
echo "$STAGED_FILES" | xargs ruff check
if [ $? -ne 0 ]; then
    echo "ruff check failed. Please fix errors before committing."
    exit 1
fi

echo "Running ruff format on staged files..."
echo "$STAGED_FILES" | xargs ruff format
if [ $? -ne 0 ]; then
    echo "ruff format failed. Please fix errors before committing."
    exit 1
fi

# Re-stage any files that were reformatted
echo "$STAGED_FILES" | xargs git add

echo "ruff checks passed!"
exit 0
