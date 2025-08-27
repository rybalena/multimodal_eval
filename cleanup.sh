#!/bin/bash
# Cleanup script for macOS/Python project

echo "🧹 Cleaning up project from unnecessary files..."

# Remove macOS Finder files
find . -name ".DS_Store" -print -delete

# Remove Python cache
find . -name "__pycache__" -type d -exec rm -r {} +

# Remove egg-info metadata
find . -name "*.egg-info" -type d -exec rm -r {} +

# Remove build artifacts
rm -rf build dist

echo "✅ Cleanup finished!"
