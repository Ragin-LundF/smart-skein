#!/usr/bin/env bash
# Renders every .mmd in this directory to ../<name>.svg.
#
# The markdown keeps the Mermaid source in a collapsed block and shows the exported SVG, so the
# diagram is visible wherever the file is read and the source stays editable. Re-run after changing
# any .mmd; needs only Node.
set -euo pipefail
cd "$(dirname "$0")"
for source in *.mmd; do
  name="${source%.mmd}"
  echo "rendering ${name}"
  npx -y @mermaid-js/mermaid-cli@11 --input "${source}" --output "../${name}.svg" --backgroundColor white
done
