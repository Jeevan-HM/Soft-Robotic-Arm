#!/usr/bin/env zsh
# run.sh — Launch the SOFA soft arm simulation
#
# Usage:
#   ./run.sh                        # open scene in SOFA GUI
#   ./run.sh mesh/generate_mesh.py  # run any other Python script with SOFA env
#
# This script sets the three environment variables that SOFA needs on macOS:
#   SOFA_ROOT            — path to the SOFA installation
#   PYTHONPATH           — path to SofaPython3 Python bindings
#   DYLD_FRAMEWORK_PATH  — tells dyld where Python.framework lives
#                          (required because SOFA's .dylibs use @rpath lookups)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export SOFA_ROOT="$SCRIPT_DIR/SOFA"
export PYTHONPATH="$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH"
export DYLD_FRAMEWORK_PATH="/Library/Frameworks:$DYLD_FRAMEWORK_PATH"

SCENE="${1:-soft_arm_scene.py}"

# SofaImGui displays Python-created numbers as read-only text.  Launch a small
# native control window with real text/spin boxes and bridge it to the scene on
# localhost.  A per-run port lets multiple simulations coexist safely.
export SOFT_ARM_PRESSURE_PORT=$((47631 + RANDOM % 1000))
PANEL_PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
[[ -x "$PANEL_PYTHON" ]] || PANEL_PYTHON="$(command -v python3)"
unsetopt BG_NICE

"$PANEL_PYTHON" "$SCRIPT_DIR/pressure_panel_app.py" \
  --port "$SOFT_ARM_PRESSURE_PORT" &
PANEL_PID=$!

cleanup_panel() {
  kill "$PANEL_PID" 2>/dev/null || true
  wait "$PANEL_PID" 2>/dev/null || true
}
trap cleanup_panel EXIT INT TERM

echo "──────────────────────────────────────────────"
echo "  SOFA_ROOT : $SOFA_ROOT"
echo "  Scene     : $SCENE"
echo "──────────────────────────────────────────────"

"$SOFA_ROOT/bin/runSofa" \
  -l "$SOFA_ROOT/plugins/SofaPython3/lib/libSofaPython3.dylib" \
  -g imgui \
  -a \
  "$SCENE"

SOFA_STATUS=$?
cleanup_panel
trap - EXIT INT TERM
exit "$SOFA_STATUS"
