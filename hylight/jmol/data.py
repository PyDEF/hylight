from .. import __version__

manifest = f"""\
# Jmol Manifest Zip Format 1.1
# Created with Hylight {__version__}
state.spt
"""

state = """\
function setupDisplay() {
  set antialiasDisplay;

  color background white;
}

function setupVectors() {
  vector on;
  color vector yellow;
  vector scale 10;
  vector 0.07;
}

function loadSystem() {
  load "Xyz::$SCRIPT_PATH$system.xyz";
  connect delete;
  script "$SCRIPT_PATH$system.spt";
}

function _setup() {
  initialize;
  set refreshing false;
  setupDisplay;
  loadSystem;
  setupVectors;
  set refreshing true;
}

_setup;
"""
