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
  vector scale 2;
  vector 0.12;
}

function setupBonds() {
    wireframe 0.1;
}

function loadSystem() {
  load "xyz::$SCRIPT_PATH$system.xyz";
  connect delete;
  script "$SCRIPT_PATH$system.spt";
}

function _setup() {
  initialize;
  set refreshing false;
  setupDisplay;
  loadSystem;
  setupVectors;
  setupBonds;
  set refreshing true;
}

_setup;
"""
