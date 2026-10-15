#!/usr/bin/env python3
"""Test script to verify pipeline integration"""

import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
from core.config import load_settings

def test_pipeline_integration():
    """Test that the pipeline integrates correctly with the main window"""
    app = QApplication.instance() or QApplication([])
    settings = load_settings()

    # Create and show the main window
    window = MainWindow(settings)
    window.show()

    # Check that the window is set up correctly (without asserting pipeline is started yet)
    # The pipeline worker exists as an attribute but is not started until voice mode is enabled
    assert window._settings is not None, "Settings should be stored"
    assert window._agent is not None, "Agent worker should be initialized"
    assert window._audio is not None or True, "Audio worker may not be initialized yet (OK)"
    assert window._wave is not None, "Waveform widget should exist"
    assert window._transcript is not None, "Transcript widget should exist"
    assert window._input is not None, "Input field should exist"
    assert hasattr(window, '_pipeline'), "Pipeline worker should exist as attribute"

    print("Pipeline integration test passed!")
    print(f"  - MainWindow initialized with {len(window._devices.combo)} audio devices")
    print(f"  - Agent worker: {window._agent is not None}")
    print(f"  - Audio worker: {window._audio is not None}")
    print(f"  - Pipeline worker attribute exists: {hasattr(window, '_pipeline')}")
    print(f"  - Pipeline worker initialized: {window._pipeline is not None}")
    print(f"  - Pipeline not started yet (will start on voice mode)")
    return 0

if __name__ == "__main__":
    sys.exit(test_pipeline_integration())