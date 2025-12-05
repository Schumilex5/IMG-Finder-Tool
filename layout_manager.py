"""
Simple layout manager with direct pixel-size control via spinners.
No dragging - users adjust sizes with spinners.
"""
from PyQt6 import QtWidgets, QtCore, QtGui


class SimpleLayoutManager(QtWidgets.QWidget):
    """
    Three-pane vertical layout:
    - Top section (40% default)
    - Bottom section (60% default)
    - With spinners for direct pixel control.
    """

    def __init__(self, top_widget, bottom_widget, parent=None):
        super().__init__(parent)
        self.top_widget = top_widget
        self.bottom_widget = bottom_widget

        # Main vertical layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Control bar removed — size controlled via Settings page only
        self._top_height_default = 300
        self._bottom_height_default = 450

        # --- Top section ---
        main_layout.addWidget(self.top_widget, stretch=0)

        # --- Divider line (static, not draggable) ---
        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        divider.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        divider.setLineWidth(2)
        divider.setFixedHeight(2)
        main_layout.addWidget(divider)

        # --- Bottom section ---
        main_layout.addWidget(self.bottom_widget, stretch=0)

        self._apply_sizes()

    def _apply_sizes(self):
        """Apply current spinner values to widgets."""
        top_h = getattr(self, "_top_height_default", 300)
        bottom_h = getattr(self, "_bottom_height_default", 450)
        self.top_widget.setFixedHeight(top_h)
        self.bottom_widget.setFixedHeight(bottom_h)

    def _on_top_changed(self, value):
        """Update top height when spinner changes."""
        self._top_height_default = value
        self.top_widget.setFixedHeight(value)

    def _on_bottom_changed(self, value):
        """Update bottom height when spinner changes."""
        self._bottom_height_default = value
        self.bottom_widget.setFixedHeight(value)

    def get_sizes(self):
        """Return current sizes as (top_height, bottom_height)."""
        return (getattr(self, "_top_height_default", 300), getattr(self, "_bottom_height_default", 450))

    def set_sizes(self, top_height, bottom_height):
        """Set sizes from config."""
        self._top_height_default = top_height
        self._bottom_height_default = bottom_height
        self._apply_sizes()
