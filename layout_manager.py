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

        # --- Control bar ---
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_layout.setSpacing(8)

        ctrl_layout.addWidget(QtWidgets.QLabel("Top:"))
        self.spin_top = QtWidgets.QSpinBox()
        self.spin_top.setRange(50, 2000)
        self.spin_top.setValue(300)
        self.spin_top.valueChanged.connect(self._on_top_changed)
        ctrl_layout.addWidget(self.spin_top)

        ctrl_layout.addWidget(QtWidgets.QLabel("Bottom:"))
        self.spin_bottom = QtWidgets.QSpinBox()
        self.spin_bottom.setRange(50, 2000)
        self.spin_bottom.setValue(450)
        self.spin_bottom.valueChanged.connect(self._on_bottom_changed)
        ctrl_layout.addWidget(self.spin_bottom)

        ctrl_layout.addStretch(1)

        main_layout.addLayout(ctrl_layout)

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
        top_h = self.spin_top.value()
        bottom_h = self.spin_bottom.value()
        self.top_widget.setFixedHeight(top_h)
        self.bottom_widget.setFixedHeight(bottom_h)

    def _on_top_changed(self, value):
        """Update top height when spinner changes."""
        self.top_widget.setFixedHeight(value)

    def _on_bottom_changed(self, value):
        """Update bottom height when spinner changes."""
        self.bottom_widget.setFixedHeight(value)

    def get_sizes(self):
        """Return current sizes as (top_height, bottom_height)."""
        return (self.spin_top.value(), self.spin_bottom.value())

    def set_sizes(self, top_height, bottom_height):
        """Set sizes from config."""
        self.spin_top.blockSignals(True)
        self.spin_bottom.blockSignals(True)

        self.spin_top.setValue(top_height)
        self.spin_bottom.setValue(bottom_height)

        self.spin_top.blockSignals(False)
        self.spin_bottom.blockSignals(False)

        self._apply_sizes()
