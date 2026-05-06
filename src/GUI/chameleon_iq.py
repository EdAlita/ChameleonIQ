import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QScrollArea,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

# Fix OpenCV/PyQt5 plugin conflict
# Must be set BEFORE importing PyQt5
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
if "cv2" in sys.modules:
    # If cv2 is already imported, we need a different approach
    import cv2

    cv2.ocl.setUseOpenCL(False)

COMMANDS: Dict[str, Dict[str, Any]] = {
    "Single Quant": {
        "exe": "chameleoniq_quant",
        "args": [
            {"name": "input_image", "type": "file", "required": True},
            {"name": "--output", "type": "save", "required": True},
            {"name": "--config", "type": "file", "required": True},
            {
                "name": "--standard",
                "type": "choice",
                "choices": ["NU_4_2008", "NU_2_2018", "DEDICATED_IQ"],
                "default": "NU_4_2008",
                "required": True,
            },
            {"name": "--save-visualizations", "type": "flag"},
            {"name": "--advanced-metrics", "type": "flag", "enables": "--gt-image"},
            {"name": "--gt-image", "type": "file", "requires": "--advanced-metrics"},
            {
                "name": "--log_level",
                "type": "choice",
                "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "default": "INFO",
            },
        ],
    },
    "Quant Iterations": {
        "exe": "nema_quant_iter",
        "args": [
            {"name": "input_path", "type": "file", "required": True},
            {"name": "--output", "type": "save", "required": True},
            {"name": "--config", "type": "file", "required": True},
            {
                "name": "--standard",
                "type": "choice",
                "choices": ["NU_4_2008", "NU_2_2018", "DEDICATED_IQ"],
                "default": "NU_4_2008",
                "required": True,
            },
            {"name": "--save-visualizations", "type": "flag"},
            {
                "name": "--log_level",
                "type": "choice",
                "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "default": "INFO",
            },
            {"name": "--verbose", "type": "flag", "default": True},
        ],
    },
    "Merge": {
        "exe": "chameleoniq_merge",
        "args": [
            {"name": "xml_config", "type": "file", "required": True},
            {"name": "--output", "type": "file", "required": True},
            {"name": "--config", "type": "file", "required": True},
            {
                "name": "--log-level",
                "type": "choice",
                "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "default": "INFO",
            },
        ],
    },
}


def resolve_banner_path() -> Path:
    """Return the most likely location for banner.png in different installs."""

    candidates = []

    # PyInstaller bundle
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "data" / "banner.png")

    current_dir = Path(__file__).resolve().parent
    # Source checkout: repo_root/data/banner.png
    candidates.append(current_dir.parent.parent / "data" / "banner.png")
    # Installed via data-files: <sys.prefix>/data/banner.png
    candidates.append(Path(sys.prefix) / "data" / "banner.png")

    for path in candidates:
        if path.is_file():
            return path

    return candidates[0]


class ComboPopupDelegate(QStyledItemDelegate):
    """Increase popup row height so combo items do not visually overlap."""

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(max(size.height(), 30))
        return size


class CommandLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Command Launcher - ChameleonIQ")
        self.setMinimumSize(900, 900)
        self.setMaximumSize(1200, 1200)
        self.widgets: Dict[str, Union[QLineEdit, QComboBox, QCheckBox]] = {}
        self.apply_stylesheet()
        self.build_selector()

    def apply_stylesheet(self):
        """Apply modern stylesheet to the application"""
        stylesheet = """
            QWidget {
                background-color: #f5f5f5;
                font-family: 'Segoe UI', Arial, sans-serif;
            }

            QLabel {
                color: #333333;
                font-weight: 500;
            }

            QLineEdit {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 11pt;
                selection-background-color: #4CAF50;
                selection-color: #ffffff;
            }

            QLineEdit:focus {
                border: 2px solid #4CAF50;
                background-color: #fafafa;
            }

            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11pt;
            }

            QPushButton:hover {
                background-color: #45a049;
            }

            QPushButton:pressed {
                background-color: #3d8b40;
            }

            QPushButton#browseBtn {
                background-color: #2196F3;
                padding: 6px 12px;
                font-size: 10pt;
            }

            QPushButton#browseBtn:hover {
                background-color: #0b7dda;
            }

            QCheckBox {
                color: #333333;
                spacing: 6px;
                font-size: 11pt;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }

            QCheckBox::indicator:unchecked {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
            }

            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 1px solid #4CAF50;
                border-radius: 3px;
            }

            QComboBox {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 11pt;
            }

            /* Ensure items in combo boxes keep readable selection colors */
            QComboBox QAbstractItemView {
                background-color: white;
                selection-background-color: #4CAF50;
                selection-color: #ffffff;
                color: #333333;
            }

            /* More explicit rules to ensure hovered/selected combo items stay readable */
            QComboBox QAbstractItemView::item:hover,
            QComboBox QListView::item:hover,
            QListView::item:hover,
            QTreeView::item:hover,
            QTableView::item:hover {
                background-color: #4CAF50;
                color: #ffffff;
            }

            QComboBox QAbstractItemView::item:selected,
            QComboBox QListView::item:selected,
            QListView::item:selected,
            QTreeView::item:selected,
            QTableView::item:selected {
                background-color: #4CAF50;
                color: #ffffff;
            }

            /* Selection styling for multi-line and list widgets */
            QTextEdit, QPlainTextEdit {
                background-color: white;
                color: #333333;
                selection-background-color: #4CAF50;
                selection-color: #ffffff;
            }

            QListWidget::item:selected, QListView::item:selected,
            QTreeView::item:selected, QTableView::item:selected,
            QAbstractItemView::item:selected {
                background-color: #4CAF50;
                color: #ffffff;
            }

            QComboBox:focus {
                border: 2px solid #4CAF50;
            }

            QComboBox::drop-down {
                border: none;
                width: 30px;
            }

            QComboBox::down-arrow {
                image: url(downArrow.png);
                width: 12px;
                height: 12px;
            }

            QGroupBox {
                color: #333333;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }

            QScrollArea {
                border: none;
                background-color: #f5f5f5;
            }
        """
        self.setStyleSheet(stylesheet)

    def style_combo_popup(self, combo: QComboBox) -> None:
        """Force readable hover/selection colors in the combo popup view."""

        view = QListView()
        view.setItemDelegate(ComboPopupDelegate(view))
        view.setSpacing(3)
        view.setStyleSheet(
            """
            QListView {
                background-color: white;
                color: #333333;
                outline: 0;
                padding: 4px;
            }

            QListView::item {
                min-height: 30px;
                padding: 6px 12px;
                background-color: white;
                color: #333333;
            }

            QListView::item:hover,
            QListView::item:selected {
                background-color: rgba(76,175,80,0.8);
                color: #ffffff;
            }
            """
        )
        combo.setView(view)

    def build_selector(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 0, 20, 20)

        # Banner image
        banner_label = QLabel()
        banner_path = resolve_banner_path()
        banner_pixmap = QPixmap(str(banner_path))
        if not banner_pixmap.isNull():
            scaled_pixmap = banner_pixmap.scaledToWidth(
                500, Qt.TransformationMode.SmoothTransformation
            )
            banner_label.setPixmap(scaled_pixmap)
            banner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(banner_label)

        # Header
        title = QLabel("ChameleonIQ Command Launcher")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        # Command selector
        selector_layout = QGridLayout()
        selector_layout.setSpacing(10)

        cmd_label = QLabel("Select Command:")
        cmd_label_font = QFont()
        cmd_label_font.setPointSize(11)
        cmd_label_font.setBold(True)
        cmd_label.setFont(cmd_label_font)

        self.command_box = QComboBox()
        self.command_box.addItems(COMMANDS.keys())
        self.style_combo_popup(self.command_box)
        self.command_box.currentTextChanged.connect(self.build_form)
        self.command_box.setMinimumWidth(300)

        selector_layout.addWidget(cmd_label, 0, 0)
        selector_layout.addWidget(self.command_box, 0, 1)
        main_layout.addLayout(selector_layout)

        # Form area with scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        form_container = QWidget()
        self.form_layout = QVBoxLayout()
        self.form_layout.setSpacing(12)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        form_container.setLayout(self.form_layout)

        scroll.setWidget(form_container)
        main_layout.addWidget(scroll)

        # Button layout
        button_layout = QGridLayout()
        button_layout.setSpacing(10)

        run_btn = QPushButton("Run Command")
        run_btn.setMinimumHeight(40)
        run_btn.setMinimumWidth(150)
        run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        run_btn.clicked.connect(self.run_command)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(40)
        cancel_btn.setMinimumWidth(100)
        cancel_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #757575;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """
        )
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(lambda: self.close())  # type: ignore[arg-type]

        button_layout.addWidget(run_btn, 0, 0)
        button_layout.addWidget(cancel_btn, 0, 1)
        button_layout.setColumnStretch(2, 1)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.build_form(self.command_box.currentText())

    def clear_form(self):
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.widgets.clear()

    def browse_file(self, line_edit: QLineEdit, is_save: bool, title: str) -> None:
        """Open file dialog and set the selected path in line_edit"""

        # Defer dialog opening to allow the GUI to fully render
        def open_dialog() -> None:
            # Force event processing before showing dialog
            for _ in range(10):
                QApplication.processEvents()

            if is_save:
                file_path = QFileDialog.getSaveFileName(self, title)[0]
            else:
                file_path = QFileDialog.getOpenFileName(self, title)[0]
            if file_path:
                line_edit.setText(file_path)

        # Use singleShot to avoid losing a local QTimer reference.
        QTimer.singleShot(50, open_dialog)

    def make_browse_handler(
        self, line_edit: QLineEdit, is_save: bool, title: str
    ) -> Callable[[bool], None]:
        """Return a slot for Browse button clicks."""

        def handler(_checked: bool = False) -> None:
            self.browse_file(line_edit, is_save, title)

        return handler

    def build_form(self, command_name):
        self.clear_form()
        cmd: Dict[str, Any] = COMMANDS[command_name]
        cmd_args: List[Dict[str, Any]] = cmd.get("args", [])

        # Group required and optional arguments
        required_group = QGroupBox("Required Parameters")
        required_layout = QGridLayout()
        required_layout.setSpacing(10)
        required_layout.setContentsMargins(10, 10, 10, 10)
        required_group.setLayout(required_layout)

        optional_group = QGroupBox("Optional Parameters")
        optional_layout = QGridLayout()
        optional_layout.setSpacing(10)
        optional_layout.setContentsMargins(10, 10, 10, 10)
        optional_group.setLayout(optional_layout)

        required_row = 0
        optional_row = 0

        for arg in cmd_args:
            if not isinstance(arg, dict):
                continue
            name = arg["name"]
            is_required = arg.get("required", False)
            target_layout = required_layout if is_required else optional_layout
            target_row = required_row if is_required else optional_row

            if arg["type"] == "flag":
                w = QCheckBox(name)
                w.setToolTip(arg.get("help", f"Enable {name}"))
                if arg.get("default", False):
                    w.setChecked(True)
                target_layout.addWidget(w, target_row, 0, 1, 2)
                self.widgets[name] = w
            else:
                label = QLabel(name)
                label.setMinimumWidth(120)
                target_layout.addWidget(label, target_row, 0)

                if arg["type"] in ("file", "save"):
                    le = QLineEdit()
                    le.setPlaceholderText(
                        f"Select {'output' if arg['type'] == 'save' else 'input'} file..."
                    )
                    le.setMinimumWidth(250)
                    btn = QPushButton("Browse")
                    btn.setObjectName("browseBtn")
                    btn.setMaximumWidth(80)
                    btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn.clicked.connect(
                        self.make_browse_handler(le, arg["type"] == "save", title=name)
                    )
                    target_layout.addWidget(le, target_row, 1)
                    target_layout.addWidget(btn, target_row, 2)
                    self.widgets[name] = le

                elif arg["type"] == "choice":
                    cb = QComboBox()
                    cb.addItems(arg["choices"])
                    cb.setCurrentText(arg.get("default", arg["choices"][0]))
                    self.style_combo_popup(cb)
                    cb.setMinimumWidth(250)
                    target_layout.addWidget(cb, target_row, 1)
                    self.widgets[name] = cb

                else:
                    le = QLineEdit(str(arg.get("default", "")))
                    le.setPlaceholderText(f"Enter {name.lstrip('-')}...")
                    le.setMinimumWidth(250)
                    target_layout.addWidget(le, target_row, 1)
                    self.widgets[name] = le

            if is_required:
                required_row += 1
            else:
                optional_row += 1

        # Add groups to form layout
        if required_row > 0:
            self.form_layout.addWidget(required_group)
        if optional_row > 0:
            self.form_layout.addWidget(optional_group)

        self.form_layout.addStretch()

    def run_command(self):
        cmd_def: Dict[str, Any] = COMMANDS[self.command_box.currentText()]
        cmd: List[str] = [str(cmd_def["exe"])]
        positional_args: List[str] = []
        cmd_args: List[Dict[str, Any]] = cmd_def.get("args", [])

        for arg in cmd_args:
            if not isinstance(arg, dict):
                continue
            name = str(arg.get("name", ""))
            w = self.widgets.get(name)
            if w is None:
                continue

            # Positional arguments (no -- prefix)
            if not name.startswith("--"):
                value = w.text()
                if value:
                    positional_args.append(str(value))
            # Flags
            elif arg["type"] == "flag":
                if isinstance(w, QCheckBox) and w.isChecked():
                    cmd.append(name)
            # ComboBox (choice)
            elif isinstance(w, QComboBox):
                cmd += [name, str(w.currentText())]
            # Regular arguments with -- prefix
            else:
                if hasattr(w, "text"):
                    value = w.text()
                    if value:
                        cmd += [name, str(value)]

        # Add positional arguments at the end
        cmd.extend(positional_args)

        self.close()
        subprocess.run(cmd)


def main() -> int:
    app = QApplication(sys.argv)
    win = CommandLauncher()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
