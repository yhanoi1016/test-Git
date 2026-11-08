import os
import sys
import json
import math
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import pydicom

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QSlider, QListWidget, QListWidgetItem, QGroupBox,
    QLineEdit, QAbstractItemView
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


OUTPUT_COLUMNS = [
    "patient_id", "x_pixel", "y_pixel", "z_pixel",
    "x_mm", "y_mm", "z_mm", "series_uid", "sop_instance_uid"
]
STATUS_COLUMNS = ["patient_id", "status", "annotation_count", "saved_at", "note"]


def as_patient_id(value):
    """Convert Excel cell values to a stable string without adding '.0'."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        # Excel may already have stripped leading zeros if the cell was numeric.
        # Format the patient_id column as Text in Excel to preserve leading zeros.
        return str(int(value))
    return str(value).strip()


def is_excluded_dwi(text):
    t = (text or "").lower().replace(" ", "").replace("_", "").replace("-", "")
    return "dwi" in t or "diffusion" in t or "adc" in t


def is_swi_gre_description(text):
    t = (text or "").lower()
    if is_excluded_dwi(t):
        return False
    terms = ("swi", "swan", "t2*", "t2star", "gre", "susceptibility", "minip")
    return any(term in t for term in terms)


def dicom_value(ds, name, default=""):
    value = getattr(ds, name, default)
    try:
        return str(value)
    except Exception:
        return default


def safe_float_list(value, expected_len):
    try:
        values = [float(x) for x in value]
        if len(values) == expected_len:
            return values
    except Exception:
        pass
    return None


def load_dicom_series(patient_folder):
    """
    Read DICOM files directly inside one patient folder and group them by
    SeriesInstanceUID. Each returned item represents one real DICOM series.
    """
    if not os.path.isdir(patient_folder):
        return []

    grouped = {}

    for filename in os.listdir(patient_folder):
        path = os.path.join(patient_folder, filename)
        if not os.path.isfile(path):
            continue

        try:
            ds = pydicom.dcmread(path, force=True)
            if not hasattr(ds, "PixelData"):
                continue

            pixels = ds.pixel_array.astype(np.float32)
            if pixels.ndim != 2:
                continue

            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            pixels = pixels * slope + intercept

            ipp = safe_float_list(getattr(ds, "ImagePositionPatient", None), 3)
            iop = safe_float_list(getattr(ds, "ImageOrientationPatient", None), 6)
            spacing = safe_float_list(getattr(ds, "PixelSpacing", None), 2)

            if iop is not None:
                row_dir = np.asarray(iop[:3], dtype=float)
                col_dir = np.asarray(iop[3:], dtype=float)
                normal = np.cross(row_dir, col_dir)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                else:
                    normal = np.array([0.0, 0.0, 1.0])
            else:
                normal = np.array([0.0, 0.0, 1.0])

            instance = int(getattr(ds, "InstanceNumber", 0) or 0)
            if ipp is not None:
                position_scalar = float(np.dot(np.asarray(ipp, dtype=float), normal))
            else:
                position_scalar = float(instance)
                ipp = [0.0, 0.0, position_scalar]

            if iop is None:
                iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            if spacing is None:
                spacing = [1.0, 1.0]

            uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            if not uid:
                # Avoid accidentally combining unrelated images with missing UIDs.
                uid = "NO_SERIES_UID_" + os.path.basename(path)

            description = (
                str(getattr(ds, "SeriesDescription", "") or "").strip()
                or str(getattr(ds, "ProtocolName", "") or "").strip()
                or "DICOM series"
            )
            protocol = str(getattr(ds, "ProtocolName", "") or "").strip()

            if uid not in grouped:
                grouped[uid] = {
                    "description": description,
                    "protocol": protocol,
                    "uid": uid,
                    "slices": []
                }

            grouped[uid]["slices"].append({
                "path": path,
                "pixels": pixels,
                "ipp": np.asarray(ipp, dtype=float),
                "iop": np.asarray(iop, dtype=float),
                "spacing": np.asarray(spacing, dtype=float),
                "normal": np.asarray(normal, dtype=float),
                "position_scalar": position_scalar,
                "sop_uid": str(getattr(ds, "SOPInstanceUID", "") or ""),
                "rows": int(getattr(ds, "Rows", pixels.shape[0])),
                "columns": int(getattr(ds, "Columns", pixels.shape[1])),
                "instance": instance,
                "photometric": str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
            })
        except Exception as exc:
            print("Skipping file {}: {}".format(path, exc))

    series_list = list(grouped.values())
    for series in series_list:
        series["slices"].sort(
            key=lambda s: (s["position_scalar"], s["instance"])
        )

    series_list.sort(key=lambda s: (s["description"].lower(), s["uid"]))
    return series_list


class MRIAnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI SWI/GRE Annotation Tool")
        self.resize(1250, 850)

        self.patient_df = None
        self.patients = []
        self.patient_index = -1
        self.swi_root_folder = ""
        self.tof_root_folder = ""

        self.swi_series_list = []
        self.tof_series_list = []
        self.output_path = os.path.abspath("output.xlsx")
        self.checkpoint_path = os.path.abspath("annotation_checkpoint.json")

        self.current_series = None       # SWI/GRE series used for annotations
        self.current_tof_series = None   # ToF series displayed alongside SWI/GRE
        self.current_slice_index = 0
        self.current_tof_slice_index = 0
        self.slider_steps = 1000         # shared normalized slider, independent of slice counts
        self.pending_points = []
        self.patient_saved_points = []
        self.patient_saved_status = None
        self.click_candidate = None
        self._loading = False

        self._build_ui()
        # self._load_checkpoint_if_available()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)

        top = QHBoxLayout()
        self.open_excel_btn = QPushButton("Open Excel")
        self.open_swi_folder_btn = QPushButton("Open SWI/GRE Folder")
        self.open_tof_folder_btn = QPushButton("Open ToF Folder")
        self.output_edit = QLineEdit(self.output_path)
        self.output_edit.setMinimumWidth(250)
        self.output_btn = QPushButton("Choose Output")
        top.addWidget(self.open_excel_btn)
        top.addWidget(self.open_swi_folder_btn)
        top.addWidget(self.open_tof_folder_btn)
        top.addWidget(QLabel("Output:"))
        top.addWidget(self.output_edit, 1)
        top.addWidget(self.output_btn)
        main.addLayout(top)

        patient_row = QHBoxLayout()
        self.patient_label = QLabel("Patient: (not loaded)")
        self.patient_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.prev_btn = QPushButton("Previous Patient")
        self.next_btn = QPushButton("Next Patient")
        self.patient_combo = QComboBox()
        self.patient_combo.setMinimumWidth(220)
        patient_row.addWidget(self.patient_label, 1)
        patient_row.addWidget(self.patient_combo)
        patient_row.addWidget(self.prev_btn)
        patient_row.addWidget(self.next_btn)
        main.addLayout(patient_row)

        body = QHBoxLayout()
        main.addLayout(body, 1)

        left = QVBoxLayout()
        body.addLayout(left, 1)

        series_row = QHBoxLayout()
        series_row.addWidget(QLabel("SWI/GRE:"))
        self.series_combo = QComboBox()
        self.series_combo.setMinimumWidth(220)
        series_row.addWidget(self.series_combo, 1)
        series_row.addWidget(QLabel("ToF:"))
        self.tof_series_combo = QComboBox()
        self.tof_series_combo.setMinimumWidth(220)
        series_row.addWidget(self.tof_series_combo, 1)
        left.addLayout(series_row)

        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(121)       # SWI/GRE; annotation clicks are here
        self.tof_ax = self.figure.add_subplot(122)   # ToF; display only
        self.ax.axis("off")
        self.tof_ax.axis("off")
        left.addWidget(self.canvas, 1)

        slice_row = QHBoxLayout()
        self.slice_label = QLabel("SWI/GRE: - | ToF: -")
        left.addWidget(self.slice_label)

        # Three sliders: independent SWI/GRE, independent ToF, and common (%).
        slider_controls = QVBoxLayout()

        swi_slider_row = QHBoxLayout()
        swi_slider_row.addWidget(QLabel("SWI/GRE slice:"))
        self.swi_slider = QSlider(Qt.Horizontal)
        self.swi_slider.setRange(0, 0)
        self.swi_slider.setEnabled(False)
        swi_slider_row.addWidget(self.swi_slider, 1)
        slider_controls.addLayout(swi_slider_row)

        tof_slider_row = QHBoxLayout()
        tof_slider_row.addWidget(QLabel("ToF slice:"))
        self.tof_slider = QSlider(Qt.Horizontal)
        self.tof_slider.setRange(0, 0)
        self.tof_slider.setEnabled(False)
        tof_slider_row.addWidget(self.tof_slider, 1)
        slider_controls.addLayout(tof_slider_row)

        common_slider_row = QHBoxLayout()
        common_slider_row.addWidget(QLabel("Common position (%):"))
        self.common_slider = QSlider(Qt.Horizontal)
        self.common_slider.setRange(0, self.slider_steps)
        self.common_slider.setEnabled(False)
        common_slider_row.addWidget(self.common_slider, 1)
        slider_controls.addLayout(common_slider_row)

        left.addLayout(slider_controls)

        right_box = QGroupBox("Annotations for current patient (not saved yet)")
        right = QVBoxLayout(right_box)
        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QAbstractItemView.SingleSelection)
        right.addWidget(self.annotation_list, 1)
        annotation_buttons = QHBoxLayout()
        self.delete_btn = QPushButton("Delete Selected Point")
        self.clear_btn = QPushButton("Clear All Points")
        annotation_buttons.addWidget(self.delete_btn)
        annotation_buttons.addWidget(self.clear_btn)
        right.addLayout(annotation_buttons)

        self.instructions = QLabel(
            "Click the SWI/GRE image to add a point. Confirm each point in the popup. "
            "Use the SWI/GRE slider or Left/Right arrow keys for SWI/GRE; use the "
            "ToF slider for ToF, or Common position (%) to move both."
        )
        self.instructions.setWordWrap(True)
        right.addWidget(self.instructions)

        self.save_patient_btn = QPushButton("Save Patient")
        self.save_patient_btn.setMinimumHeight(42)
        right.addWidget(self.save_patient_btn)
        body.addWidget(right_box, 0)

        self.statusBar().showMessage("Open an Excel patient list to begin.")

        self.open_excel_btn.clicked.connect(self.open_excel)
        self.open_swi_folder_btn.clicked.connect(self.open_swi_folder)
        self.open_tof_folder_btn.clicked.connect(self.open_tof_folder)
        self.output_btn.clicked.connect(self.choose_output)
        self.prev_btn.clicked.connect(lambda: self.change_patient(-1))
        self.next_btn.clicked.connect(lambda: self.change_patient(1))
        self.patient_combo.currentIndexChanged.connect(self.on_patient_combo_changed)
        self.series_combo.currentIndexChanged.connect(self.on_series_changed)
        self.tof_series_combo.currentIndexChanged.connect(self.on_tof_series_changed)
        self.swi_slider.valueChanged.connect(self.on_swi_slice_changed)
        self.tof_slider.valueChanged.connect(self.on_tof_slice_changed)
        self.common_slider.valueChanged.connect(self.on_common_slice_changed)
        self.delete_btn.clicked.connect(self.delete_selected_point)
        self.clear_btn.clicked.connect(self.clear_points)
        self.save_patient_btn.clicked.connect(self.save_patient)
        self.canvas.mpl_connect("button_press_event", self.on_image_click)

    def showEvent(self, event):
        super().showEvent(event)
        self.showMaximized()
        
    def open_excel(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select patient Excel file", "", "Excel files (*.xlsx *.xls)"
        )
        if not path:
            return
        try:
            df = pd.read_excel(path, dtype={"patient_id": str})
            df.columns = [str(c).strip() for c in df.columns]
            if "patient_id" not in df.columns:
                raise ValueError("Excel must contain a 'patient_id' column.")
            if "data_path" not in df.columns:
                df["data_path"] = ""
            df["patient_id"] = df["patient_id"].apply(as_patient_id)
            df["data_path"] = df["data_path"].fillna("").astype(str).str.strip()
            df = df[df["patient_id"] != ""].reset_index(drop=True)
            if df.empty:
                raise ValueError("No valid patient IDs were found.")
            self.patient_df = df
            self.patients = df.to_dict("records")
            self.patient_index = -1
            self.patient_combo.blockSignals(True)
            self.patient_combo.clear()
            for row in self.patients:
                self.patient_combo.addItem(str(row["patient_id"]))
            self.patient_combo.blockSignals(False)
            self.patient_combo.setCurrentIndex(0)
            self.patient_index = 0
            self.load_patient()
            self.statusBar().showMessage("Loaded {} patients from {}".format(len(self.patients), path))
        except Exception as exc:
            QMessageBox.critical(self, "Excel error", str(exc))

    def open_swi_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select SWI/GRE root folder"
        )
        if not path:
            return

        self.swi_root_folder = path
        self.statusBar().showMessage("SWI/GRE root: " + path)

        if self.patient_index >= 0:
            self.load_patient()


    def open_tof_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select ToF root folder"
        )
        if not path:
            return

        self.tof_root_folder = path
        self.statusBar().showMessage("ToF root: " + path)

        if self.patient_index >= 0:
            self.load_patient()

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save output Excel", self.output_edit.text(), "Excel files (*.xlsx)"
        )
        if path:
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            self.output_path = path
            self.output_edit.setText(path)

    def patient_folder_for_current(self, modality):
        if self.patient_index < 0 or self.patient_index >= len(self.patients):
            return ""

        pid = str(self.patients[self.patient_index]["patient_id"])

        if modality == "swi":
            base = self.swi_root_folder
        elif modality == "tof":
            base = self.tof_root_folder
        else:
            return ""

        if not base:
            return ""

        return os.path.join(base, pid)

    def load_patient(self):
        if self.patient_index < 0 or self.patient_index >= len(self.patients):
            return
        self._loading = True
        row = self.patients[self.patient_index]
        pid = str(row["patient_id"])
        self.patient_label.setText("Patient ID: {}".format(pid))
        self.pending_points = []
        self.patient_saved_points = []
        self.patient_saved_status = None
        self._load_saved_points_for_patient(pid)
        self._restore_pending_for_patient(pid)
        self._refresh_annotation_list()

        swi_folder = self.patient_folder_for_current("swi")
        tof_folder = self.patient_folder_for_current("tof")

        self.swi_series_list = []
        self.tof_series_list = []
        self.current_series = None
        self.current_tof_series = None
        self.current_series = None
        self.current_tof_series = None
        self.series_combo.blockSignals(True)
        self.tof_series_combo.blockSignals(True)
        self.series_combo.clear()
        self.tof_series_combo.clear()
        self.series_combo.blockSignals(False)
        self.tof_series_combo.blockSignals(False)
        self.ax.clear()
        self.tof_ax.clear()
        self.ax.axis("off")
        self.tof_ax.axis("off")
        self.canvas.draw_idle()

        if not swi_folder or not os.path.isdir(swi_folder) or not tof_folder or not os.path.isdir(tof_folder):
            self.slice_label.setText("Slice: -")
            self._set_sliders_enabled(False)
            self.statusBar().showMessage("Patient folder not found: " + str(swi_folder))
            self._loading = False
            return


        self.statusBar().showMessage(
            "Loading SWI/GRE and ToF series..."
        )

        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            if swi_folder and os.path.isdir(swi_folder):
                self.swi_series_list = load_dicom_series(swi_folder)

            if tof_folder and os.path.isdir(tof_folder):
                self.tof_series_list = load_dicom_series(tof_folder)

        except Exception as exc:
            QMessageBox.warning(self, "DICOM error", str(exc))

        finally:
            QApplication.restoreOverrideCursor()

        self.series_combo.blockSignals(True)
        self.tof_series_combo.blockSignals(True)
        for series_idx, series in enumerate(self.swi_series_list):
            label = "{} ({} slices)".format(
                series["description"], len(series["slices"])
            )
            self.series_combo.addItem(label, series_idx)
        for series_idx, series in enumerate(self.tof_series_list):
            label = "{} ({} slices)".format(
                series["description"],
                len(series["slices"])
            )
            self.tof_series_combo.addItem(label, series_idx)
        self.series_combo.blockSignals(False)
        self.tof_series_combo.blockSignals(False)

        if not self.swi_series_list:
            self.slice_label.setText("SWI/GRE: - | ToF: -")
            self._set_sliders_enabled(False)
            self.statusBar().showMessage("No readable DICOM series found for patient " + pid)
        else:
            # Best-effort defaults based on series descriptions. The user can
            # change either selection manually if the scanner uses other names.
            swi_idx = self._find_series_index(("swi", "swan", "gre", "t2*", "t2star", "susceptibility", "minip"))
            tof_idx = self._find_series_index(("tof", "3d tof", "time of flight", "angio", "mra"))
            if swi_idx is None:
                swi_idx = 0
            if tof_idx is None:
                tof_idx = 0 # if len(self.swi_series_list) > 1 and swi_idx == 0 else 0
            self.series_combo.setCurrentIndex(swi_idx)
            self.tof_series_combo.setCurrentIndex(tof_idx)
            self.set_current_series(swi_idx)
            self.set_current_tof_series(tof_idx)
            self.common_slider.setValue(0)
            self.statusBar().showMessage(
                "Patient {}: loaded {} DICOM series. Confirm SWI/GRE and ToF selections.".format(
                    pid, len(self.swi_series_list)
                )
            )
        self._loading = False
        self.draw_slice()
        self.canvas.draw()
        QApplication.processEvents()

    def on_patient_combo_changed(self, index):
        if self._loading or index < 0 or index >= len(self.patients):
            return
        if self.patient_index >= 0 and self.pending_points:
            answer = QMessageBox.question(
                self, "Unsaved annotations",
                "There are unsaved points for the current patient. Save this patient before switching?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            if answer == QMessageBox.Cancel:
                self.patient_combo.blockSignals(True)
                self.patient_combo.setCurrentIndex(self.patient_index)
                self.patient_combo.blockSignals(False)
                return
            if answer == QMessageBox.Save:
                if not self.save_patient():
                    self.patient_combo.blockSignals(True)
                    self.patient_combo.setCurrentIndex(self.patient_index)
                    self.patient_combo.blockSignals(False)
                    return
        self.patient_index = index
        self.load_patient()

    def change_patient(self, delta):
        new_index = self.patient_index + delta
        if new_index < 0 or new_index >= len(self.patients):
            return
        self.patient_combo.setCurrentIndex(new_index)

    def _find_series_index(self, terms):
        """Return first series whose description/protocol matches one of terms."""
        for i, series in enumerate(self.swi_series_list):
            text = "{} {}".format(
                series.get("description", ""), series.get("protocol", "")
            ).lower()
            if any(term in text for term in terms):
                return i
        return None

    def on_series_changed(self, index):
        if self._loading or index < 0:
            return
        self.set_current_series(index)
        self.draw_slice()

    def on_tof_series_changed(self, index):
        if self._loading or index < 0:
            return
        self.set_current_tof_series(index)
        self.draw_slice()

    def _set_sliders_enabled(self, enabled):
        self.swi_slider.setEnabled(bool(enabled and self.current_series is not None))
        self.tof_slider.setEnabled(bool(enabled and self.current_tof_series is not None))
        self.common_slider.setEnabled(bool(
            enabled and (self.current_series is not None or self.current_tof_series is not None)
        ))

    def set_current_series(self, index):
        if index < 0 or index >= len(self.swi_series_list):
            self.current_series = None
            self.current_slice_index = 0
            self.swi_slider.blockSignals(True)
            self.swi_slider.setRange(0, 0)
            self.swi_slider.setValue(0)
            self.swi_slider.blockSignals(False)
            self._set_sliders_enabled(True)
            return
        self.current_series = self.swi_series_list[index]
        self.current_slice_index = 0
        count = len(self.current_series.get("slices", []))
        self.swi_slider.blockSignals(True)
        self.swi_slider.setRange(0, max(0, count - 1))
        self.swi_slider.setValue(0)
        self.swi_slider.blockSignals(False)
        self._set_sliders_enabled(True)

    def set_current_tof_series(self, index):
        if index < 0 or index >= len(self.tof_series_list):
            self.current_tof_series = None
            self.current_tof_slice_index = 0
            self.tof_slider.blockSignals(True)
            self.tof_slider.setRange(0, 0)
            self.tof_slider.setValue(0)
            self.tof_slider.blockSignals(False)
            self._set_sliders_enabled(True)
            return
        self.current_tof_series = self.tof_series_list[index]
        self.current_tof_slice_index = 0
        count = len(self.current_tof_series.get("slices", []))
        self.tof_slider.blockSignals(True)
        self.tof_slider.setRange(0, max(0, count - 1))
        self.tof_slider.setValue(0)
        self.tof_slider.blockSignals(False)
        self._set_sliders_enabled(True)

    def _slice_index_for_slider(self, series, slider_value):
        if not series or not series.get("slices"):
            return 0
        count = len(series["slices"])
        if count <= 1:
            return 0
        ratio = float(slider_value) / float(max(1, self.slider_steps))
        return int(round(ratio * (count - 1)))

    def on_common_slice_changed(self, value):
        """Move both modalities to the same relative (%) position."""
        self.current_slice_index = self._slice_index_for_slider(
            self.current_series, value
        )
        self.current_tof_slice_index = self._slice_index_for_slider(
            self.current_tof_series, value
        )

        # Update individual sliders without triggering their independent handlers.
        self.swi_slider.blockSignals(True)
        self.swi_slider.setValue(self.current_slice_index)
        self.swi_slider.blockSignals(False)
        self.tof_slider.blockSignals(True)
        self.tof_slider.setValue(self.current_tof_slice_index)
        self.tof_slider.blockSignals(False)
        self.draw_slice()

    def on_swi_slice_changed(self, value):
        """Move SWI/GRE only; leave ToF and common percentage unchanged."""
        self.current_slice_index = int(value)
        self.draw_slice()

    def on_tof_slice_changed(self, value):
        """Move ToF only; leave SWI/GRE and common percentage unchanged."""
        self.current_tof_slice_index = int(value)
        self.draw_slice()

    def _draw_one_series(self, ax, series, idx, title_prefix, show_points=False):
        ax.clear()
        if not series or not series.get("slices"):
            ax.text(0.5, 0.5, "No series selected", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            return

        slices = series["slices"]
        idx = max(0, min(int(idx), len(slices) - 1))
        sl = slices[idx]
        image = sl["pixels"]
        cmap = "gray_r" if sl["photometric"] == "MONOCHROME1" else "gray"
        ax.imshow(image, cmap=cmap, interpolation="nearest")
        ax.set_xlim(-0.5, image.shape[1] - 0.5)
        ax.set_ylim(image.shape[0] - 0.5, -0.5)
        ax.set_title("{}: {} | Slice {}/{}".format(
            title_prefix, series["description"], idx, len(slices) - 1
        ), fontsize=9)
        ax.axis("off")

        if show_points:
            for point in self.pending_points:
                if (point.get("series_uid", "") == series["uid"]
                        and int(point.get("z_pixel", -1)) == idx):
                    ax.plot(point["x_pixel"], point["y_pixel"], "r+",
                            markersize=12, mew=2)
                    ax.text(point["x_pixel"] + 4, point["y_pixel"] + 4,
                            str(point.get("point_number", "")),
                            color="yellow", fontsize=9)

    def draw_slice(self):
        if not self.current_series and not self.current_tof_series:
            self.ax.clear()
            self.tof_ax.clear()
            self.ax.axis("off")
            self.tof_ax.axis("off")
            self.canvas.draw_idle()
            return

        self._draw_one_series(
            self.ax, self.current_series, self.current_slice_index,
            "SWI/GRE", show_points=True
        )
        self._draw_one_series(
            self.tof_ax, self.current_tof_series, self.current_tof_slice_index,
            "ToF", show_points=False
        )

        swi_count = len(self.current_series["slices"]) if self.current_series else 0
        tof_count = len(self.current_tof_series["slices"]) if self.current_tof_series else 0
        self.slice_label.setText(
            "SWI/GRE: {} / {} | ToF: {} / {}".format(
                self.current_slice_index, max(0, swi_count - 1),
                self.current_tof_slice_index, max(0, tof_count - 1)
            )
        )
        self.canvas.draw_idle()

    def current_patient_id(self):
        if 0 <= self.patient_index < len(self.patients):
            return str(self.patients[self.patient_index]["patient_id"])
        return ""

    def on_image_click(self, event):
        if event.inaxes != self.ax or self.current_series is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        sl = self.current_series["slices"][self.current_slice_index]
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if x < 0 or y < 0 or x >= sl["columns"] or y >= sl["rows"]:
            return

        pid = self.current_patient_id()
        answer = QMessageBox.question(
            self, "Confirm annotation",
            "Bạn chắc chắn muốn chọn vị trí này cho bệnh nhân {}?\n\n"
            "Series: {}\nSlice: {}\nPixel: x={}, y={}".format(
                pid, self.current_series["description"], self.current_slice_index, x, y
            ),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Yes
        )
        if answer != QMessageBox.Yes:
            return

        # DICOM patient coordinate: IPP + x*column spacing*first IOP vector
        #                    + y*row spacing*second IOP vector.
        ipp = sl["ipp"]
        iop = sl["iop"]
        spacing = sl["spacing"]
        xyz = ipp + x * spacing[1] * iop[:3] + y * spacing[0] * iop[3:]
        point = {
            "patient_id": pid,
            "x_pixel": x,
            "y_pixel": y,
            "z_pixel": int(self.current_slice_index),
            "x_mm": round(float(xyz[0]), 4),
            "y_mm": round(float(xyz[1]), 4),
            "z_mm": round(float(xyz[2]), 4),
            "series_uid": self.current_series["uid"],
            "sop_instance_uid": sl["sop_uid"],
            "point_number": len(self.pending_points) + 1
        }
        self.pending_points.append(point)
        self._renumber_points()
        self._refresh_annotation_list()
        self._save_checkpoint()
        self.draw_slice()

    def _renumber_points(self):
        for i, point in enumerate(self.pending_points, start=1):
            point["point_number"] = i

    def _refresh_annotation_list(self):
        self.annotation_list.clear()
        for i, p in enumerate(self.pending_points):
            series_name = p.get("series_description", "")
            text = "#{:03d} | slice {} | x={}, y={} | ({:.2f}, {:.2f}, {:.2f}) mm".format(
                i + 1, p["z_pixel"], p["x_pixel"], p["y_pixel"],
                p["x_mm"], p["y_mm"], p["z_mm"]
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, i)
            self.annotation_list.addItem(item)
        self.instructions.setText(
            "{} point(s) pending. Click the image to add more points. "
            "Save Patient commits all points for this patient.".format(len(self.pending_points))
        )

    def delete_selected_point(self):
        item = self.annotation_list.currentItem()
        if item is None:
            return
        index = item.data(Qt.UserRole)
        if index is not None and 0 <= int(index) < len(self.pending_points):
            del self.pending_points[int(index)]
            self._renumber_points()
            self._refresh_annotation_list()
            self._save_checkpoint()
            self.draw_slice()

    def clear_points(self):
        if not self.pending_points:
            return
        answer = QMessageBox.question(
            self, "Clear points",
            "Delete all unsaved annotation points for patient {}?".format(self.current_patient_id()),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel
        )
        if answer == QMessageBox.Yes:
            self.pending_points = []
            self._refresh_annotation_list()
            self._save_checkpoint()
            self.draw_slice()

    # def _load_saved_points_for_patient(self, pid):
    #     if not os.path.exists(self.output_path):
    #         return
    #     try:
    #         xls = pd.ExcelFile(self.output_path)
    #         if "annotations" in xls.sheet_names:
    #             df = pd.read_excel(self.output_path, sheet_name="annotations", dtype={"patient_id": str})
    #             if "patient_id" in df.columns:
    #                 df["patient_id"] = df["patient_id"].apply(as_patient_id)
    #                 self.patient_saved_points = df[df["patient_id"] == pid].to_dict("records")
    #         if "patient_status" in xls.sheet_names:
    #             st = pd.read_excel(self.output_path, sheet_name="patient_status", dtype={"patient_id": str})
    #             if "patient_id" in st.columns:
    #                 st["patient_id"] = st["patient_id"].apply(as_patient_id)
    #                 rows = st[st["patient_id"] == pid]
    #                 if not rows.empty:
    #                     self.patient_saved_status = rows.iloc[-1].to_dict()
    #     except Exception:
    #         # A missing or older output file should not prevent annotation.
    #         self.patient_saved_points = []
    #         self.patient_saved_status = None

    def _load_saved_points_for_patient(self, pid):
        """
        Load this patient's previously saved points from output.xlsx into
        pending_points so the user can review/add/delete points and save again.
        """
        self.patient_saved_points = []
        self.patient_saved_status = None

        if not os.path.exists(self.output_path):
            self.pending_points = []
            return

        try:
            # Use a context manager so the workbook is closed before any later
            # attempt to replace output.xlsx on Windows.
            with pd.ExcelFile(self.output_path) as xls:
                if "annotations" in xls.sheet_names:
                    df = pd.read_excel(
                        xls,
                        sheet_name="annotations",
                        dtype={"patient_id": str}
                    )
                else:
                    df = pd.DataFrame(columns=OUTPUT_COLUMNS)

                if "patient_status" in xls.sheet_names:
                    st = pd.read_excel(
                        xls,
                        sheet_name="patient_status",
                        dtype={"patient_id": str}
                    )
                else:
                    st = pd.DataFrame(columns=STATUS_COLUMNS)

            if "patient_id" in df.columns:
                df["patient_id"] = df["patient_id"].apply(as_patient_id)
                saved_df = df[df["patient_id"] == pid].copy()
            else:
                saved_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

            self.patient_saved_points = saved_df.to_dict("records")

            # Rehydrate the saved Excel rows into the same point structure used
            # for new clicks. This makes existing points visible and editable.
            loaded_points = []
            for row in self.patient_saved_points:
                def _as_int(value, default=0):
                    try:
                        if pd.isna(value):
                            return default
                        return int(round(float(value)))
                    except (TypeError, ValueError):
                        return default

                def _as_float(value, default=0.0):
                    try:
                        if pd.isna(value):
                            return default
                        return float(value)
                    except (TypeError, ValueError):
                        return default

                loaded_points.append({
                    "patient_id": pid,
                    "x_pixel": _as_int(row.get("x_pixel")),
                    "y_pixel": _as_int(row.get("y_pixel")),
                    "z_pixel": _as_int(row.get("z_pixel")),
                    "x_mm": _as_float(row.get("x_mm")),
                    "y_mm": _as_float(row.get("y_mm")),
                    "z_mm": _as_float(row.get("z_mm")),
                    "series_uid": "" if pd.isna(row.get("series_uid", "")) else str(row.get("series_uid", "")),
                    "sop_instance_uid": "" if pd.isna(row.get("sop_instance_uid", "")) else str(row.get("sop_instance_uid", "")),
                })

            self.pending_points = loaded_points
            self._renumber_points()

            if "patient_id" in st.columns:
                st["patient_id"] = st["patient_id"].apply(as_patient_id)
                rows = st[st["patient_id"] == pid]
                if not rows.empty:
                    self.patient_saved_status = rows.iloc[-1].to_dict()

        except Exception as exc:
            # Keep the app usable, but surface the problem instead of silently
            # making the patient's previously saved points appear to vanish.
            self.patient_saved_points = []
            self.patient_saved_status = None
            self.pending_points = []
            QMessageBox.warning(
                self,
                "Load saved annotations",
                "Could not load previously saved points for patient {}.\\\\n\\\\n{}".format(
                    pid, exc
                )
            )
            
    def _restore_pending_for_patient(self, pid):
        if not os.path.exists(self.checkpoint_path):
            return
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            points = data.get(pid, [])
            if points:
                answer = QMessageBox.question(
                    self, "Restore checkpoint",
                    "Found {} unsaved point(s) for patient {}. Restore them?".format(len(points), pid),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if answer == QMessageBox.Yes:
                    self.pending_points = points
                    self._renumber_points()
        except Exception:
            pass

    def _save_checkpoint(self):
        try:
            data = {}
            if os.path.exists(self.checkpoint_path):
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            pid = self.current_patient_id()
            if pid:
                data[pid] = self.pending_points
            temp_path = self.checkpoint_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.checkpoint_path)
        except Exception as exc:
            self.statusBar().showMessage("Checkpoint warning: {}".format(exc))

    def _remove_checkpoint_for_patient(self, pid):
        try:
            if not os.path.exists(self.checkpoint_path):
                return
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.pop(pid, None)
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def save_patient(self):
        pid = self.current_patient_id()
        if not pid:
            return False
        output = self.output_edit.text().strip() or os.path.abspath("output.xlsx")
        if not output.lower().endswith(".xlsx"):
            output += ".xlsx"
        self.output_path = output

        answer = QMessageBox.question(
            self, "Save patient",
            "Save {} point(s) for patient {}?\n"
            "If this patient already has saved annotations, they will be replaced.".format(
                len(self.pending_points), pid
            ),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Yes
        )
        if answer != QMessageBox.Yes:
            return False

        try:
            if os.path.exists(output):
                try:
                    with pd.ExcelFile(output) as old_xls:
                        sheet_names = old_xls.sheet_names

                        if "annotations" in sheet_names:
                            annotations = pd.read_excel(
                                old_xls,
                                sheet_name="annotations",
                                dtype={"patient_id": str}
                            )
                        else:
                            annotations = pd.DataFrame(columns=OUTPUT_COLUMNS)

                        if "patient_status" in sheet_names:
                            statuses = pd.read_excel(
                                old_xls,
                                sheet_name="patient_status",
                                dtype={"patient_id": str}
                            )
                        else:
                            statuses = pd.DataFrame(columns=STATUS_COLUMNS)

                except Exception as exc:
                    raise ValueError(
                        "Cannot read the existing output.xlsx. "
                        "Details: {}".format(exc)
                    )
            else:
                annotations = pd.DataFrame(columns=OUTPUT_COLUMNS)
                statuses = pd.DataFrame(columns=STATUS_COLUMNS)

            if "patient_id" not in annotations.columns:
                annotations = pd.DataFrame(columns=OUTPUT_COLUMNS)
            annotations["patient_id"] = annotations["patient_id"].apply(as_patient_id)
            # Replace only this patient's rows; preserve every other patient.
            annotations = annotations[annotations["patient_id"] != pid].copy()
            new_rows = []
            for p in self.pending_points:
                new_rows.append({col: p.get(col, "") for col in OUTPUT_COLUMNS})
            if new_rows:
                annotations = pd.concat([annotations, pd.DataFrame(new_rows, columns=OUTPUT_COLUMNS)],
                                        ignore_index=True)
            annotations = annotations.reindex(columns=OUTPUT_COLUMNS)

            if "patient_id" not in statuses.columns:
                statuses = pd.DataFrame(columns=STATUS_COLUMNS)
            statuses["patient_id"] = statuses["patient_id"].apply(as_patient_id)
            statuses = statuses[statuses["patient_id"] != pid].copy()
            status = "annotated" if self.pending_points else "no_points"
            status_row = {
                "patient_id": pid,
                "status": status,
                "annotation_count": len(self.pending_points),
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Explicitly saved by user"
            }
            statuses = pd.concat([statuses, pd.DataFrame([status_row])], ignore_index=True)
            statuses = statuses.reindex(columns=STATUS_COLUMNS)

            output_dir = os.path.dirname(os.path.abspath(output))
            if output_dir and not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            temp_output = output + ".tmp.xlsx"
            with pd.ExcelWriter(temp_output, engine="openpyxl") as writer:
                annotations.to_excel(writer, sheet_name="annotations", index=False)
                statuses.to_excel(writer, sheet_name="patient_status", index=False)
            os.replace(temp_output, output)

            self.output_path = output
            self.output_edit.setText(output)
            self.patient_saved_points = list(self.pending_points)
            self.patient_saved_status = status_row
            self._remove_checkpoint_for_patient(pid)
            self.statusBar().showMessage(
                "Saved patient {}: {} point(s), status '{}'".format(pid, len(self.pending_points), status)
            )
            QMessageBox.information(
                self, "Saved",
                "Patient {} saved successfully.\nPoints: {}\nStatus: {}\nFile: {}".format(
                    pid, len(self.pending_points), status, output
                )
            )
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save error", "{}\n\n{}".format(exc, traceback.format_exc()))
            return False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left and self.current_series:
            self.swi_slider.setValue(max(0, self.swi_slider.value() - 1))
            return
        if event.key() == Qt.Key_Right and self.current_series:
            self.swi_slider.setValue(min(self.swi_slider.maximum(), self.swi_slider.value() + 1))
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.pending_points:
            answer = QMessageBox.question(
                self, "Unsaved annotations",
                "There are unsaved points for patient {}. Save before closing?".format(
                    self.current_patient_id()
                ),
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save:
                if not self.save_patient():
                    event.ignore()
                    return
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MRIAnnotationApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()