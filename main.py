import os
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
from PIL import Image, ImageGrab

import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms

import imagehash
from skimage.metrics import structural_similarity as ssim
import psutil

from PyQt6 import QtWidgets, QtGui, QtCore

CONFIG_NAME = "imgfinder_config.json"
MAX_TABS = 10


def pil_to_qpixmap(pil_img: Image.Image) -> QtGui.QPixmap:
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    data = pil_img.tobytes("raw", "RGBA")
    w, h = pil_img.size
    qimg = QtGui.QImage(data, w, h, QtGui.QImage.Format.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(qimg)


# ---------------------- RESULTS TAB ----------------------


class ResultsTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: list[tuple[float, str]] = []
        self.selected_path: str | None = None
        self.selected_frame: QtWidgets.QFrame | None = None
        self.thumb_refs: list[QtGui.QPixmap] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.title_label = QtWidgets.QLabel("No results yet.")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: #cccccc;")
        layout.addWidget(self.title_label)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none;")
        layout.addWidget(self.scroll_area)

        self.grid_container = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_container)
        self.grid.setContentsMargins(10, 10, 10, 10)
        self.grid.setHorizontalSpacing(16)
        self.grid.setVerticalSpacing(16)

        self.scroll_area.setWidget(self.grid_container)

    def clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.thumb_refs.clear()
        self.selected_frame = None
        self.selected_path = None

    def set_results(self, results: list[tuple[float, str]]):
        self.results = results
        self.display_results()

    def display_results(self):
        self.clear_grid()
        if not self.results:
            self.title_label.setText("No results.")
            return

        self.title_label.setText("Top Matches")
        cols = 5

        for idx, (score, path) in enumerate(self.results):
            row = idx // cols
            col = idx % cols

            frame = QtWidgets.QFrame()
            frame.setStyleSheet("background-color: #2b2d31; border-radius: 6px;")
            v = QtWidgets.QVBoxLayout(frame)
            v.setContentsMargins(6, 6, 6, 6)
            v.setSpacing(4)

            try:
                pil = Image.open(path).convert("RGBA")
                pil.thumbnail((120, 120))
                pix = pil_to_qpixmap(pil)
                self.thumb_refs.append(pix)
                img_lbl = QtWidgets.QLabel()
                img_lbl.setPixmap(pix)
                img_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                v.addWidget(img_lbl)
            except Exception:
                img_lbl = QtWidgets.QLabel("[img error]")
                img_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                img_lbl.setStyleSheet("color: #ff8888;")
                v.addWidget(img_lbl)

            name_lbl = QtWidgets.QLabel(os.path.basename(path))
            name_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            name_lbl.setStyleSheet("color: #f2f3f5; font-size: 10px;")
            v.addWidget(name_lbl)

            score_lbl = QtWidgets.QLabel(f"{score:.3f}")
            score_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            score_lbl.setStyleSheet("color: #999999; font-size: 9px;")
            v.addWidget(score_lbl)

            frame.mousePressEvent = self._make_select_handler(frame, path)
            self.grid.addWidget(frame, row, col)

    def _make_select_handler(self, frame, path):
        def handler(event):
            if self.selected_frame is not None:
                self.selected_frame.setStyleSheet("background-color: #2b2d31; border-radius: 6px;")
            self.selected_frame = frame
            self.selected_path = path
            frame.setStyleSheet("background-color: #5865f2; border-radius: 6px;")
        return handler

    def move_selected(self, output_folder: str):
        if not self.selected_path:
            QtWidgets.QMessageBox.critical(self, "Error", "No result selected in this results tab.")
            return
        src = self.selected_path
        if not os.path.isfile(src):
            QtWidgets.QMessageBox.critical(self, "Error", "Selected file no longer exists.")
            return

        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.basename(src)
        base, ext = os.path.splitext(filename)
        dest = os.path.join(output_folder, filename)
        counter = 1
        while os.path.exists(dest):
            dest = os.path.join(output_folder, f"{base}_moved{counter}{ext}")
            counter += 1
        try:
            shutil.move(src, dest)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to move:\n{e}")
            return

        self.results = [(s, p) for (s, p) in self.results if p != src]
        self.display_results()


# ---------------------- IMAGE TAB (INPUT) ----------------------


class ImageTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.query_image_pil: Image.Image | None = None
        self.query_phash = None
        self.query_embed = None
        self.query_gray = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.info_label = QtWidgets.QLabel("Paste screenshot with CTRL+V")
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        layout.addWidget(self.info_label)

        self.preview_frame = QtWidgets.QFrame()
        self.preview_frame.setStyleSheet("background-color: #2b2d31; border-radius: 6px;")
        self.preview_frame.setFixedSize(320, 320)
        pf_layout = QtWidgets.QVBoxLayout(self.preview_frame)
        pf_layout.setContentsMargins(0, 0, 0, 0)
        pf_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("color: #777777;")
        pf_layout.addWidget(self.preview_label)

        layout.addWidget(self.preview_frame, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)

    def set_query_from_clipboard(self):
        try:
            img = ImageGrab.grabclipboard()
            if not isinstance(img, Image.Image):
                QtWidgets.QMessageBox.critical(self, "Error", "Clipboard does not contain an image.")
                return False
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to grab clipboard image:\n{e}")
            return False

        self.query_image_pil = img
        disp = img.copy()
        disp.thumbnail((280, 280))
        self.preview_label.setPixmap(pil_to_qpixmap(disp))
        return True


# ---------------------- MATCH WORKER (THREAD) ----------------------


class MatchWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(dict)

    def __init__(
        self,
        tabs_info,
        image_paths,
        threads,
        batch_size,
        feature_extractor,
        transform,
        device,
        cpu_limit_percent,
    ):
        super().__init__()
        self.tabs_info = tabs_info
        self.image_paths = image_paths
        self.threads = threads
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.device = device
        self.cpu_limit_percent = cpu_limit_percent

        self.proc = psutil.Process(os.getpid())
        self._last_check = time.time()

    # ---- throttling ----

    def throttle(self):
        if self.cpu_limit_percent <= 0:
            return
        now = time.time()
        if now - self._last_check < 0.25:
            return
        self._last_check = now
        try:
            usage = self.proc.cpu_percent(interval=0.0)
            if usage > self.cpu_limit_percent:
                time.sleep(0.05)
        except Exception:
            pass

    # ---- embedding helper ----

    def compute_batch_embeddings(self, pil_list):
        """
        pil_list: list of PIL RGB images
        returns: np.ndarray of shape (N, D)
        """
        self.feature_extractor.eval()
        embeds = []
        with torch.no_grad():
            for i in range(0, len(pil_list), self.batch_size):
                self.throttle()
                batch = pil_list[i : i + self.batch_size]
                if not batch:
                    continue
                tensors = [self.transform(img) for img in batch]
                t = torch.stack(tensors, dim=0).to(self.device)
                feat = self.feature_extractor(t)  # (B, C, 1, 1)
                feat = feat.view(feat.size(0), -1)
                feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)
                embeds.append(feat.cpu().numpy())
        if embeds:
            return np.concatenate(embeds, axis=0)
        return np.zeros((len(pil_list), 576), dtype=np.float32)

    # ---- main run ----

    def run(self):
        total_images = len(self.image_paths)
        if total_images == 0 or not self.tabs_info:
            self.finished.emit({})
            return

        num_queries = len(self.tabs_info)
        # rough estimate: loading + hashing + embedding + scoring
        total_work = total_images * 2 + total_images * num_queries
        done = 0

        # --- step 1: load images, phash, gray (multi-threaded) ---
        asset_hashes = [None] * total_images
        asset_grays = [None] * total_images
        asset_pils = [None] * total_images

        def prep_worker(idx, path):
            try:
                pil = Image.open(path).convert("RGB")
            except Exception:
                return idx, None, None, None
            try:
                h = imagehash.phash(pil)
            except Exception:
                h = imagehash.ImageHash(np.zeros((8, 8), dtype=bool))
            gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
            gray_small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
            return idx, h, gray_small, pil

        with ThreadPoolExecutor(max_workers=self.threads) as ex:
            futures = {ex.submit(prep_worker, i, p): i for i, p in enumerate(self.image_paths)}
            for fut in as_completed(futures):
                self.throttle()
                idx, h, gray, pil = fut.result()
                asset_hashes[idx] = h
                asset_grays[idx] = gray
                asset_pils[idx] = pil
                done += 1
                self.progress.emit(done, total_work)

        # --- step 2: embeddings (batched) ---
        asset_embeds_arr = self.compute_batch_embeddings(asset_pils)
        done += total_images
        self.progress.emit(done, total_work)

        asset_norms = np.linalg.norm(asset_embeds_arr, axis=1) + 1e-8

        # --- step 3: queries ---
        results_by_tab: dict[int, list[tuple[float, str]]] = {}

        for qinfo in self.tabs_info:
            q_idx = qinfo["index"]
            q_hash = qinfo["phash"]
            q_embed = qinfo["embed"]
            q_gray = qinfo["gray"]

            # phash similarity
            ph_dists = np.array([q_hash - h for h in asset_hashes], dtype=np.float32)
            ph_sim = 1.0 - (ph_dists / 64.0)
            ph_sim = np.clip(ph_sim, 0.0, 1.0)

            # embedding cosine
            dot = asset_embeds_arr @ q_embed
            q_norm = np.linalg.norm(q_embed) + 1e-8
            cos = dot / (asset_norms * q_norm)
            cos = np.clip(cos, -1.0, 1.0)
            cos_norm = (cos + 1.0) / 2.0  # 0..1

            base_score = 0.7 * cos_norm + 0.3 * ph_sim
            cand_count = min(80, total_images)
            idx_sorted = np.argsort(base_score)[::-1][:cand_count]

            results = []
            for i in idx_sorted:
                self.throttle()
                g = asset_grays[i]
                try:
                    s_ssim = ssim(q_gray, g, data_range=255)
                except Exception:
                    s_ssim = 0.0
                s_ssim = float(np.clip(s_ssim, 0.0, 1.0))

                d = ph_dists[i]
                boost = 0.0
                if d <= 6:
                    boost = 0.15

                final = 0.55 * cos_norm[i] + 0.25 * ph_sim[i] + 0.20 * s_ssim + boost
                results.append((float(final), self.image_paths[i]))

                done += 1
                self.progress.emit(done, total_work)

            results.sort(key=lambda x: x[0], reverse=True)
            results_by_tab[q_idx] = results[:25]

        self.finished.emit(results_by_tab)


# ---------------------- MAIN WINDOW ----------------------


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("IMG Finder Tool (pHash + MobileNetV3 + SSIM)")
        self.resize(1280, 850)

        self.apply_discord_style()

        # model will be created depending on compute mode
        self.feature_extractor = None
        self.transform = None
        self.device = "cpu"

        self.folder_path = None
        self.output_folder_path = None

        self.thread: QtCore.QThread | None = None
        self.worker: MatchWorker | None = None

        self.build_ui()
        self.load_config()

        # default compute mode: GPU Only
        self.mode_combo.setCurrentText("GPU Only")
        self.setup_model()

        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Paste, self, self.paste_to_current_tab)

    # ----- style -----

    def apply_discord_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #313338;
            }
            QWidget {
                background-color: #313338;
                color: #ffffff;
                font-family: Segoe UI, Arial;
            }
            QPushButton {
                background-color: #5865f2;
                color: #ffffff;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #4752c4;
            }
            QPushButton:disabled {
                background-color: #4f545c;
                color: #999999;
            }
            QLabel {
                background-color: transparent;
            }
            QTabBar::tab {
                background: #1e1f22;
                color: #dcddde;
                padding: 4px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #2b2d31;
                color: #ffffff;
            }
            QScrollArea {
                border: none;
            }
            QSpinBox {
                background-color: #1e1f22;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QComboBox {
                background-color: #1e1f22;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QProgressBar {
                border: 1px solid #1e1f22;
                border-radius: 4px;
                text-align: center;
                background: #1e1f22;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #43b581;
                border-radius: 4px;
            }
        """)

    # ----- model -----

    def setup_model(self):
        mode = self.mode_combo.currentText()
        wants_gpu = mode in ("GPU Only", "CPU + GPU") and torch.cuda.is_available()
        self.device = "cuda" if wants_gpu else "cpu"

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        base_model = mobilenet_v3_small(weights=weights)
        modules = list(base_model.children())[:-1]  # features + avgpool
        self.feature_extractor = torch.nn.Sequential(*modules).to(self.device)
        self.feature_extractor.eval()

        # use the weights' recommended transforms (keeps your original images untouched)
        self.transform = weights.transforms()

    def compute_embedding(self, pil_img: Image.Image) -> np.ndarray:
        self.feature_extractor.eval()
        with torch.no_grad():
            t = self.transform(pil_img).unsqueeze(0).to(self.device)
            feat = self.feature_extractor(t)
            feat = feat.view(feat.size(0), -1)
            feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)
            return feat.cpu().numpy()[0]

    # ----- UI -----

    def build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=2)

        # left: paste tabs
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.paste_tabs = QtWidgets.QTabWidget()
        for i in range(MAX_TABS):
            tab = ImageTab()
            self.paste_tabs.addTab(tab, f"Tab {i+1}")
        left_layout.addWidget(self.paste_tabs)
        splitter.addWidget(left_widget)

        # right: global controls
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        self.btn_scan = QtWidgets.QPushButton("Scan Folder...")
        self.btn_scan.clicked.connect(self.select_scan_folder)
        right_layout.addWidget(self.btn_scan)

        self.lbl_scan = QtWidgets.QLabel("Scan: (not set)")
        self.lbl_scan.setStyleSheet("color: #b9bbbe;")
        right_layout.addWidget(self.lbl_scan)

        self.btn_output = QtWidgets.QPushButton("Output Folder...")
        self.btn_output.clicked.connect(self.select_output_folder)
        right_layout.addWidget(self.btn_output)

        self.lbl_output = QtWidgets.QLabel("Output: (not set)")
        self.lbl_output.setStyleSheet("color: #b9bbbe;")
        right_layout.addWidget(self.lbl_output)

        # compute mode dropdown
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Compute mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["CPU Only", "GPU Only", "CPU + GPU"])
        self.mode_combo.currentIndexChanged.connect(self.setup_model)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch(1)
        right_layout.addLayout(mode_row)

        # threads
        threads_row = QtWidgets.QHBoxLayout()
        threads_row.addWidget(QtWidgets.QLabel("Threads:"))
        self.spin_threads = QtWidgets.QSpinBox()
        self.spin_threads.setRange(1, 32)
        self.spin_threads.setValue(4)
        threads_row.addWidget(self.spin_threads)

        # batch size dropdown
        threads_row.addWidget(QtWidgets.QLabel("Batch size:"))
        self.batch_combo = QtWidgets.QComboBox()
        for b in [4, 8, 16, 32, 64]:
            self.batch_combo.addItem(str(b))
        self.batch_combo.setCurrentText("8")
        threads_row.addWidget(self.batch_combo)
        threads_row.addStretch(1)
        right_layout.addLayout(threads_row)

        self.btn_run = QtWidgets.QPushButton("Run All")
        self.btn_run.clicked.connect(self.run_all)
        right_layout.addWidget(self.btn_run)

        self.btn_move = QtWidgets.QPushButton("Move Selected (Result Tab)")
        self.btn_move.clicked.connect(self.move_selected_from_active_result_tab)
        right_layout.addWidget(self.btn_move)

        right_layout.addStretch(1)
        splitter.addWidget(right_widget)
        splitter.setSizes([640, 640])

        # progress bar + label (in the gap above results)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setValue(0)
        main_layout.addWidget(self.progress)

        self.lbl_progress = QtWidgets.QLabel("Progress: 0% (0/0)")
        self.lbl_progress.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_progress.setStyleSheet("color: #b9bbbe;")
        main_layout.addWidget(self.lbl_progress)

        # bottom: results tabs
        self.results_tabs = QtWidgets.QTabWidget()
        for i in range(MAX_TABS):
            rt = ResultsTab()
            self.results_tabs.addTab(rt, f"Results {i+1}")
        main_layout.addWidget(self.results_tabs, stretch=3)

    # ----- config -----

    @property
    def config_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_NAME)

    def load_config(self):
        if not os.path.exists(self.config_path):
            data = {"scan_folder": "", "output_folder": ""}
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"scan_folder": "", "output_folder": ""}

        scan = data.get("scan_folder") or ""
        out = data.get("output_folder") or ""

        if scan and os.path.isdir(scan):
            self.folder_path = scan
            self.lbl_scan.setText(f"Scan: {os.path.basename(scan) or scan}")
        if out and os.path.isdir(out):
            self.output_folder_path = out
            self.lbl_output.setText(f"Output: {os.path.basename(out) or out}")

    def save_config(self):
        data = {
            "scan_folder": self.folder_path or "",
            "output_folder": self.output_folder_path or "",
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # ----- clipboard -----

    def paste_to_current_tab(self):
        tab: ImageTab = self.paste_tabs.currentWidget()
        if tab.set_query_from_clipboard():
            pil = tab.query_image_pil.convert("RGB")
            tab.query_phash = imagehash.phash(pil)
            tab.query_embed = self.compute_embedding(pil)
            gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
            tab.query_gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    # ----- folder selection -----

    def select_scan_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Scan Folder")
        if path:
            self.folder_path = path
            self.lbl_scan.setText(f"Scan: {os.path.basename(path) or path}")
            self.save_config()

    def select_output_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_folder_path = path
            self.lbl_output.setText(f"Output: {os.path.basename(path) or path}")
            self.save_config()

    # ----- run all -----

    def run_all(self):
        if self.folder_path is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Scan folder is not set.")
            return

        tabs_info = []
        for i in range(self.paste_tabs.count()):
            tab: ImageTab = self.paste_tabs.widget(i)
            if tab.query_image_pil is not None and tab.query_embed is not None:
                tabs_info.append(
                    {
                        "index": i,
                        "phash": tab.query_phash,
                        "embed": tab.query_embed,
                        "gray": tab.query_gray,
                    }
                )

        if not tabs_info:
            QtWidgets.QMessageBox.critical(self, "Error", "No tabs have pasted images.")
            return

        image_paths = []
        for root_dir, dirs, files in os.walk(self.folder_path):
            for fn in files:
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    image_paths.append(os.path.join(root_dir, fn))
        if not image_paths:
            QtWidgets.QMessageBox.information(self, "No images", "No image files found in scan folder.")
            return

        # clear previous results
        for i in range(self.results_tabs.count()):
            rt: ResultsTab = self.results_tabs.widget(i)
            rt.set_results([])

        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_progress.setText("Progress: 0% (0/0)")

        # CPU free % -> 70% of that as limit
        baseline = psutil.cpu_percent(interval=0.3)
        free = max(0.0, 100.0 - baseline)
        cpu_limit = free * 0.7
        cpu_limit = max(5.0, cpu_limit)  # minimum limit

        # threads cap: 70% of cores
        cpu_count = os.cpu_count() or 4
        max_threads = max(1, int(cpu_count * 0.7))
        threads = min(self.spin_threads.value(), max_threads)

        total_images = len(image_paths)
        num_queries = len(tabs_info)
        total_work = total_images * 2 + total_images * num_queries
        self.progress.setMaximum(total_work)

        batch_size = int(self.batch_combo.currentText())

        self.thread = QtCore.QThread()
        self.worker = MatchWorker(
            tabs_info,
            image_paths,
            threads,
            batch_size,
            self.feature_extractor,
            self.transform,
            self.device,
            cpu_limit,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def update_progress(self, done, total):
        self.progress.setValue(done)
        pct = (done / total) * 100.0 if total else 0.0
        self.lbl_progress.setText(f"Progress: {pct:.1f}% ({done}/{total})")

    def on_worker_finished(self, results_by_tab: dict):
        for tab_index, results in results_by_tab.items():
            if 0 <= tab_index < self.results_tabs.count():
                rt: ResultsTab = self.results_tabs.widget(tab_index)
                rt.set_results(results)
        self.btn_run.setEnabled(True)

    # ----- move from results -----

    def move_selected_from_active_result_tab(self):
        if not self.output_folder_path:
            QtWidgets.QMessageBox.critical(self, "Error", "Output folder is not set.")
            return
        rt: ResultsTab = self.results_tabs.currentWidget()
        rt.move_selected(self.output_folder_path)


def main():
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
