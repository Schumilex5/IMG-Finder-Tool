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

from layout_manager import SimpleLayoutManager

CONFIG_NAME = "imgfinder_config.json"
MAX_TABS = 11
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# ---- defaults for settings ----
DEFAULT_COMPUTE_MODE = "GPU Only"
DEFAULT_THREADS = 4
DEFAULT_BATCH_SIZE = 8
DEFAULT_FONT_SIZE = 16
DEFAULT_THUMB_SIZE = 140
DEFAULT_COLUMNS = 5
DEFAULT_W_COS = 0.55
DEFAULT_W_PHASH = 0.25
DEFAULT_W_SSIM = 0.20
DEFAULT_BOOST_DIST = 6
DEFAULT_BOOST_VAL = 0.15
DEFAULT_CAND_LIMIT = 80


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
        self.selected_paths: set[str] = set()
        self.thumb_refs: list[QtGui.QPixmap] = []

        # layout settings (overridden by MainWindow)
        self.columns = DEFAULT_COLUMNS
        self.font_size = DEFAULT_FONT_SIZE
        self.thumb_size = DEFAULT_THUMB_SIZE

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

    def set_layout_settings(self, columns: int, font_size: int, thumb_size: int):
        self.columns = max(1, columns)
        self.font_size = max(6, font_size)
        self.thumb_size = max(40, thumb_size)

    def clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.thumb_refs.clear()
        self.selected_paths.clear()

    def set_results(self, results: list[tuple[float, str]]):
        self.results = results
        self.display_results()

    def display_results(self):
        self.clear_grid()
        if not self.results:
            self.title_label.setText("No results.")
            return

        self.title_label.setText("Top Matches")
        cols = self.columns

        for idx, (score, path) in enumerate(self.results):
            row = idx // cols
            col = idx % cols

            frame = QtWidgets.QFrame()
            if path in self.selected_paths:
                frame.setStyleSheet("background-color: #5865f2; border-radius: 6px;")
            else:
                frame.setStyleSheet("background-color: #2b2d31; border-radius: 6px;")

            v = QtWidgets.QVBoxLayout(frame)
            v.setContentsMargins(6, 6, 6, 6)
            v.setSpacing(4)

            try:
                pil = Image.open(path).convert("RGBA")
                pil.thumbnail((self.thumb_size, self.thumb_size))
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
            name_lbl.setStyleSheet(f"color: #f2f3f5; font-size: {self.font_size}px;")
            v.addWidget(name_lbl)

            score_font = max(self.font_size - 1, 6)
            score_lbl = QtWidgets.QLabel(f"{score:.3f}")
            score_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            score_lbl.setStyleSheet(f"color: #999999; font-size: {score_font}px;")
            v.addWidget(score_lbl)

            frame.mousePressEvent = self._make_select_handler(frame, path)
            self.grid.addWidget(frame, row, col)

    def _make_select_handler(self, frame, path):
        # toggle selection
        def handler(event):
            if path in self.selected_paths:
                self.selected_paths.remove(path)
                frame.setStyleSheet("background-color: #2b2d31; border-radius: 6px;")
            else:
                self.selected_paths.add(path)
                frame.setStyleSheet("background-color: #5865f2; border-radius: 6px;")
        return handler

    def move_selected(self, output_folder: str):
        if not self.selected_paths:
            QtWidgets.QMessageBox.critical(self, "Error", "No results selected in this results tab.")
            return

        os.makedirs(output_folder, exist_ok=True)

        remaining_results: list[tuple[float, str]] = []
        moved_count = 0
        errors = []

        for src in list(self.selected_paths):
            if not os.path.isfile(src):
                errors.append(f"File not found: {src}")
                self.selected_paths.discard(src)
                continue

            filename = os.path.basename(src)
            base, ext = os.path.splitext(filename)
            dest = os.path.join(output_folder, filename)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(output_folder, f"{base}_moved{counter}{ext}")
                counter += 1
            try:
                shutil.move(src, dest)
                moved_count += 1
                self.selected_paths.discard(src)
            except Exception as e:
                errors.append(f"Failed to move {src}: {e}")

        for score, path in self.results:
            if path not in self.selected_paths and os.path.exists(path):
                remaining_results.append((score, path))

        self.results = remaining_results
        self.display_results()

        if errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Move Completed With Errors",
                "Moved {} files.\nErrors:\n{}".format(moved_count, "\n".join(errors)),
            )
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Move Completed",
                f"Moved {moved_count} files.",
            )


# ---------------------- IMAGE TAB (INPUT) ----------------------


class ImageTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.query_image_pil: Image.Image | None = None
        self.query_phash = None
        self.query_embed = None
        self.query_gray = None
        self.is_bw = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # compact info "i" - tooltip on hover shows instructions
        info_icon = QtWidgets.QLabel("i")
        info_icon.setFixedSize(22, 22)
        info_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        info_icon.setStyleSheet(
            "background-color: #1e1f22; color: #ffffff; border-radius: 11px; font-weight: bold;"
        )
        info_icon.setToolTip(
            "Paste screenshot or image (CTRL+V)\n"
            "Drag & drop images onto tabs.\n"
            "Or use 'Load Image into Current Tab...'"
        )
        # place the icon at the top-right of the tab area
        info_row = QtWidgets.QHBoxLayout()
        info_row.addStretch(1)
        info_row.addWidget(info_icon)
        layout.addLayout(info_row)

        self.preview_frame = QtWidgets.QFrame()
        self.preview_frame.setStyleSheet("background-color: #2b2d31; border-radius: 6px;")
        self.preview_frame.setMinimumSize(0, 0)
        self.preview_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        pf_layout = QtWidgets.QVBoxLayout(self.preview_frame)
        pf_layout.setContentsMargins(0, 0, 0, 0)
        pf_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("color: #777777;")
        pf_layout.addWidget(self.preview_label)

        layout.addWidget(self.preview_frame, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)

    def set_query_image(self, img: Image.Image):
        self.query_image_pil = img
        disp = img.copy()
        disp.thumbnail((280, 280))
        self.preview_label.setPixmap(pil_to_qpixmap(disp))


# ---------------------- PASTE TAB WIDGET (DRAG & DROP) ----------------------


class PasteTabWidget(QtWidgets.QTabWidget):
    filesDropped = QtCore.pyqtSignal(list, int)  # paths, start_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(IMAGE_EXTS):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        urls = event.mimeData().urls()
        paths: list[str] = []
        for url in urls:
            if url.isLocalFile():
                p = url.toLocalFile()
                if p.lower().endswith(IMAGE_EXTS):
                    paths.append(p)

        if not paths:
            event.ignore()
            return

        tab_bar = self.tabBar()
        bar_pos = tab_bar.mapFrom(self, event.position().toPoint())
        idx = tab_bar.tabAt(bar_pos)
        if idx < 0:
            idx = self.currentIndex()

        self.filesDropped.emit(paths, idx)
        event.acceptProposedAction()


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
        weight_cos,
        weight_phash,
        weight_ssim,
        boost_distance,
        boost_value,
        candidate_limit,
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

        self.weight_cos = weight_cos
        self.weight_phash = weight_phash
        self.weight_ssim = weight_ssim
        self.boost_distance = boost_distance
        self.boost_value = boost_value
        self.candidate_limit = candidate_limit

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
        self.feature_extractor.eval()
        embeds = []
        with torch.no_grad():
            for i in range(0, len(pil_list), self.batch_size):
                self.throttle()
                batch = pil_list[i: i + self.batch_size]
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
        cand_count = min(self.candidate_limit, total_images)
        total_work = total_images + 1 + (num_queries * cand_count)
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

        # --- step 2: embeddings (batched, counted as 1 step) ---
        asset_embeds_arr = self.compute_batch_embeddings(asset_pils)
        done += 1
        self.progress.emit(done, total_work)

        asset_norms = np.linalg.norm(asset_embeds_arr, axis=1) + 1e-8

        eps = 1e-8
        sum_final = self.weight_cos + self.weight_phash + self.weight_ssim
        if sum_final <= 0:
            w_cos_f = 1.0
            w_ph_f = 0.0
            w_ss_f = 0.0
        else:
            w_cos_f = self.weight_cos / sum_final
            w_ph_f = self.weight_phash / sum_final
            w_ss_f = self.weight_ssim / sum_final

        sum_base = self.weight_cos + self.weight_phash
        if sum_base <= 0:
            w_cos_b = 0.7
            w_ph_b = 0.3
        else:
            w_cos_b = self.weight_cos / sum_base
            w_ph_b = self.weight_phash / sum_base

        results_by_tab: dict[int, list[tuple[float, str]]] = {}

        for qinfo in self.tabs_info:
            q_idx = qinfo["index"]
            q_hash = qinfo["phash"]
            q_embed = qinfo["embed"]
            q_gray = qinfo["gray"]

            # support special mode for black & white tab
            mode = qinfo.get("mode", "default")

            ph_dists = np.array([q_hash - h for h in asset_hashes], dtype=np.float32)
            ph_sim = 1.0 - (ph_dists / 64.0)
            ph_sim = np.clip(ph_sim, 0.0, 1.0)

            dot = asset_embeds_arr @ q_embed
            q_norm = np.linalg.norm(q_embed) + eps
            cos = dot / (asset_norms * q_norm)
            cos = np.clip(cos, -1.0, 1.0)
            cos_norm = (cos + 1.0) / 2.0

            # adjust weights when matching black & white imagery
            if mode == "bw":
                # favor structural similarity and pHash for B&W
                adj_cos = self.weight_cos * 0.20
                adj_ph = self.weight_phash * 1.10
                adj_ss = self.weight_ssim * 2.0

                sum_final_adj = adj_cos + adj_ph + adj_ss
                if sum_final_adj <= 0:
                    w_cos_f = 0.0
                    w_ph_f = 0.5
                    w_ss_f = 0.5
                else:
                    w_cos_f = adj_cos / sum_final_adj
                    w_ph_f = adj_ph / sum_final_adj
                    w_ss_f = adj_ss / sum_final_adj

                sum_base_adj = adj_cos + adj_ph
                if sum_base_adj <= 0:
                    w_cos_b = 0.3
                    w_ph_b = 0.7
                else:
                    w_cos_b = adj_cos / sum_base_adj
                    w_ph_b = adj_ph / sum_base_adj
            else:
                # use global-normalized weights computed earlier
                w_cos_f = w_cos_f
                w_ph_f = w_ph_f
                w_ss_f = w_ss_f
                w_cos_b = w_cos_b
                w_ph_b = w_ph_b

            base_score = w_cos_b * cos_norm + w_ph_b * ph_sim
            idx_sorted = np.argsort(base_score)[::-1][:cand_count]

            results = []
            for i in idx_sorted:
                self.throttle()
                g = asset_grays[i]
                try:
                    # if in B&W mode, compare binarized versions to emphasize shape/contrast
                    if mode == "bw":
                        _, q_bin = cv2.threshold(q_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        _, g_bin = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        s_ssim = ssim(q_bin, g_bin, data_range=255)
                    else:
                        s_ssim = ssim(q_gray, g, data_range=255)
                except Exception:
                    s_ssim = 0.0
                s_ssim = float(np.clip(s_ssim, 0.0, 1.0))

                d = ph_dists[i]
                boost = 0.0
                if d <= self.boost_distance:
                    boost = self.boost_value

                final = (
                    w_cos_f * cos_norm[i]
                    + w_ph_f * ph_sim[i]
                    + w_ss_f * s_ssim
                    + boost
                )
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

        # icon
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(base_dir, "icon.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QtGui.QIcon(icon_path))
        except Exception:
            pass

        self.apply_discord_style()

        # model and settings
        self.feature_extractor = None
        self.transform = None
        self.device = "cpu"

        self.folder_path = None
        self.output_folder_path = None

        self.thread: QtCore.QThread | None = None
        self.worker: MatchWorker | None = None

        # settings state
        self.compute_mode = DEFAULT_COMPUTE_MODE
        self.threads = DEFAULT_THREADS
        self.batch_size = DEFAULT_BATCH_SIZE
        self.results_font_size = DEFAULT_FONT_SIZE
        self.results_thumb_size = DEFAULT_THUMB_SIZE
        self.results_columns = DEFAULT_COLUMNS

        self.weight_cos = DEFAULT_W_COS
        self.weight_phash = DEFAULT_W_PHASH
        self.weight_ssim = DEFAULT_W_SSIM
        self.boost_distance = DEFAULT_BOOST_DIST
        self.boost_value = DEFAULT_BOOST_VAL
        self.candidate_limit = DEFAULT_CAND_LIMIT

        self.hsplit_sizes = None
        self.vsplit_sizes = None
        self.split_top_height = 300     # default top pane height in pixels
        self.split_bottom_height = 450  # default bottom pane height in pixels
        self.split_left_width = 600     # default left pane width in pixels

        self.build_ui()
        self.load_config()
        self.apply_settings_to_ui()
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
                font-size: 16px;
            }
            QPushButton {
                background-color: #5865f2;
                color: #ffffff;
                border-radius: 6px;
                padding: 8px 14px;
                font-size: 14px;
                font-weight: 500;
                border: none;
            }
            QPushButton:hover {
                background-color: #4752c4;
            }
            QPushButton:pressed {
                background-color: #3c41a8;
            }
            QPushButton:disabled {
                background-color: #4f545c;
                color: #999999;
            }
            QLabel {
                background-color: transparent;
                font-size: 15px;
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
            QSpinBox, QDoubleSpinBox {
                background-color: #1e1f22;
                border-radius: 4px;
                padding: 8px 10px;
                font-size: 14px;
                min-height: 28px;
                min-width: 100px;
            }
            QComboBox {
                background-color: #1e1f22;
                border-radius: 4px;
                padding: 8px 10px;
                font-size: 14px;
                min-height: 28px;
                min-width: 140px;
            }
            QGroupBox {
                font-size: 15px;
                font-weight: 600;
                border: 1px solid #404249;
                border-radius: 4px;
                padding-top: 10px;
                margin-top: 8px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px 0 0;
            }
            QFormLayout {
                spacing: 8px 12px;
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
        mode = self.compute_mode
        wants_gpu = mode in ("GPU Only", "CPU + GPU") and torch.cuda.is_available()
        self.device = "cuda" if wants_gpu else "cpu"

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        base_model = mobilenet_v3_small(weights=weights)
        modules = list(base_model.children())[:-1]  # features + avgpool
        self.feature_extractor = torch.nn.Sequential(*modules).to(self.device)
        self.feature_extractor.eval()

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

        # --- TOP CONTAINER (with horizontal splitter for left/right) ---
        self.top_container = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(self.top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        # inside top: horizontal splitter (tabs / sidebar)
        self.hsplit = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.hsplit.setChildrenCollapsible(False)
        top_layout.addWidget(self.hsplit)

        # left: paste tabs
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.paste_tabs = PasteTabWidget()
        self.paste_tabs.filesDropped.connect(self.on_files_dropped_into_paste_tabs)
        for i in range(MAX_TABS):
            tab = ImageTab()
            # tab 11 (index 10) is a special black & white matcher
            if i == 10:
                self.paste_tabs.addTab(tab, "B&W")
            else:
                self.paste_tabs.addTab(tab, f"Tab {i+1}")
        left_layout.addWidget(self.paste_tabs)
        self.hsplit.addWidget(left_widget)

        # right: sidebar with stacked pages (main controls / settings)
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        self.sidebar_stack = QtWidgets.QStackedWidget()
        right_layout.addWidget(self.sidebar_stack)

        self.hsplit.addWidget(right_widget)

        # allow both sides to shrink
        left_widget.setMinimumSize(0, 0)
        right_widget.setMinimumSize(0, 0)

        # create sidebar pages
        self.build_main_sidebar_page()
        self.build_settings_page()

        # --- BOTTOM CONTAINER (RESULTS) ---
        self.bottom_container = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(self.bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        self.results_tabs = QtWidgets.QTabWidget()
        for i in range(MAX_TABS):
            rt = ResultsTab()
            rt.set_layout_settings(self.results_columns, self.results_font_size, self.results_thumb_size)
            if i == 10:
                self.results_tabs.addTab(rt, "B&W")
            else:
                self.results_tabs.addTab(rt, f"Results {i+1}")
        bottom_layout.addWidget(self.results_tabs, stretch=1)

        # --- USE SimpleLayoutManager FOR TOP/BOTTOM SPLIT ---
        self.layout_manager = SimpleLayoutManager(self.top_container, self.bottom_container)
        main_layout.addWidget(self.layout_manager, stretch=1)

        # default proportions: 40% top, 60% bottom
        self.layout_manager.set_sizes(300, 450)

        # sync paste tabs <-> results tabs
        self.paste_tabs.currentChanged.connect(self.on_paste_tab_changed)
        self.results_tabs.currentChanged.connect(self.on_results_tab_changed)

    def build_main_sidebar_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Use grid layout for buttons
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # Row 0: Scan + Output (2 columns)
        self.btn_scan = QtWidgets.QPushButton("Scan Folder")
        self.btn_scan.setMinimumHeight(32)
        self.btn_scan.clicked.connect(self.select_scan_folder)
        grid.addWidget(self.btn_scan, 0, 0)

        self.btn_output = QtWidgets.QPushButton("Output Folder")
        self.btn_output.setMinimumHeight(32)
        self.btn_output.clicked.connect(self.select_output_folder)
        grid.addWidget(self.btn_output, 0, 1)

        # Row 1: Scan path label (full width, 2 columns)
        self.lbl_scan = QtWidgets.QLabel("Scan: (not set)")
        self.lbl_scan.setStyleSheet("color: #9fa0a3; font-size: 11px; padding: 4px 8px; background-color: #1e1f22; border-radius: 2px;")
        self.lbl_scan.setWordWrap(True)
        grid.addWidget(self.lbl_scan, 1, 0, 1, 2)

        # Row 2: Output path label (full width, 2 columns)
        self.lbl_output = QtWidgets.QLabel("Output: (not set)")
        self.lbl_output.setStyleSheet("color: #9fa0a3; font-size: 11px; padding: 4px 8px; background-color: #1e1f22; border-radius: 2px;")
        self.lbl_output.setWordWrap(True)
        grid.addWidget(self.lbl_output, 2, 0, 1, 2)

        # Row 3: Load Image (full width, 2 columns)
        self.btn_load_image = QtWidgets.QPushButton("Load Image")
        self.btn_load_image.setMinimumHeight(32)
        self.btn_load_image.clicked.connect(self.browse_image_into_current_tab)
        grid.addWidget(self.btn_load_image, 3, 0, 1, 2)

        layout.addLayout(grid)
        layout.addSpacing(4)

        # Row 4: Run All (full width)
        self.btn_run = QtWidgets.QPushButton("Run All")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.setStyleSheet("QPushButton { background-color: #43b581; font-weight: bold; font-size: 13px; }")
        self.btn_run.clicked.connect(self.run_all)
        layout.addWidget(self.btn_run)

        # Row 5: Move + Settings (2 columns)
        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(8)
        
        self.btn_move = QtWidgets.QPushButton("Move")
        self.btn_move.setMinimumHeight(32)
        self.btn_move.clicked.connect(self.move_selected_from_active_result_tab)
        button_row.addWidget(self.btn_move)
        
        self.btn_settings = QtWidgets.QPushButton("Settings")
        self.btn_settings.setMinimumHeight(32)
        self.btn_settings.clicked.connect(self.show_settings_page)
        button_row.addWidget(self.btn_settings)
        
        layout.addLayout(button_row)

        layout.addSpacing(4)

        # Row 6: Progress (full width)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setValue(0)
        self.progress.setMinimumHeight(20)
        layout.addWidget(self.progress)

        self.lbl_progress = QtWidgets.QLabel("Progress: 0% (0/0)")
        self.lbl_progress.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_progress.setStyleSheet("color: #b9bbbe; font-size: 11px;")
        layout.addWidget(self.lbl_progress)

        layout.addStretch(1)

        self.sidebar_stack.addWidget(page)
        self.main_sidebar_page = page

    def build_settings_page(self):
        # Main container with sticky back button
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Sticky back button at top
        self.btn_settings_back = QtWidgets.QPushButton("← Back")
        self.btn_settings_back.setMinimumHeight(38)
        self.btn_settings_back.clicked.connect(self.on_settings_back)
        back_frame = QtWidgets.QFrame()
        back_layout = QtWidgets.QVBoxLayout(back_frame)
        back_layout.setContentsMargins(10, 8, 10, 8)
        back_layout.setSpacing(0)
        back_layout.addWidget(self.btn_settings_back)
        back_frame.setStyleSheet("QFrame { border-bottom: 1px solid #1e1f22; }")
        container_layout.addWidget(back_frame)

        # Scrollable settings area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        page = QtWidgets.QWidget()
        scroll.setWidget(page)
        
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        # Appearance group
        grp_appearance = QtWidgets.QGroupBox("Appearance")
        g_layout = QtWidgets.QFormLayout(grp_appearance)
        g_layout.setSpacing(12)
        g_layout.setHorizontalSpacing(16)
        g_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        g_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.spin_results_font = QtWidgets.QSpinBox()
        self.spin_results_font.setRange(6, 32)
        self.spin_results_font.setValue(DEFAULT_FONT_SIZE)
        self.spin_results_font.setMinimumWidth(120)
        g_layout.addRow("Results font size:", self.spin_results_font)

        self.combo_thumb_size = QtWidgets.QComboBox()
        self.combo_thumb_size.addItem("Small (80px)", 80)
        self.combo_thumb_size.addItem("Medium (120px)", 120)
        self.combo_thumb_size.addItem("Large (160px)", 160)
        self.combo_thumb_size.setCurrentIndex(1)
        self.combo_thumb_size.setMinimumWidth(140)
        g_layout.addRow("Thumbnail size:", self.combo_thumb_size)

        self.spin_columns = QtWidgets.QSpinBox()
        self.spin_columns.setRange(1, 10)
        self.spin_columns.setValue(DEFAULT_COLUMNS)
        self.spin_columns.setMinimumWidth(120)
        g_layout.addRow("Results per row:", self.spin_columns)

        layout.addWidget(grp_appearance)

        # Performance group
        grp_perf = QtWidgets.QGroupBox("Performance")
        p_layout = QtWidgets.QFormLayout(grp_perf)
        p_layout.setSpacing(12)
        p_layout.setHorizontalSpacing(16)
        p_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        p_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["CPU Only", "GPU Only", "CPU + GPU"])
        self.mode_combo.setMinimumWidth(140)
        p_layout.addRow("Compute mode:", self.mode_combo)

        self.spin_threads = QtWidgets.QSpinBox()
        self.spin_threads.setRange(1, 32)
        self.spin_threads.setValue(DEFAULT_THREADS)
        self.spin_threads.setMinimumWidth(120)
        p_layout.addRow("Threads:", self.spin_threads)

        self.batch_combo = QtWidgets.QComboBox()
        for b in [4, 8, 16, 32, 64]:
            self.batch_combo.addItem(str(b))
        self.batch_combo.setCurrentText(str(DEFAULT_BATCH_SIZE))
        self.batch_combo.setMinimumWidth(120)
        p_layout.addRow("Batch size:", self.batch_combo)

        layout.addWidget(grp_perf)

        # Matching group
        grp_match = QtWidgets.QGroupBox("Matching Weights")
        m_layout = QtWidgets.QFormLayout(grp_match)
        m_layout.setSpacing(12)
        m_layout.setHorizontalSpacing(16)
        m_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        m_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.spin_w_cos = QtWidgets.QDoubleSpinBox()
        self.spin_w_cos.setRange(0.0, 10.0)
        self.spin_w_cos.setSingleStep(0.05)
        self.spin_w_cos.setDecimals(3)
        self.spin_w_cos.setValue(DEFAULT_W_COS)
        self.spin_w_cos.setMinimumWidth(120)
        m_layout.addRow("Cosine weight:", self.spin_w_cos)

        self.spin_w_phash = QtWidgets.QDoubleSpinBox()
        self.spin_w_phash.setRange(0.0, 10.0)
        self.spin_w_phash.setSingleStep(0.05)
        self.spin_w_phash.setDecimals(3)
        self.spin_w_phash.setValue(DEFAULT_W_PHASH)
        self.spin_w_phash.setMinimumWidth(120)
        m_layout.addRow("pHash weight:", self.spin_w_phash)

        self.spin_w_ssim = QtWidgets.QDoubleSpinBox()
        self.spin_w_ssim.setRange(0.0, 10.0)
        self.spin_w_ssim.setSingleStep(0.05)
        self.spin_w_ssim.setDecimals(3)
        self.spin_w_ssim.setValue(DEFAULT_W_SSIM)
        self.spin_w_ssim.setMinimumWidth(120)
        m_layout.addRow("SSIM weight:", self.spin_w_ssim)

        self.spin_boost_dist = QtWidgets.QSpinBox()
        self.spin_boost_dist.setRange(0, 64)
        self.spin_boost_dist.setValue(DEFAULT_BOOST_DIST)
        self.spin_boost_dist.setMinimumWidth(120)
        m_layout.addRow("Boost distance ≤:", self.spin_boost_dist)

        self.spin_boost_val = QtWidgets.QDoubleSpinBox()
        self.spin_boost_val.setRange(0.0, 1.0)
        self.spin_boost_val.setSingleStep(0.01)
        self.spin_boost_val.setDecimals(3)
        self.spin_boost_val.setValue(DEFAULT_BOOST_VAL)
        self.spin_boost_val.setMinimumWidth(120)
        m_layout.addRow("Boost value:", self.spin_boost_val)

        self.spin_cand_limit = QtWidgets.QSpinBox()
        self.spin_cand_limit.setRange(10, 1000)
        self.spin_cand_limit.setValue(DEFAULT_CAND_LIMIT)
        self.spin_cand_limit.setMinimumWidth(120)
        m_layout.addRow("Candidate limit:", self.spin_cand_limit)

        layout.addWidget(grp_match)

        # Layout group
        grp_layout = QtWidgets.QGroupBox("Layout")
        l_layout = QtWidgets.QFormLayout(grp_layout)
        l_layout.setSpacing(12)
        l_layout.setHorizontalSpacing(16)
        l_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        l_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.spin_split_top_height = QtWidgets.QSpinBox()
        self.spin_split_top_height.setRange(50, 1500)
        self.spin_split_top_height.setValue(300)  # top pane height (40% default)
        self.spin_split_top_height.setMinimumWidth(120)
        l_layout.addRow("Top pane height (px):", self.spin_split_top_height)

        self.spin_split_bottom_height = QtWidgets.QSpinBox()
        self.spin_split_bottom_height.setRange(50, 1500)
        self.spin_split_bottom_height.setValue(450)  # bottom pane height (60% default)
        self.spin_split_bottom_height.setMinimumWidth(120)
        l_layout.addRow("Bottom pane height (px):", self.spin_split_bottom_height)

        self.spin_split_left_width = QtWidgets.QSpinBox()
        self.spin_split_left_width.setRange(100, 1500)
        self.spin_split_left_width.setValue(600)  # left pane width
        self.spin_split_left_width.setMinimumWidth(120)
        l_layout.addRow("Left pane width (px):", self.spin_split_left_width)

        layout.addWidget(grp_layout)
        layout.addStretch(1)

        # Credits
        credits = QtWidgets.QLabel("Made by Mia Iceberg")
        credits.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        credits.setStyleSheet("color: #7d8084; font-size: 10px; padding: 8px;")
        layout.addWidget(credits)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_reset_defaults = QtWidgets.QPushButton("Reset to Defaults")
        self.btn_reset_defaults.setMinimumHeight(36)
        self.btn_reset_defaults.clicked.connect(self.reset_defaults)
        self.btn_apply_settings = QtWidgets.QPushButton("Apply")
        self.btn_apply_settings.setMinimumHeight(36)
        self.btn_apply_settings.clicked.connect(self.apply_settings_from_ui)
        btn_row.addWidget(self.btn_reset_defaults)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_apply_settings)
        layout.addLayout(btn_row)

        container_layout.addWidget(scroll, 1)
        self.sidebar_stack.addWidget(container)
        self.settings_page = container

    # ----- config -----

    @property
    def config_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_NAME)

    def load_config(self):
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        scan = data.get("scan_folder") or ""
        out = data.get("output_folder") or ""

        if scan and os.path.isdir(scan):
            self.folder_path = scan
        if out and os.path.isdir(out):
            self.output_folder_path = out

        self.compute_mode = data.get("compute_mode", self.compute_mode)
        self.threads = int(data.get("threads", self.threads))
        self.batch_size = int(data.get("batch_size", self.batch_size))
        self.results_font_size = int(data.get("results_font_size", self.results_font_size))
        self.results_thumb_size = int(data.get("results_thumb_size", self.results_thumb_size))
        self.results_columns = int(data.get("results_columns", self.results_columns))

        weights = data.get("weights", {})
        self.weight_cos = float(weights.get("cosine", self.weight_cos))
        self.weight_phash = float(weights.get("phash", self.weight_phash))
        self.weight_ssim = float(weights.get("ssim", self.weight_ssim))
        self.boost_distance = int(weights.get("boost_distance", self.boost_distance))
        self.boost_value = float(weights.get("boost_value", self.boost_value))
        self.candidate_limit = int(weights.get("candidate_limit", self.candidate_limit))

        self.split_top_height = int(data.get("split_top_height", 300))
        self.split_bottom_height = int(data.get("split_bottom_height", 450))
        self.split_left_width = int(data.get("split_left_width", 600))
        self.hsplit_sizes = data.get("hsplit_sizes")

    def save_config(self):
        data = {
            "scan_folder": self.folder_path or "",
            "output_folder": self.output_folder_path or "",
            "compute_mode": self.compute_mode,
            "threads": self.threads,
            "batch_size": self.batch_size,
            "results_font_size": self.results_font_size,
            "results_thumb_size": self.results_thumb_size,
            "results_columns": self.results_columns,
            "weights": {
                "cosine": self.weight_cos,
                "phash": self.weight_phash,
                "ssim": self.weight_ssim,
                "boost_distance": self.boost_distance,
                "boost_value": self.boost_value,
                "candidate_limit": self.candidate_limit,
            },
            "split_top_height": self.split_top_height,
            "split_bottom_height": self.split_bottom_height,
            "split_left_width": self.split_left_width,
        }

        try:
            if self.hsplit is not None:
                data["hsplit_sizes"] = self.hsplit.sizes()
        except Exception:
            pass

        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def apply_settings_to_ui(self):
        if self.folder_path:
            self.lbl_scan.setText(f"Scan: {os.path.basename(self.folder_path) or self.folder_path}")
        if self.output_folder_path:
            self.lbl_output.setText(f"Output: {os.path.basename(self.output_folder_path) or self.output_folder_path}")

        self.spin_results_font.setValue(self.results_font_size)

        idx = 1
        for i in range(self.combo_thumb_size.count()):
            if int(self.combo_thumb_size.itemData(i)) == self.results_thumb_size:
                idx = i
                break
        self.combo_thumb_size.setCurrentIndex(idx)

        self.spin_columns.setValue(self.results_columns)

        mode_index = self.mode_combo.findText(self.compute_mode)
        if mode_index < 0:
            mode_index = 1  # GPU Only
        self.mode_combo.setCurrentIndex(mode_index)

        self.spin_threads.setValue(self.threads)
        self.batch_combo.setCurrentText(str(self.batch_size))

        self.spin_w_cos.setValue(self.weight_cos)
        self.spin_w_phash.setValue(self.weight_phash)
        self.spin_w_ssim.setValue(self.weight_ssim)
        self.spin_boost_dist.setValue(self.boost_distance)
        self.spin_boost_val.setValue(self.boost_value)
        self.spin_cand_limit.setValue(self.candidate_limit)

        self.update_results_tab_layout()

        try:
            if self.hsplit_sizes:
                self.hsplit.setSizes(self.hsplit_sizes)
            # Apply layout manager sizes from config
            self.layout_manager.set_sizes(self.split_top_height, self.split_bottom_height)
        except Exception:
            pass

    def update_results_tab_layout(self):
        for i in range(self.results_tabs.count()):
            rt: ResultsTab = self.results_tabs.widget(i)
            rt.set_layout_settings(self.results_columns, self.results_font_size, self.results_thumb_size)
            if rt.results:
                rt.display_results()

    # ----- settings actions -----

    def show_settings_page(self):
        self.sidebar_stack.setCurrentWidget(self.settings_page)

    def on_settings_back(self):
        self.apply_settings_from_ui()
        self.sidebar_stack.setCurrentWidget(self.main_sidebar_page)

    def apply_settings_from_ui(self):
        self.results_font_size = self.spin_results_font.value()
        self.results_thumb_size = int(self.combo_thumb_size.currentData())
        self.results_columns = self.spin_columns.value()

        self.compute_mode = self.mode_combo.currentText()
        self.threads = self.spin_threads.value()
        self.batch_size = int(self.batch_combo.currentText())

        self.weight_cos = self.spin_w_cos.value()
        self.weight_phash = self.spin_w_phash.value()
        self.weight_ssim = self.spin_w_ssim.value()
        self.boost_distance = self.spin_boost_dist.value()
        self.boost_value = self.spin_boost_val.value()
        self.candidate_limit = self.spin_cand_limit.value()

        # Apply layout sizes from layout manager spinners
        self.split_top_height, self.split_bottom_height = self.layout_manager.get_sizes()

        self.update_results_tab_layout()
        self.save_config()
        self.weight_ssim = self.spin_w_ssim.value()
        self.boost_distance = self.spin_boost_dist.value()
        self.boost_value = self.spin_boost_val.value()
        self.candidate_limit = self.spin_cand_limit.value()

        self.update_results_tab_layout()
        self.setup_model()
        self.save_config()

    def reset_defaults(self):
        self.spin_results_font.setValue(DEFAULT_FONT_SIZE)

        idx = 1
        for i in range(self.combo_thumb_size.count()):
            if int(self.combo_thumb_size.itemData(i)) == DEFAULT_THUMB_SIZE:
                idx = i
                break
        self.combo_thumb_size.setCurrentIndex(idx)

        self.spin_columns.setValue(DEFAULT_COLUMNS)

        mode_index = self.mode_combo.findText(DEFAULT_COMPUTE_MODE)
        if mode_index < 0:
            mode_index = 1
        self.mode_combo.setCurrentIndex(mode_index)

        self.spin_threads.setValue(DEFAULT_THREADS)
        self.batch_combo.setCurrentText(str(DEFAULT_BATCH_SIZE))

        self.spin_w_cos.setValue(DEFAULT_W_COS)
        self.spin_w_phash.setValue(DEFAULT_W_PHASH)
        self.spin_w_ssim.setValue(DEFAULT_W_SSIM)
        self.spin_boost_dist.setValue(DEFAULT_BOOST_DIST)
        self.spin_boost_val.setValue(DEFAULT_BOOST_VAL)
        self.spin_cand_limit.setValue(DEFAULT_CAND_LIMIT)

        self.apply_settings_from_ui()

    # ----- tab syncing -----

    def on_paste_tab_changed(self, idx: int):
        if 0 <= idx < self.results_tabs.count():
            if self.results_tabs.currentIndex() != idx:
                self.results_tabs.setCurrentIndex(idx)

    def on_results_tab_changed(self, idx: int):
        if 0 <= idx < self.paste_tabs.count():
            if self.paste_tabs.currentIndex() != idx:
                self.paste_tabs.setCurrentIndex(idx)

    # ----- clipboard & drag/drop helpers -----

    def load_pil_into_tab(self, img: Image.Image, tab_index: int):
        if not (0 <= tab_index < self.paste_tabs.count()):
            return
        tab: ImageTab = self.paste_tabs.widget(tab_index)
        # mark B&W special tab
        tab.is_bw = (tab_index == 10)
        tab.set_query_image(img)

        pil = img.convert("RGB")
        tab.query_phash = imagehash.phash(pil)
        tab.query_embed = self.compute_embedding(pil)
        gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
        tab.query_gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    def load_images_into_tabs(self, paths: list[str], start_index: int):
        idx = start_index
        used = 0
        for path in paths:
            if idx >= self.paste_tabs.count():
                if used > 0:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Not enough tabs",
                        f"Loaded {used} images; no more tabs available.",
                    )
                break
            if not os.path.isfile(path):
                continue
            if not path.lower().endswith(IMAGE_EXTS):
                continue
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue

            self.load_pil_into_tab(img, idx)
            used += 1
            idx += 1

        if used == 0 and paths:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to load any dropped images.")

    def on_files_dropped_into_paste_tabs(self, paths: list[str], start_index: int):
        self.load_images_into_tabs(paths, start_index)

    def paste_to_current_tab(self):
        try:
            clip = ImageGrab.grabclipboard()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to grab clipboard:\n{e}")
            return

        if isinstance(clip, Image.Image):
            self.load_pil_into_tab(clip, self.paste_tabs.currentIndex())
            return

        if isinstance(clip, list):
            paths: list[str] = []
            for item in clip:
                if isinstance(item, str) and os.path.isfile(item) and item.lower().endswith(IMAGE_EXTS):
                    paths.append(item)
            if not paths:
                QtWidgets.QMessageBox.critical(self, "Error", "Clipboard file list does not contain image files.")
                return
            self.load_images_into_tabs(paths, self.paste_tabs.currentIndex())
            return

        QtWidgets.QMessageBox.critical(self, "Error", "Clipboard does not contain an image or image files.")

    # ----- browse image into current tab -----

    def browse_image_into_current_tab(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open image:\n{e}")
            return

        self.load_pil_into_tab(img, self.paste_tabs.currentIndex())

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
                        "mode": "bw" if getattr(tab, "is_bw", False) else "default",
                    }
                )

        if not tabs_info:
            QtWidgets.QMessageBox.critical(self, "Error", "No tabs have pasted/loaded images.")
            return

        image_paths = []
        for root_dir, dirs, files in os.walk(self.folder_path):
            for fn in files:
                if fn.lower().endswith(IMAGE_EXTS):
                    image_paths.append(os.path.join(root_dir, fn))
        if not image_paths:
            QtWidgets.QMessageBox.information(self, "No images", "No image files found in scan folder.")
            return

        for i in range(self.results_tabs.count()):
            rt: ResultsTab = self.results_tabs.widget(i)
            rt.set_results([])

        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_progress.setText("Progress: 0% (0/0)")

        baseline = psutil.cpu_percent(interval=0.3)
        free = max(0.0, 100.0 - baseline)
        cpu_limit = free * 0.7
        cpu_limit = max(5.0, cpu_limit)

        cpu_count = os.cpu_count() or 4
        max_threads = max(1, int(cpu_count * 0.7))
        threads = min(self.threads, max_threads)

        total_images = len(image_paths)
        num_queries = len(tabs_info)
        cand_count = min(self.candidate_limit, total_images)
        total_work = total_images + 1 + (num_queries * cand_count)
        self.progress.setMaximum(total_work)

        batch_size = int(self.batch_size)

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
            self.weight_cos,
            self.weight_phash,
            self.weight_ssim,
            self.boost_distance,
            self.boost_value,
            self.candidate_limit,
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
