I made this for personal use, I mainly used it with snipping tool screenshots from Cat Fantasy gatcha game to find skill/character icons for my wikipedia fan project. 

# IMG Finder Tool
<img width="1918" height="1035" alt="image" src="https://github.com/user-attachments/assets/3f90aa11-a5b5-4358-a84f-b5bb2452492a" />
<img width="1919" height="1025" alt="image" src="https://github.com/user-attachments/assets/b519f839-da25-4a77-badc-54ee9004d2b8" />



A PyQt6-based image similarity finder that uses deep learning embeddings and multiple similarity metrics to find duplicate or visually similar images.

## Features

- **Deep Learning Embeddings**: Uses MobileNetV3 for efficient image feature extraction
- **Multiple Similarity Metrics**: Combines cosine similarity, perceptual hashing, and SSIM
- **Configurable Matching**: Adjust weights for different similarity algorithms
- **B&W Mode**: Per-tab toggle for black & white image matching with optimized SSIM on binarized images
- **Tab Renaming**: Right-click any tab to rename it (tabs are saved in config)
- **Cross-Tab Moving**: Select images across multiple result tabs and move them all at once
- **Responsive UI**: Compact, grid-based layout that adapts to window size
- **Batch Processing**: Process multiple images efficiently with configurable batch sizes
- **GPU Support**: Optional GPU acceleration (CUDA)
- **Results Management**: Click to select images across tabs, move with silent operation

## UI Layout

The application uses an optimized responsive grid layout:
- **Tabs**: Load/paste images into tabs (max 10 tabs). Rename any tab by right-clicking it.
- **B&W Button**: Each tab has a B&W toggle button (left of the preview). Enable for black & white matching mode.
- **Info Icon**: Click the "i" icon in the top-right corner of each tab for instructions on paste/drag/load.
- **Folder Buttons**: Scan and Output folder selection on the same row
- **Action Buttons**: Run All (green), Move, and Settings with efficient spacing
- **Path Display**: Clear folder path information below buttons
- **Progress Tracking**: Real-time progress bar and status updates
- **Responsive Design**: Buttons resize based on window size
- **Cross-Tab Selection**: Select images across multiple result tabs and click Move to batch-move them all (no confirmation popup)

## Running the App

### Windows (recommended)
Just double-click:
```
run_imgfinder.bat
```

The script will:
1. Look for `.venv\Scripts\python.exe`
2. If found → run `main.py` using the venv interpreter  
3. If not found → run using system Python

### Manual Run
```
.venv\Scripts\activate
python main.py
```

## Creating the Virtual Environment

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Settings

Customize the following in the Settings panel:
- **Appearance**: Font size, thumbnail size, results per row
- **Performance**: Compute mode (CPU/GPU), thread count, batch size
- **Matching Weights**: Adjust similarity metric weights and boost parameters
- **Layout**: Configure top/bottom pane heights and left pane width (these control the split between tabs and results)

## Config Persistence

The app saves the following to `imgfinder_config.json`:
- Scan and output folder paths
- All settings (appearance, performance, weights, layout)
- **Tab names** (set via right-click rename)
- **Per-tab B&W mode state** (each tab's B&W toggle setting is saved)
