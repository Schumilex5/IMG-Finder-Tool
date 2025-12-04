I made this for personal use, I mainly used it with snipping tool screenshots from Cat Fantasy gatcha game to find skill/character icons for my wikipedia fan project. 

# IMG Finder Tool
<img width="1919" height="1028" alt="image" src="https://github.com/user-attachments/assets/2b5c48e3-93c1-4393-a279-64d6b089a5bf" />

A PyQt6-based image similarity finder that uses deep learning embeddings and multiple similarity metrics to find duplicate or visually similar images.

## Features

- **Deep Learning Embeddings**: Uses MobileNetV3 for efficient image feature extraction
- **Multiple Similarity Metrics**: Combines cosine similarity, perceptual hashing, and SSIM
- **Configurable Matching**: Adjust weights for different similarity algorithms
- **Responsive UI**: Compact, grid-based layout that adapts to window size
- **Batch Processing**: Process multiple images efficiently with configurable batch sizes
- **GPU Support**: Optional GPU acceleration (CUDA)
- **Results Management**: Easy selection and batch operations on matched images

## UI Layout

The application uses an optimized responsive grid layout:
- **Folder Buttons**: Scan and Output folder selection on the same row
- **Action Buttons**: Run All (green), Move, and Settings with efficient spacing
- **Path Display**: Clear folder path information below buttons
- **Progress Tracking**: Real-time progress bar and status updates
- **Responsive Design**: Buttons resize based on window size

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
- **Layout**: Configure pane heights and widths
