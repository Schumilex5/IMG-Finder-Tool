I made this for personal use, I mainly used it with snipping tool screenshots from Cat Fantasy gatcha game to find skill/character icons for my wikipedia fan project. 

# IMG Finder Tool
This repository includes a startup BAT script that:
- Uses the local `.venv` virtual environment if available
- Falls back to system Python if the venv is missing

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

## Creating the Virtual Environment

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then simply run:
```
run_imgfinder.bat
```
