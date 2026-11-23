# SWCV

**SWCV** (*Scroll With Computer Vision*) is an open-source, educational prototype that uses a webcam and MediaPipe hand landmarks to control vertical scrolling with simple hand gestures.

The project includes a Tkinter desktop interface, adjustable scroll settings, camera controls, persistent preferences, and Instagram, Facebook, and YouTube themes. The repository preserves six development iterations, showing how the application evolved.

> **Project status:** Experimental / archived. The repository is best suited to learning and experimentation; features and stability vary between versions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Archived%20Prototype-orange.svg)

## Highlights

- Real-time hand and finger tracking with MediaPipe
- One-finger gesture to scroll down
- Two-finger gesture to scroll up
- Adjustable scroll speed, smoothing, and finger-circle size
- Camera selection and camera visibility controls
- Tkinter settings and about panels
- Persistent local preferences
- Instagram, Facebook, and YouTube visual themes
- Camera failure detection and recovery prompts

## Supported Gestures

| Gesture | Action |
|---|---|
| One finger | Scroll down |
| Two fingers | Scroll up |
| Pinch | Detected; no default scrolling action |
| Open palm | Detected; no default scrolling action |
| Fist | Detected; no default scrolling action |

## Repository Structure

```text
SWCV/
├── attempt_1/                    # Initial app version
├── attempt_2/                    # Updated interface and preferences
├── attempt_3/                    # UI and gesture refinements
├── attempt_4/                    # Preference improvements
├── attempt_5/                    # Redesigned interface
├── attempt_6/
│   ├── App_v6_camError.py        # Latest version
│   └── cookies.py                # Preference storage
├── requirements.txt
└── README.md
```

The latest implementation is `attempt_6/App_v6_camError.py`.

## Getting Started

### Prerequisites

- Windows 10 or later is recommended
- Python 3.8 or later
- A webcam
- `pip` and a virtual-environment tool
- Access to a desktop or application window containing scrollable content

Some scripts use Windows-specific Tkinter behavior and global desktop input.

### Installation

Clone the repository:

```bash
git clone https://github.com/TageYassir/SWCV.git
cd SWCV
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Application

Start the latest implementation:

```bash
python attempt_6/App_v6_camError.py
```

The application will open a Tkinter window with the camera feed and scrolling controls.

1. Choose a camera index from the settings panel if needed.
2. Adjust scroll speed and smoothing.
3. Select a theme.
4. Press **Start Scrolling**.
5. Raise one finger to scroll down or two fingers to scroll up.

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `F12` | Show or hide the camera |
| `S` | Open settings |
| `Q` | Quit the application |

## Configuration

Preferences are stored in a local `.swcv_preferences.json` file inside the selected attempt directory. Common settings include:

- Theme
- Scroll speed
- Gesture smoothing
- Finger-circle scale
- Camera index

Generated preference and cache files should not contain sensitive information. Delete the preference file if you want to restore defaults.

## Known Limitations

- Gesture accuracy depends on camera quality, lighting, distance, and hand orientation.
- The repository is a sequence of prototypes rather than a finalized library or package.
- Most of the code is Windows-oriented and may require changes on other operating systems.
- `pyautogui` sends input to whichever window is currently focused.
- Some older attempts may contain experimental or incomplete behavior.
- Camera and system-input permissions may be required.

## Security and Responsible Use

SWCV can send system scrolling input and access your webcam. Before using it:

- Review the script you intend to run.
- Allow camera access only when needed.
- Keep the application visible and verified while control is active.
- Avoid using it where unintended input could affect important work.
- Do not share generated preference files or recordings containing personal data.

## Contributing

Contributions for bug fixes, documentation, gesture improvements, cross-platform support, tests, and reproducibility are welcome.

1. Fork the repository.
2. Create a focused branch.
3. Test your change with the latest implementation.
4. Document setup and reproduction steps.
5. Open a pull request with a clear explanation.

For larger ideas or direct collaboration, contact Yassir Tagemouati.

## Author and Contributor

| Author | Area |
|:--:|:--|
| <img src="https://github.com/TageYassir.png" width="120"/> | **Yassir Tagemouati** <br><br> Big Data Engineering Student (Cycle d'Ingénieur) <br> Azure Data Engineering • Data Warehousing • Analytics <br><br> **GitHub:** [@TageYassir](https://github.com/TageYassir) <br> **Email:** [tagemouatiyassir@gmail.com](mailto:tagemouatiyassir@gmail.com) |

## Open Source and Study Use

This project is published for learning and educational study. It is released under the [MIT License](LICENSE), so you can study, modify, and reuse it subject to the license terms.

## License

Copyright (c) 2026 Yassir Tagemouati

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
