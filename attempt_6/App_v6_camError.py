"""
SWCV - Scroll With Computer Vision
Gesture-based scrolling using hand tracking.
Themes: Instagram | Facebook | YouTube
Press 'S' for settings, 'F12' to hide/show camera, 'Q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pyautogui
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
import threading
import os
import tempfile
import json
import webbrowser
import datetime

# ======================
# Disable PyAutoGUI fail-safe
# ======================
pyautogui.FAILSAFE = False

# ======================
# Helper for debug prints
# ======================
def debug_print(*args):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    thread_name = threading.current_thread().name
    print(f"[{ts}][{thread_name}]", *args, flush=True)

# ======================
# COLOR THEMES
# ======================
THEMES = {
    "Instagram": {
        "primary": "#FAFAFA",
        "secondary": "#262626",
        "on_dark": "#FFFFFF",
        "accent": "#E1306C",
        "light": "#FFFFFF",
        "neutral": "#DBDBDB",
        "text": "#262626",
        "muted": "#6B6B6B",
        "error": "#C0245B",
        "social": "#405DE6",
        "warning": "#B5540B",
        "font": ("Helvetica", 10),
        "main_bgr": (108, 48, 225),
        "accent_bgr": (55, 119, 247),
        "text_bgr": (255, 255, 255),
        "circle_bgr": (108, 48, 225),
        "hand_bgr": (180, 58, 131),
        "hand_conn_bgr": (55, 119, 247),
    },
    "Facebook": {
        "primary": "#F0F2F5",
        "secondary": "#1C1E21",
        "on_dark": "#FFFFFF",
        "accent": "#1877F2",
        "light": "#FFFFFF",
        "neutral": "#DADDE1",
        "text": "#1C1E21",
        "muted": "#65676B",
        "error": "#C4262B",
        "social": "#1877F2",
        "warning": "#B4690E",
        "font": ("Helvetica", 10),
        "main_bgr": (242, 119, 24),
        "accent_bgr": (42, 183, 42),
        "text_bgr": (255, 255, 255),
        "circle_bgr": (242, 119, 24),
        "hand_bgr": (200, 160, 50),
        "hand_conn_bgr": (42, 183, 42),
    },
    "YouTube": {
        "primary": "#F9F9F9",
        "secondary": "#0F0F0F",
        "on_dark": "#FFFFFF",
        "accent": "#CC0000",
        "light": "#FFFFFF",
        "neutral": "#E5E5E5",
        "text": "#0F0F0F",
        "muted": "#606060",
        "error": "#CC0000",
        "social": "#065FD4",
        "warning": "#B4690E",
        "font": ("Helvetica", 10),
        "main_bgr": (0, 0, 255),
        "accent_bgr": (255, 255, 255),
        "text_bgr": (255, 255, 255),
        "circle_bgr": (0, 0, 255),
        "hand_bgr": (80, 80, 255),
        "hand_conn_bgr": (255, 255, 255),
    }
}

# ======================
# PREFERENCES – saved in the same folder as this script
# ======================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREFS_FILE = os.path.join(SCRIPT_DIR, ".swcv_preferences.json")

class UserPreferences:
    def __init__(self):
        self.default_preferences = {
            "theme": "Instagram",
            "scroll_speed": 30,
            "smoothing": 0.7,
            "circle_scale": 1.5,
            "camera_index": 0,
        }
        self.current_preferences = self.default_preferences.copy()
        self.load_preferences()

    def load_preferences(self):
        try:
            if os.path.exists(PREFS_FILE):
                with open(PREFS_FILE, "r") as f:
                    loaded = json.load(f)
                self.current_preferences = {**self.default_preferences, **loaded}
                debug_print("Preferences loaded from", PREFS_FILE)
                return True
        except Exception as e:
            debug_print("Error loading preferences:", e)
        return False

    def save_preferences(self):
        try:
            os.makedirs(os.path.dirname(PREFS_FILE), exist_ok=True)
            with open(PREFS_FILE, "w") as f:
                json.dump(self.current_preferences, f, indent=4)
            debug_print("Preferences saved to", PREFS_FILE)
            return True
        except Exception as e:
            debug_print("Error saving preferences:", e)
        return False

    def get(self, key, default=None):
        return self.current_preferences.get(key, default)

    def set(self, key, value):
        self.current_preferences[key] = value
        self.save_preferences()

# ======================
# MEDIAPIPE SETUP
# ======================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ======================
# GESTURE DETECTION
# ======================
def is_finger_open(landmarks, tip_id, pip_id):
    return landmarks.landmark[tip_id].y < landmarks.landmark[pip_id].y

def detect_gesture(hand_landmarks):
    thumb_open = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
    index_open = is_finger_open(hand_landmarks, 8, 6)
    middle_open = is_finger_open(hand_landmarks, 12, 10)
    ring_open = is_finger_open(hand_landmarks, 16, 14)
    pinky_open = is_finger_open(hand_landmarks, 20, 18)

    if index_open and not middle_open and not ring_open and not pinky_open:
        return "ONE FINGER"
    if index_open and middle_open and not ring_open and not pinky_open:
        return "TWO FINGERS"

    fingers = [thumb_open, index_open, middle_open, ring_open, pinky_open]
    count = sum(fingers)

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    dist = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

    if dist < 0.05 and not index_open:
        return "PINCH"
    if count == 5:
        return "OPEN PALM"
    if count == 0:
        return "FIST"
    return "UNKNOWN"

# ======================
# FINGER CIRCLE
# ======================
class FingerCircle:
    def __init__(self, smoothing=0.7, tip_buffer_size=5):
        self.smoothing = smoothing
        self.center = None
        self.radius = None
        self.tip_buffer = deque(maxlen=tip_buffer_size)

    def update(self, hand_landmarks, fw, fh, radius_scale, smoothing):
        self.smoothing = smoothing
        tip = hand_landmarks.landmark[8]
        tx, ty = tip.x * fw, tip.y * fh
        self.tip_buffer.append((tx, ty))
        mx = np.median([p[0] for p in self.tip_buffer])
        my = np.median([p[1] for p in self.tip_buffer])
        raw_center = (int(mx), int(my))

        wrist = hand_landmarks.landmark[0]
        mcp = hand_landmarks.landmark[9]
        dx, dy, dz = wrist.x - mcp.x, wrist.y - mcp.y, wrist.z - mcp.z
        norm = np.sqrt(dx * dx + dy * dy + dz * dz)
        raw_radius = int(norm * fw * radius_scale)

        if self.center is None:
            self.center = raw_center
            self.radius = max(raw_radius, 1)
        else:
            a = self.smoothing
            self.center = (
                int(a * raw_center[0] + (1 - a) * self.center[0]),
                int(a * raw_center[1] + (1 - a) * self.center[1]),
            )
            self.radius = max(int(a * raw_radius + (1 - a) * self.radius), 1)
        return self.center, self.radius

    def reset(self):
        self.center = None
        self.radius = None
        self.tip_buffer.clear()

def draw_circle_on_frame(frame, center, radius, color):
    if center is None or radius is None or radius <= 0:
        return frame
    c = (int(center[0]), int(center[1]))
    r = int(radius)
    overlay = frame.copy()
    cv2.circle(overlay, c, r, color, -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.circle(frame, c, r, color, 2)
    cv2.circle(frame, c, 5, (0, 0, 255), -1)
    return frame

# ======================
# ICON GENERATOR (fallback for taskbar icon only)
# ======================
def create_icon_file():
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((2, 2, 62, 62), fill=(225, 48, 108), outline=(255, 255, 255), width=3)
    draw.text((16, 8), "S", fill=(255, 255, 255))
    draw.text((20, 24), "W", fill=(255, 255, 255))
    draw.text((18, 40), "C", fill=(255, 255, 255))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ico")
    img.save(tmp.name, format="ICO")
    tmp.close()
    return tmp.name

# ======================
# MAIN APPLICATION CLASS
# ======================
class SWCVApp:
    def __init__(self, root):
        debug_print("Initializing SWCVApp")
        self.root = root
        self.root.title("SWCV - Scroll With Computer Vision")
        self.preferences = UserPreferences()
        self.current_theme = self.preferences.get("theme", "Instagram")
        self.colors = THEMES[self.current_theme]
        self.root.geometry("1300x720")
        self.root.resizable(False, False)

        self.scroll_speed = self.preferences.get("scroll_speed", 30)
        self.smoothing = self.preferences.get("smoothing", 0.7)
        self.circle_scale = self.preferences.get("circle_scale", 1.5)

        self.running = True
        self.control_active = False
        self.show_camera = True
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.current_gesture = "NONE"
        self.gesture_lock = threading.Lock()
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.05
        self._icon_path = None
        self.main_app_active = False

        # Camera error handling
        self._camera_failure_count = 0
        self._camera_failure_threshold = 5      # consecutive failures before popup
        self._camera_error_shown = False

        # Camera handling (thread‑safe)
        self.available_cameras = []
        self.selected_camera_index = self.preferences.get("camera_index", 0)
        self.update_camera_list()
        self.cap_lock = threading.RLock()
        self.cap = self.open_camera(self.selected_camera_index)

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.finger_circle = FingerCircle()

        self.setup_icons()
        self.configure_styles()
        self.create_menu()
        self.setup_hotkeys()

        # Start video thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

        self.update_video_label()
        self.create_welcome_screen()
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)
        debug_print("SWCVApp initialized successfully")

    # ======================
    # CAMERA HELPERS
    # ======================
    def update_camera_list(self):
        debug_print("Updating camera list")
        self.available_cameras = []
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for idx, name in enumerate(devices):
                self.available_cameras.append((idx, name))
        except Exception:
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.available_cameras.append((i, f"Camera {i}"))
                    cap.release()
        if not self.available_cameras:
            self.available_cameras = [(0, "Default Camera")]
        debug_print("Available cameras:", self.available_cameras)

    def open_camera(self, index):
        debug_print(f"Opening camera index {index}")
        try:
            cap = cv2.VideoCapture(index)
        except Exception as e:
            debug_print(f"Exception opening camera {index}: {e}")
            return None
        if not cap.isOpened():
            debug_print(f"ERROR: Could not open camera index {index}")
            return None
        # Reset failure counters on a successful open
        self._camera_failure_count = 0
        self._camera_error_shown = False
        debug_print(f"Camera index {index} opened successfully")
        return cap

    def switch_camera(self, index):
        debug_print(f"switch_camera called with index={index}")
        with self.cap_lock:
            new_cap = self.open_camera(index)
            if new_cap is not None:
                old_cap = self.cap
                self.cap = new_cap
                self.selected_camera_index = index
                self.preferences.set("camera_index", index)
                debug_print("Camera switched, old cap will be released in 1 sec")
                threading.Timer(1.0, self._release_camera, args=[old_cap]).start()
                return True
            else:
                debug_print("Failed to open new camera, keeping old one")
                return False

    def _release_camera(self, cap):
        debug_print("Releasing old camera")
        if cap is not None:
            try:
                cap.release()
                debug_print("Old camera released")
            except Exception as e:
                debug_print("Error releasing camera:", e)

    # ======================
    # CAMERA ERROR POPUP
    # ======================
    def show_camera_error(self):
        """Show camera error message exactly once, then quit the app."""
        if self._camera_error_shown:
            return
        self._camera_error_shown = True

        error_window = tk.Toplevel(self.root)
        error_window.title("Camera Error")
        error_window.resizable(False, False)

        # Try to set the same icon as the main app
        if self._icon_path:
            try:
                error_window.iconbitmap(self._icon_path)
            except Exception:
                pass

        error_window.configure(bg=self.colors["primary"])
        window_width = 400
        window_height = 200
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        error_window.geometry(
            f"{window_width}x{window_height}+"
            f"{(screen_width // 2) - (window_width // 2)}+"
            f"{(screen_height // 2) - (window_height // 2)}"
        )

        error_content = [
            "Camera Error",
            "",
            "Another application is already using the camera",
            "",
        ]

        for i, line in enumerate(error_content):
            fg_color = self.colors["error"] if i == 0 else self.colors["text"]
            font_weight = "bold" if i == 0 else "normal"

            label = tk.Label(
                error_window,
                text=line,
                font=(self.colors["font"][0], self.colors["font"][1], font_weight),
                bg=self.colors["primary"],
                fg=fg_color,
                justify=tk.CENTER
            )
            label.pack(pady=0)

        ok_button = ttk.Button(
            error_window,
            text="OK",
            command=self.quit_application,
            style="Accent.TButton"
        )
        ok_button.pack(pady=10)

        error_window.protocol("WM_DELETE_WINDOW", self.quit_application)
        error_window.grab_set()

    # ======================
    # STYLES, ICONS, MENU, HOTKEYS
    # ======================
    def configure_styles(self):
        debug_print("Configuring styles")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=self.colors["primary"], foreground=self.colors["text"], font=self.colors["font"])
        style.configure("TFrame", background=self.colors["primary"])
        style.configure("TLabel", background=self.colors["primary"], foreground=self.colors["text"])
        style.configure("TNotebook", background=self.colors["primary"])
        style.configure("TNotebook.Tab", background=self.colors["neutral"], foreground=self.colors["text"], padding=(12, 6))
        style.map("TNotebook.Tab",
                  background=[("selected", self.colors["accent"])],
                  foreground=[("selected", "#FFFFFF")])
        style.configure("TButton", background=self.colors["neutral"], foreground=self.colors["text"],
                         bordercolor=self.colors["neutral"], focusthickness=1, padding=6)
        style.map("TButton",
                  background=[("disabled", "#E8E8E8"), ("active", self.colors["light"])],
                  foreground=[("disabled", "#A0A0A0"), ("active", self.colors["text"])])
        style.configure("Accent.TButton", background=self.colors["accent"], foreground="#FFFFFF", padding=6)
        style.map("Accent.TButton",
                  background=[("disabled", "#E8E8E8"), ("active", self.colors["accent"])],
                  foreground=[("disabled", "#A0A0A0")])
        style.configure("Social.TButton", background=self.colors["social"], foreground="#FFFFFF", padding=6)
        style.map("Social.TButton",
                  background=[("disabled", "#E8E8E8"), ("active", self.colors["social"])],
                  foreground=[("disabled", "#A0A0A0")])
        style.configure("Toolbutton", background=self.colors["neutral"], foreground=self.colors["text"])
        style.map("Toolbutton",
                  background=[("selected", self.colors["accent"]), ("active", self.colors["light"])],
                  foreground=[("selected", "#FFFFFF"), ("active", self.colors["text"])])
        style.configure("TCombobox", fieldbackground=self.colors["light"], background=self.colors["neutral"],
                         foreground=self.colors["text"])

    def setup_icons(self):
        debug_print("Setting up icons")
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "app.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
                self._icon_path = icon_path
            else:
                icon_path = create_icon_file()
                self.root.iconbitmap(icon_path)
                self._icon_path = icon_path
            if os.name == "nt":
                try:
                    from ctypes import windll
                    windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.swcv.app")
                except Exception:
                    pass
        except Exception as e:
            debug_print("Could not load icons:", e)
            self._icon_path = None
            try:
                icon = tk.PhotoImage(width=16, height=16)
                self.root.iconphoto(False, icon)
            except Exception:
                pass

    def create_menu(self):
        debug_print("Creating menu")
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit_application)
        menubar.add_cascade(label="File", menu=file_menu)
        sections_menu = tk.Menu(menubar, tearoff=0)
        sections_menu.add_command(label="Welcome", command=lambda: self.create_welcome_screen())
        sections_menu.add_command(label="Main App", command=lambda: self.show_main_app())
        sections_menu.add_command(label="Settings", command=lambda: self.show_settings())
        sections_menu.add_command(label="About", command=lambda: self.show_about())
        menubar.add_cascade(label="Sections", menu=sections_menu)
        self.root.config(menu=menubar)

    def setup_hotkeys(self):
        debug_print("Setting up hotkeys")
        self.root.bind("<F12>", lambda e: self.toggle_camera_visibility())
        self.root.bind("<s>", lambda e: self.show_settings())
        self.root.bind("<S>", lambda e: self.show_settings())
        self.root.bind("<q>", lambda e: self.quit_application())
        self.root.bind("<Q>", lambda e: self.quit_application())

    def toggle_camera_visibility(self):
        self.show_camera = not self.show_camera
        debug_print(f"Toggled camera visibility: {'on' if self.show_camera else 'off'}")
        if not self.show_camera and hasattr(self, "video_label") and self.video_label.winfo_exists():
            self.video_label.configure(image="", bg="black")
            self.video_label.image = None

    def clear_frame(self):
        debug_print("clear_frame called, setting main_app_active=False")
        self.main_app_active = False
        for widget in self.root.winfo_children():
            if not isinstance(widget, tk.Menu):
                widget.destroy()
        debug_print("Frame cleared")

    # ======================
    # WELCOME SCREEN (no logo image)
    # ======================
    def create_welcome_screen(self):
        debug_print("Creating welcome screen")
        self.clear_frame()
        self.stop_control()
        self.current_screen_method = self.create_welcome_screen

        welcome_frame = tk.Frame(self.root, bg=self.colors["primary"])
        welcome_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(welcome_frame, text="Welcome to SWCV", font=("Helvetica", 28, "bold"),
                 bg=self.colors["primary"], fg=self.colors["text"]).pack(pady=(30, 5))
        tk.Label(welcome_frame, text="Scroll With Computer Vision", font=("Helvetica", 14, "italic"),
                 bg=self.colors["primary"], fg="gray").pack(pady=(0, 20))
        tk.Label(welcome_frame,
                 text="IMPORTANT: For full functionality, the camera must be directly facing the user with no obstacles in between.",
                 font=("Helvetica", 11, "bold"), bg=self.colors["primary"], fg=self.colors["error"],
                 wraplength=700).pack(pady=(0, 20))
        tk.Label(welcome_frame,
                 text="This application allows you to scroll through any content\nusing hand gestures without any additional hardware.\n\nSimply hold up 1 finger to scroll down, or 2 fingers to scroll up.",
                 font=("Helvetica", 12), bg=self.colors["primary"], fg=self.colors["text"],
                 justify=tk.CENTER).pack(pady=10)

        guide_frame = tk.LabelFrame(welcome_frame, text="Quick Guide", font=("Helvetica", 12, "bold"),
                                    bg=self.colors["primary"], fg=self.colors["text"], padx=20, pady=10)
        guide_frame.pack(pady=20, padx=100, fill=tk.X)
        gestures = [
            ("1 Finger Up", "Scroll Down"),
            ("2 Fingers Up", "Scroll Up"),
            ("Pinch", "Detected (no action)"),
            ("Open Palm", "Detected (no action)"),
            ("Fist", "Detected (no action)"),
        ]
        for gesture, action in gestures:
            row = tk.Frame(guide_frame, bg=self.colors["primary"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=gesture, font=("Helvetica", 11, "bold"),
                     bg=self.colors["primary"], fg=self.colors["social"], width=18, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(row, text=f"->  {action}", font=("Helvetica", 11),
                     bg=self.colors["primary"], fg=self.colors["text"], anchor=tk.W).pack(side=tk.LEFT)

        ttk.Button(welcome_frame, text="Start SWCV", command=self.show_main_app,
                   style="Accent.TButton").pack(pady=20)
        tk.Label(welcome_frame, text="© 2026 SWCV  |  Scroll With Computer Vision  |  All rights reserved",
                 font=("Helvetica", 8), bg=self.colors["primary"], fg="gray").pack(side=tk.BOTTOM, pady=10)
        debug_print("Welcome screen displayed")

    # ======================
    # MAIN APP SCREEN
    # ======================
    def show_main_app(self):
        debug_print("show_main_app called")
        self.clear_frame()
        self.create_main_app()
        self.current_screen_method = self.show_main_app

    def create_main_app(self):
        debug_print("Creating main app")
        self.stop_control()
        self.main_app_active = True

        main_frame = tk.Frame(self.root, bg=self.colors["primary"])
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(main_frame, bg=self.colors["secondary"])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(left_panel, text="Camera Feed", font=("Helvetica", 12, "bold"),
                 bg=self.colors["secondary"], fg=self.colors["on_dark"]).pack(pady=(10, 5))
        self.video_label = tk.Label(left_panel, bg="black", width=640, height=480)
        self.video_label.pack(pady=5, padx=10)
        self.camera_hidden_label = tk.Label(left_panel, text="", font=("Helvetica", 10, "bold"),
                                            bg=self.colors["secondary"], fg="#FF6B6B")
        self.camera_hidden_label.pack()
        btn_frame = tk.Frame(left_panel, bg=self.colors["secondary"])
        btn_frame.pack(pady=10)
        self.start_btn = ttk.Button(btn_frame, text="Start Scrolling", command=self.start_control,
                                    style="Accent.TButton")
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Stop Scrolling", command=self.stop_control, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Hide/Show Camera", command=self.toggle_camera_visibility).pack(side=tk.LEFT, padx=5)
        tk.Label(left_panel, text=f"Theme: {self.current_theme}  |  F12: Hide/Show  |  S: Settings  |  Q: Quit",
                 font=("Helvetica", 9), bg=self.colors["secondary"], fg="#B3B3B3").pack(side=tk.BOTTOM, pady=5)

        right_panel = tk.Frame(main_frame, bg=self.colors["light"])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(right_panel, text="Control Status", font=("Helvetica", 16, "bold"),
                 bg=self.colors["light"], fg=self.colors["text"]).pack(pady=(15, 10))
        status_card = tk.LabelFrame(right_panel, text="Status", font=("Helvetica", 11, "bold"),
                                    bg=self.colors["light"], fg=self.colors["text"], padx=15, pady=10)
        status_card.pack(fill=tk.X, padx=15, pady=5)
        self.status_var = tk.StringVar(value="Ready to start scrolling")
        tk.Label(status_card, textvariable=self.status_var, font=("Helvetica", 12),
                 bg=self.colors["light"], fg=self.colors["text"], wraplength=350, justify=tk.LEFT).pack(anchor=tk.W, pady=3)
        gesture_card = tk.LabelFrame(right_panel, text="Detected Gesture", font=("Helvetica", 11, "bold"),
                                     bg=self.colors["light"], fg=self.colors["text"], padx=15, pady=10)
        gesture_card.pack(fill=tk.X, padx=15, pady=5)
        self.gesture_var = tk.StringVar(value="NONE")
        tk.Label(gesture_card, textvariable=self.gesture_var, font=("Helvetica", 18, "bold"),
                 bg=self.colors["light"], fg=self.colors["social"], wraplength=350).pack(anchor=tk.W, pady=3)
        self.gesture_action_var = tk.StringVar(value="No action")
        tk.Label(gesture_card, textvariable=self.gesture_action_var, font=("Helvetica", 11),
                 bg=self.colors["light"], fg=self.colors["text"]).pack(anchor=tk.W, pady=3)
        settings_card = tk.LabelFrame(right_panel, text="Current Settings", font=("Helvetica", 11, "bold"),
                                      bg=self.colors["light"], fg=self.colors["text"], padx=15, pady=10)
        settings_card.pack(fill=tk.X, padx=15, pady=5)
        self.settings_summary_var = tk.StringVar(
            value=f"Scroll Speed: {self.scroll_speed}\nSmoothing: {self.smoothing:.2f}\nCircle Scale: {self.circle_scale:.1f}\nTheme: {self.current_theme}"
        )
        tk.Label(settings_card, textvariable=self.settings_summary_var, font=("Consolas", 10),
                 bg=self.colors["light"], fg=self.colors["text"], justify=tk.LEFT).pack(anchor=tk.W)
        nav_frame = tk.Frame(right_panel, bg=self.colors["light"])
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=10)
        ttk.Button(nav_frame, text="Settings", command=self.show_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="About", command=self.show_about).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Welcome", command=self.create_welcome_screen).pack(side=tk.RIGHT, padx=5)
        tk.Label(right_panel, text="© 2026 SWCV  |  Scroll With Computer Vision",
                 font=("Helvetica", 8), bg=self.colors["light"], fg="gray").pack(side=tk.BOTTOM, pady=5)

        self._update_main_status()
        debug_print("Main app created")

    def _update_main_status(self):
        if not self.main_app_active:
            return
        if not hasattr(self, "gesture_var"):
            return
        with self.gesture_lock:
            g = self.current_gesture
        self.gesture_var.set(g)
        action_map = {
            "ONE FINGER": "Scrolling Down",
            "TWO FINGERS": "Scrolling Up",
            "PINCH": "Pinch detected",
            "OPEN PALM": "Open palm detected",
            "FIST": "Fist detected",
            "UNKNOWN": "Unknown gesture",
            "NONE": "No hand detected",
        }
        self.gesture_action_var.set(action_map.get(g, "No action"))
        self.status_var.set("Scrolling control is ACTIVE" if self.control_active else "Scrolling control is PAUSED")
        if hasattr(self, "camera_hidden_label") and self.camera_hidden_label.winfo_exists():
            self.camera_hidden_label.config(
                text="Camera is hidden (press F12 to show)" if not self.show_camera else ""
            )
        self.root.after(100, self._update_main_status)

    # ======================
    # SETTINGS SCREEN
    # ======================
    def show_settings(self):
        debug_print("show_settings called")
        self.stop_control()
        self.clear_frame()
        self.current_screen_method = self.show_settings
        self.create_settings()

    def create_settings(self):
        debug_print("Creating settings screen")
        main_frame = tk.Frame(self.root, bg=self.colors["primary"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(main_frame, text="SWCV Preferences", font=("Helvetica", 20, "bold"),
                 bg=self.colors["primary"], fg=self.colors["text"]).pack(pady=(5, 2))
        tk.Label(main_frame, text="Scroll With Computer Vision", font=("Helvetica", 9, "italic"),
                 bg=self.colors["primary"], fg="gray").pack(pady=(0, 10))
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tab_general = ttk.Frame(notebook)
        notebook.add(tab_general, text="General")
        theme_frame = tk.LabelFrame(tab_general, text="Appearance", font=("Helvetica", 12, "bold"),
                                    bg=self.colors["primary"], fg=self.colors["text"])
        theme_frame.pack(fill=tk.X, pady=10, padx=5)
        tk.Label(theme_frame, text="Theme:", font=("Helvetica", 11),
                 bg=self.colors["primary"], fg=self.colors["text"]).pack(side=tk.LEFT, padx=10)
        self.theme_var = tk.StringVar(value=self.current_theme)
        theme_menu = ttk.Combobox(theme_frame, textvariable=self.theme_var,
                                  values=list(THEMES.keys()), state="readonly", width=15)
        theme_menu.pack(side=tk.LEFT, padx=5)
        theme_menu.bind("<<ComboboxSelected>>", self.change_theme)

        tab_scroll = ttk.Frame(notebook)
        notebook.add(tab_scroll, text="Scrolling")
        sf = tk.LabelFrame(tab_scroll, text="Scroll Speed", font=("Helvetica", 12, "bold"),
                           bg=self.colors["primary"], fg=self.colors["text"], padx=15, pady=10)
        sf.pack(fill=tk.X, pady=10, padx=15)
        self.speed_val_label = tk.Label(sf, text=f"Value: {self.scroll_speed}",
                                        font=("Helvetica", 10, "bold"), bg=self.colors["primary"], fg=self.colors["social"])
        self.speed_val_label.pack(anchor=tk.W)
        self.speed_slider = tk.Scale(sf, from_=5, to=100, orient=tk.HORIZONTAL, command=self._on_speed,
                                     bg=self.colors["primary"], fg=self.colors["text"],
                                     highlightbackground=self.colors["primary"], troughcolor=self.colors["neutral"],
                                     length=500, font=("Helvetica", 9))
        self.speed_slider.set(self.scroll_speed)
        self.speed_slider.pack(fill=tk.X, pady=5)

        smf = tk.LabelFrame(tab_scroll, text="Smoothing", font=("Helvetica", 12, "bold"),
                            bg=self.colors["primary"], fg=self.colors["text"], padx=15, pady=10)
        smf.pack(fill=tk.X, pady=10, padx=15)
        self.smooth_val_label = tk.Label(smf, text=f"Value: {self.smoothing:.2f}",
                                         font=("Helvetica", 10, "bold"), bg=self.colors["primary"], fg=self.colors["social"])
        self.smooth_val_label.pack(anchor=tk.W)
        self.smooth_slider = tk.Scale(smf, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                                      command=self._on_smooth, bg=self.colors["primary"], fg=self.colors["text"],
                                      highlightbackground=self.colors["primary"], troughcolor=self.colors["neutral"],
                                      length=500, font=("Helvetica", 9))
        self.smooth_slider.set(self.smoothing)
        self.smooth_slider.pack(fill=tk.X, pady=5)

        cf = tk.LabelFrame(tab_scroll, text="Circle Scale", font=("Helvetica", 12, "bold"),
                           bg=self.colors["primary"], fg=self.colors["text"], padx=15, pady=10)
        cf.pack(fill=tk.X, pady=10, padx=15)
        self.circle_val_label = tk.Label(cf, text=f"Value: {self.circle_scale:.1f}",
                                         font=("Helvetica", 10, "bold"), bg=self.colors["primary"], fg=self.colors["social"])
        self.circle_val_label.pack(anchor=tk.W)
        self.circle_slider_w = tk.Scale(cf, from_=0.0, to=4.0, resolution=0.1, orient=tk.HORIZONTAL,
                                        command=self._on_circle, bg=self.colors["primary"], fg=self.colors["text"],
                                        highlightbackground=self.colors["primary"], troughcolor=self.colors["neutral"],
                                        length=500, font=("Helvetica", 9))
        self.circle_slider_w.set(self.circle_scale)
        self.circle_slider_w.pack(fill=tk.X, pady=5)

        tab_cam = ttk.Frame(notebook)
        notebook.add(tab_cam, text="Camera")
        cam_frame = tk.LabelFrame(tab_cam, text="Camera Selection", font=("Helvetica", 12, "bold"),
                                  bg=self.colors["primary"], fg=self.colors["text"])
        cam_frame.pack(fill=tk.X, pady=10, padx=5)
        tk.Label(cam_frame, text="Camera:", font=("Helvetica", 11),
                 bg=self.colors["primary"], fg=self.colors["text"]).pack(side=tk.LEFT, padx=10)
        cam_names = [name for idx, name in self.available_cameras]
        self.camera_combo = ttk.Combobox(cam_frame, values=cam_names, state="readonly", width=40)
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        current_idx = self.selected_camera_index
        for idx, name in self.available_cameras:
            if idx == current_idx:
                self.camera_combo.set(name)
                break
        ttk.Button(cam_frame, text="Refresh", command=self.refresh_camera_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(cam_frame, text="Apply", command=self.apply_camera_selection).pack(side=tk.LEFT, padx=5)

        back_frame = tk.Frame(main_frame, bg=self.colors["primary"])
        back_frame.pack(fill=tk.X, pady=10)
        ttk.Button(back_frame, text="Back to Main", command=self.show_main_app).pack(side=tk.LEFT, padx=10)

        tk.Label(main_frame, text="© 2026 SWCV  |  Scroll With Computer Vision  |  All rights reserved",
                 font=("Helvetica", 8), bg=self.colors["primary"], fg="gray").pack(side=tk.BOTTOM, pady=8)
        debug_print("Settings screen created")

    def refresh_camera_list(self):
        debug_print("Refreshing camera list")
        self.update_camera_list()
        cam_names = [name for idx, name in self.available_cameras]
        if hasattr(self, "camera_combo") and self.camera_combo.winfo_exists():
            self.camera_combo["values"] = cam_names
            if cam_names:
                self.camera_combo.set(cam_names[0])

    def apply_camera_selection(self):
        debug_print("Applying camera selection")
        if not hasattr(self, "camera_combo") or not self.camera_combo.winfo_exists():
            debug_print("Camera combo does not exist")
            return
        selected_name = self.camera_combo.get()
        debug_print(f"Selected camera name: {selected_name}")
        for idx, name in self.available_cameras:
            if name == selected_name:
                debug_print(f"Found matching camera index {idx}")
                if self.switch_camera(idx):
                    self.camera_combo.set(name)
                break

    def show_about(self):
        debug_print("show_about called")
        self.clear_frame()
        self.stop_control()
        self.current_screen_method = self.show_about
        about_frame = tk.Frame(self.root, bg=self.colors["primary"])
        about_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(about_frame, text="About SWCV", font=("Helvetica", 20, "bold"),
                 bg=self.colors["primary"], fg=self.colors["text"]).pack(pady=(10, 20))
        desc_frame = tk.Frame(about_frame, bg=self.colors["light"])
        desc_frame.pack(fill=tk.X, pady=10, padx=20)
        desc_text = (
            "SWCV - Scroll With Computer Vision\n"
            "Gesture-based scrolling using hand tracking.\n"
            "Control your scrolling with simple hand gestures\n"
            "and no additional hardware.\n\n"
            "This application provides a hands-free scrolling experience, "
            "ideal for reading, presentations, or any situation where you "
            "want to keep your hands away from the keyboard."
        )
        tk.Label(desc_frame, text=desc_text, font=("Helvetica", 11),
                 bg=self.colors["light"], fg=self.colors["text"], wraplength=600, justify=tk.LEFT).pack(pady=10, padx=10)
        dev_frame = tk.Frame(about_frame, bg=self.colors["light"])
        dev_frame.pack(fill=tk.X, pady=10, padx=20)
        tk.Label(dev_frame, text="Developed by Yassir Tagemouati", font=("Helvetica", 11, "bold"),
                 bg=self.colors["light"], fg=self.colors["text"]).pack(pady=5)
        social_frame = tk.Frame(about_frame, bg=self.colors["light"])
        social_frame.pack(fill=tk.X, pady=10, padx=20)
        ttk.Button(social_frame, text="LinkedIn", command=lambda: webbrowser.open("https://www.linkedin.com/in/yassir-tagemouati-471a6a2a2/"),
                   style="Social.TButton").pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        ttk.Button(social_frame, text="Instagram", command=lambda: webbrowser.open("https://www.instagram.com/yassir_tage/"),
                   style="Social.TButton").pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        ttk.Button(social_frame, text="Github", command=lambda: webbrowser.open("https://github.com/TageYassir"),
                   style="Social.TButton").pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        support_frame = tk.Frame(about_frame, bg=self.colors["light"])
        support_frame.pack(fill=tk.X, pady=10, padx=20)
        ttk.Button(support_frame, text="Support via PayPal",
                   command=lambda: webbrowser.open("https://paypal.me/YassirTagemouati?country.x=MA&locale.x=fr_XC"),
                   style="Accent.TButton").pack(pady=10, fill=tk.X)
        email_frame = tk.Frame(support_frame, bg=self.colors["light"])
        email_frame.pack(fill=tk.X, pady=5)
        tk.Label(email_frame, text="Support: ", font=("Helvetica", 11, "bold"),
                 bg=self.colors["light"], fg=self.colors["text"]).pack(side=tk.LEFT)
        email_link = tk.Label(email_frame, text="tagemouatiyassir@gmail.com",
                              font=("Helvetica", 11, "bold", "underline"),
                              bg=self.colors["light"], fg="blue", cursor="hand2")
        email_link.pack(side=tk.LEFT)
        email_link.bind("<Button-1>", lambda e: webbrowser.open("mailto:tagemouatiyassir@gmail.com"))
        ttk.Button(about_frame, text="Back to Main", command=self.show_main_app).pack(pady=10)
        debug_print("About screen displayed")

    # ======================
    # SETTINGS CALLBACKS
    # ======================
    def change_theme(self, event=None):
        debug_print("change_theme called")
        self.current_theme = self.theme_var.get()
        self.colors = THEMES[self.current_theme]
        self.preferences.set("theme", self.current_theme)
        self.configure_styles()
        self._refresh_summary()
        if hasattr(self, 'current_screen_method'):
            self.current_screen_method()
        else:
            self.show_main_app()

    def _refresh_summary(self):
        if hasattr(self, "settings_summary_var") and self.main_app_active:
            self.settings_summary_var.set(
                f"Scroll Speed: {self.scroll_speed}\nSmoothing: {self.smoothing:.2f}\nCircle Scale: {self.circle_scale:.1f}\nTheme: {self.current_theme}"
            )

    def _on_speed(self, val):
        self.scroll_speed = int(float(val))
        if hasattr(self, "speed_val_label") and self.speed_val_label.winfo_exists():
            self.speed_val_label.config(text=f"Value: {self.scroll_speed}")
        self.preferences.set("scroll_speed", self.scroll_speed)
        self._refresh_summary()

    def _on_smooth(self, val):
        self.smoothing = float(val)
        if hasattr(self, "smooth_val_label") and self.smooth_val_label.winfo_exists():
            self.smooth_val_label.config(text=f"Value: {self.smoothing:.2f}")
        self.preferences.set("smoothing", self.smoothing)
        self._refresh_summary()

    def _on_circle(self, val):
        self.circle_scale = float(val)
        if hasattr(self, "circle_val_label") and self.circle_val_label.winfo_exists():
            self.circle_val_label.config(text=f"Value: {self.circle_scale:.1f}")
        self.preferences.set("circle_scale", self.circle_scale)
        self._refresh_summary()

    # ======================
    # CONTROL START / STOP
    # ======================
    def start_control(self):
        debug_print("start_control called")
        self.control_active = True
        if hasattr(self, "start_btn") and self.start_btn.winfo_exists():
            self.start_btn.config(state=tk.DISABLED)
        if hasattr(self, "stop_btn") and self.stop_btn.winfo_exists():
            self.stop_btn.config(state=tk.NORMAL)

    def stop_control(self):
        debug_print("stop_control called")
        self.control_active = False
        if hasattr(self, "start_btn") and self.start_btn.winfo_exists():
            self.start_btn.config(state=tk.NORMAL)
        if hasattr(self, "stop_btn") and self.stop_btn.winfo_exists():
            self.stop_btn.config(state=tk.DISABLED)

    # ======================
    # VIDEO LOOP (with camera error handling)
    # ======================
    def video_loop(self):
        debug_print("Video loop started")
        while self.running:
            with self.cap_lock:
                cap = self.cap
            if cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue

            try:
                ret, frame = cap.read()
            except Exception as e:
                debug_print("Error reading frame:", e)
                ret = False

            if not ret:
                # Increment consecutive failure counter
                self._camera_failure_count += 1
                if self._camera_failure_count > self._camera_failure_threshold and not self._camera_error_shown:
                    # Schedule the error popup on the main thread
                    self.root.after(0, self.show_camera_error)
                time.sleep(0.1)
                continue
            else:
                # Successful read → reset the counter
                self._camera_failure_count = 0

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            tc = THEMES.get(self.current_theme, THEMES["Instagram"])

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            gesture = "NONE"
            if results.multi_hand_landmarks:
                hl = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=tc["hand_bgr"], thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=tc["hand_conn_bgr"], thickness=2),
                )
                gesture = detect_gesture(hl)
                wrist = hl.landmark[0]
                gx, gy = int(wrist.x * w), int(wrist.y * h)
                (tw, th), _ = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                cv2.rectangle(frame, (gx - tw // 2 - 6, gy - 38), (gx + tw // 2 + 6, gy - 38 + th + 8),
                              tc["main_bgr"], -1)
                cv2.putText(frame, gesture, (gx - tw // 2, gy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, tc["text_bgr"], 2)
                center, radius = self.finger_circle.update(hl, w, h, self.circle_scale, self.smoothing)
                draw_circle_on_frame(frame, center, radius, tc["circle_bgr"])
                now = time.time()
                if self.control_active and now - self.last_scroll_time > self.scroll_cooldown:
                    if gesture == "ONE FINGER":
                        pyautogui.scroll(-self.scroll_speed)
                        self.last_scroll_time = now
                        if center and radius:
                            cv2.putText(frame, "SCROLL DOWN", (center[0] - 90, center[1] + radius + 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, tc["accent_bgr"], 2)
                    elif gesture == "TWO FINGERS":
                        pyautogui.scroll(self.scroll_speed)
                        self.last_scroll_time = now
                        if center and radius:
                            cv2.putText(frame, "SCROLL UP", (center[0] - 65, center[1] - radius - 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, tc["accent_bgr"], 2)
            else:
                self.finger_circle.reset()

            with self.gesture_lock:
                self.current_gesture = gesture

            frame[:4, :] = tc["main_bgr"]
            cv2.putText(frame, f"Theme: {self.current_theme}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc["main_bgr"], 2)
            cv2.putText(frame, "SWCV", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            if self.control_active:
                cv2.putText(frame, "ACTIVE", (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            else:
                cv2.putText(frame, "PAUSED", (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
            cv2.putText(frame, "(c) SWCV 2026", (w - 130, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

            with self.frame_lock:
                self.current_frame = frame
            time.sleep(0.01)

    def update_video_label(self):
        if not self.running:
            return
        if self.show_camera and hasattr(self, "video_label") and self.video_label.winfo_exists():
            with self.frame_lock:
                frame = self.current_frame
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=pil_img)
                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk
        self.root.after(30, self.update_video_label)

    def quit_application(self):
        debug_print("quit_application called")
        self.running = False
        self.stop_control()
        time.sleep(0.1)
        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()
        self.hands.close()
        try:
            self.root.destroy()
        except Exception:
            pass
        if self._icon_path and os.path.exists(self._icon_path):
            try:
                os.unlink(self._icon_path)
            except Exception:
                pass
        debug_print("Application closed")

    def _hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))

if __name__ == "__main__":
    root = tk.Tk()
    app = SWCVApp(root)
    root.mainloop()