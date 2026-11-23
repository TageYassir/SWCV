# cookies.py – User Preferences for SWCV
import json
import os

class UserPreferences:
    def __init__(self):
        # Store preferences in the user's home directory – cross‑platform and safe
        self.preferences_file = os.path.join(
            os.path.expanduser("~"), ".swcv_preferences.json"
        )
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
        """Load user preferences from file, merge with defaults."""
        try:
            if os.path.exists(self.preferences_file):
                with open(self.preferences_file, "r") as f:
                    loaded = json.load(f)
                # Merge: keep defaults for missing keys
                self.current_preferences = {**self.default_preferences, **loaded}
                return True
        except Exception as e:
            print(f"Error loading preferences: {e}")
        return False

    def save_preferences(self):
        """Save current preferences to file."""
        try:
            # Ensure the directory exists (home always exists, but just in case)
            os.makedirs(os.path.dirname(self.preferences_file), exist_ok=True)
            with open(self.preferences_file, "w") as f:
                json.dump(self.current_preferences, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving preferences: {e}")
        return False

    def get(self, key, default=None):
        return self.current_preferences.get(key, default)

    def set(self, key, value):
        self.current_preferences[key] = value
        self.save_preferences()