# ======================
# COOKIES.PY – User Preferences for SWCV
# ======================
import json
from pathlib import Path

class UserPreferences:
    def __init__(self):
        # Save preferences in the same directory as this script
        script_dir = Path(__file__).parent
        self.preferences_file = script_dir / "user_preferences.json"
        self.default_preferences = {
            "theme": "Instagram",
            "scroll_speed": 30,
            "smoothing": 0.7,
            "circle_scale": 1.5,
            "camera_index": 0
        }
        self.current_preferences = self.default_preferences.copy()
        self.load_preferences()

    def load_preferences(self):
        """Load user preferences from file"""
        try:
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r') as f:
                    loaded = json.load(f)
                    # Merge defaults with loaded values
                    self.current_preferences = {**self.default_preferences, **loaded}
                    return True
        except Exception as e:
            print(f"Error loading preferences: {e}")
        return False

    def save_preferences(self):
        """Save current preferences to file"""
        try:
            self.preferences_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.preferences_file, 'w') as f:
                json.dump(self.current_preferences, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving preferences: {e}")
        return False

    def get(self, key, default=None):
        """Get a preference value"""
        return self.current_preferences.get(key, default)

    def set(self, key, value):
        """Set a preference value and save immediately"""
        self.current_preferences[key] = value
        self.save_preferences()