"""Usage limits for the built-in demo key, so no single visitor can use up the shared free quota."""
from datetime import date
from threading import Lock

SESSION_LIMITS = {"pdfs": 3, "questions": 25}
DAILY_LIMITS = {"pdfs": 40, "questions": 300}
# Each passage costs one embedding request, so cap PDF size on the shared key.
DEMO_MAX_CHUNKS = 150


class DailyLimit:
    """Counts uses across all visitors. Resets each day, and when the app restarts (memory only)."""
    def __init__(self, limits: dict[str, int], today=date.today):
        self.limits = limits
        self.today = today
        self.lock = Lock()  # Streamlit serves visitors from parallel threads.
        self.day = None
        self.counts = {}

    def take(self, kind: str) -> bool:
        """Count one use and return True, or return False if today's limit is reached."""
        with self.lock:
            if self.day != self.today():
                self.day, self.counts = self.today(), {}
            if self.counts.get(kind, 0) >= self.limits[kind]:
                return False
            self.counts[kind] = self.counts.get(kind, 0) + 1
            return True
