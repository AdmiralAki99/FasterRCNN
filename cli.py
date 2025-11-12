import shutil, time, sys

def _term_cols(default=100):
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default

def banner(title: str, width: int | None = None):
    width = width or _term_cols()
    line = "=" * max(8, min(width, 120))
    print(f"\n{line}\n{title}\n{line}")

class CLIProgressBar:
    """Minimal Keras-ish progress bar with metrics & ETA."""
    def __init__(self, total, bar_len=30):
        self.total = max(1, int(total))
        self.bar_len = bar_len
        self.start_time = None
        self.last_line_len = 0
        self.i = 0

    def start(self):
        self.start_time = time.time()
        self.i = 0

    def update(self, i, **metrics):
        self.i = i
        now = time.time()
        if self.start_time is None:
            self.start()
        elapsed = now - self.start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        remaining = max(0.0, (self.total - (i + 1)) / (rate + 1e-9))

        frac = (i + 1) / self.total
        fill = int(self.bar_len * frac)
        bar = "[" + "=" * max(0, fill - 1) + (">" if fill > 0 else "") + "." * (self.bar_len - fill) + "]"

        # format metrics like Keras: " - loss: 0.1234 - acc: 0.9876"
        parts = []
        for k, v in metrics.items():
            try:
                parts.append(f"- {k}: {float(v):.4f}")
            except Exception:
                parts.append(f"- {k}: {v}")

        line = f" {i+1:>4}/{self.total:<4} {bar} - ETA: {int(remaining)}s " + " ".join(parts)
        # ensure we overwrite previous line
        sys.stdout.write("\r" + line + " " * max(0, self.last_line_len - len(line)))
        sys.stdout.flush()
        self.last_line_len = len(line)

    def end(self):
        # move to next line cleanly
        sys.stdout.write("\n")
        sys.stdout.flush()
