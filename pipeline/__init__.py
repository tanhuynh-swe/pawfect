"""Pawfect Love Animals — YouTube production pipeline."""
import os
import os.path

__version__ = "1.1.0"

# launchd starts scheduled jobs with a minimal PATH (/usr/bin:/bin:/usr/sbin:
# /sbin), which excludes Homebrew. That makes ffmpeg invisible to the
# scheduler even though it works perfectly in an interactive shell — the job
# then fails at the first narration step with "No such file or directory".
# Putting the usual install locations back on PATH fixes it for every
# subprocess this package starts.
_EXTRA_BIN = [
    "/opt/homebrew/bin",      # Homebrew on Apple Silicon
    "/opt/homebrew/sbin",
    "/usr/local/bin",         # Homebrew on Intel, and most manual installs
    "/opt/local/bin",         # MacPorts
]
_present = [p for p in _EXTRA_BIN if os.path.isdir(p)]
if _present:
    _current = os.environ.get("PATH", "")
    _missing = [p for p in _present if p not in _current.split(os.pathsep)]
    if _missing:
        os.environ["PATH"] = os.pathsep.join(_missing + ([_current] if _current else []))

# launchd also starts without the certificate environment a GUI shell has, so
# HTTPS downloads (the Piper voice model) can fail with CERTIFICATE_VERIFY_FAILED.
if not os.environ.get("SSL_CERT_FILE"):
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:
        pass
