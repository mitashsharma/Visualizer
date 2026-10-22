"""Application entry point.

    python app.py                 # http://localhost:8000
    python app.py --port 8501     # choose another port
    python app.py --debug         # auto-reload while developing

The same app object works with a WSGI server:
    flask --app app run
    gunicorn "app:app" --workers 1 --threads 4
"""

from __future__ import annotations

import argparse
import os

from src.api import create_app

app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Data Visualizer")
    parser.add_argument("--host", default=os.environ.get("AIVIZ_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIVIZ_PORT", "8000")))
    parser.add_argument("--debug", action="store_true", help="enable the reloader and debugger")
    args = parser.parse_args()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        use_reloader=args.debug,
    )


if __name__ == "__main__":
    main()
