import threading
import time
import urllib.request
import webbrowser

import uvicorn


def open_browser_when_ready(url: str) -> None:
    """Poll the given URL until it responds, then open it in the browser."""
    for _ in range(100):  # try for ~20 seconds total
        try:
            with urllib.request.urlopen(url, timeout=0.5):
                pass
        except Exception:
            time.sleep(0.2)
            continue
        else:
            try:
                webbrowser.open(url)
            except Exception:
                pass
            return
    # If we get here, server never came up; just give up quietly.


def main() -> None:
    host = "0.0.0.0"   # listen on all interfaces (like before)
    port = 8000

    # For the browser we usually want localhost
    url = f"http://127.0.0.1:{port}/"

    # Start a helper thread that waits until "/" responds, then opens browser
    threading.Thread(
        target=open_browser_when_ready,
        args=(url,),
        daemon=True,
    ).start()

    # This blocks and runs the server in the main thread
    uvicorn.run("qudit_visualizer.app:app", host=host, port=port)


if __name__ == "__main__":
    main()