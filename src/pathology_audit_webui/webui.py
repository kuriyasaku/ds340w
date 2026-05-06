from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    host = "127.0.0.1"
    port = 7860
    print(f"Serving Pathology Audit Web UI at http://{host}:{port}/webui.html")
    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
