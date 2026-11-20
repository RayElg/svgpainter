"""Local static server for the SVG Painter web demo.

Usage:
    python3 serve.py [port]

Sends the Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers
required for multi-threaded WebAssembly (SharedArrayBuffer).
"""
import http.server
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'wasm-unsafe-eval' blob:; "
            "worker-src 'self' blob:; "
            "style-src 'self'; "
            "img-src 'self' blob: data:; "
            "connect-src 'self' blob:; "
            "object-src 'none'; "
            "frame-ancestors 'none'",
        )
        if self.path.startswith("/vendor/") or self.path.startswith("/model/"):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving SVG Painter at http://127.0.0.1:{port}/")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
