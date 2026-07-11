#!/usr/bin/env python3
"""Range-capable HTTP server for the stable OpenARM tailnet artifact URL."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = os.fstat(source.fileno()).st_size
        start, end = 0, size - 1
        value = self.headers.get("Range")
        if value and value.startswith("bytes="):
            first, _, last = value[6:].partition("-")
            try:
                start = int(first) if first else 0
                end = min(int(last), end) if last else end
            except ValueError:
                source.close()
                self.send_error(400, "Invalid Range")
                return None
            if start > end or start >= size:
                source.close()
                self.send_error(416, "Range Not Satisfiable")
                return None
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(os.fstat(source.fileno()).st_mtime))
        self.end_headers()
        source.seek(start)
        self._remaining = end - start + 1
        return source

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()

    def handler(*handler_args, **handler_kwargs):
        return RangeHandler(*handler_args, directory=str(args.directory), **handler_kwargs)

    ThreadingHTTPServer((args.bind, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
