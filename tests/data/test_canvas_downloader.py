import builtins
import sys
import types

import pytest

from my_krml_24999690.data.canvas_downloader import (
    download_canvas_courses,
)


def test_canvas_downloader_requires_optional_dependencies(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name in {"requests", "bs4", "canvasapi"}:
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"\[canvas\]"):
        download_canvas_courses(
            "https://canvas.example",
            "secret",
            output_dir=tmp_path,
        )


def test_canvas_downloader_handles_empty_course_list_without_network(
    monkeypatch,
    tmp_path,
):
    class FakeCanvas:
        def __init__(self, api_url, api_key):
            self.api_url = api_url
            self.api_key = api_key

        def get_courses(self, enrollment_state):
            return []

    requests = types.ModuleType("requests")
    requests.Session = object
    bs4 = types.ModuleType("bs4")
    bs4.BeautifulSoup = object
    canvasapi = types.ModuleType("canvasapi")
    canvasapi.Canvas = FakeCanvas

    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setitem(sys.modules, "bs4", bs4)
    monkeypatch.setitem(sys.modules, "canvasapi", canvasapi)

    progress = []
    result = download_canvas_courses(
        "https://canvas.example",
        "secret",
        output_dir=tmp_path,
        progress_cb=lambda done, total, message: progress.append(
            (done, total, message)
        ),
    )

    assert result == []
    assert progress[0][:2] == (0, 1)
    assert progress[-1][:2] == (1, 1)
