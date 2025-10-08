import click
from click.testing import CliRunner

from fitsmap.__main__ import cli


def test_cli_group():
    """Test that cli is a click Group."""
    assert isinstance(cli, click.core.Group)


def test_dir_command_invokes(monkeypatch):
    """Test the dir command calls convert.dir_to_map with correct args."""
    called = {}

    def fake_dir_to_map(
        directory,
        out_dir,
        title,
        task_procs,
        procs_per_task,
        catalog_delim,
        cat_wcs_fits_file,
    ):
        called.update(locals())

    monkeypatch.setattr("fitsmap.convert.dir_to_map", fake_dir_to_map)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dir",
            "input_dir",
            "--out_dir",
            "output_dir",
            "--title",
            "TestTitle",
            "--task_procs",
            "2",
            "--procs_per_task",
            "3",
            "--catalog_delim",
            ",",
            "--cat_wcs_fits_file",
            "wcs.fits",
        ],
    )
    assert result.exit_code == 0
    assert called["directory"] == "input_dir"
    assert called["out_dir"] == "output_dir"
    assert called["title"] == "TestTitle"
    assert called["task_procs"] == 2
    assert called["procs_per_task"] == 3
    assert called["catalog_delim"] == ","
    assert called["cat_wcs_fits_file"] == "wcs.fits"


def test_files_command_invokes(monkeypatch):
    """Test the files command calls convert.files_to_map with correct args."""
    called = {}

    def fake_files_to_map(
        files,
        out_dir,
        title,
        task_procs,
        procs_per_task,
        catalog_delim,
        cat_wcs_fits_file,
    ):
        called.update(locals())

    monkeypatch.setattr("fitsmap.convert.files_to_map", fake_files_to_map)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "files",
            "a.fits,b.fits,c.cat",
            "--out_dir",
            "output_dir",
            "--title",
            "TestTitle",
            "--task_procs",
            "2",
            "--procs_per_task",
            "3",
            "--catalog_delim",
            ";",
            "--cat_wcs_fits_file",
            "wcs.fits",
        ],
    )
    assert result.exit_code == 0
    assert called["files"] == ["a.fits", "b.fits", "c.cat"]
    assert called["out_dir"] == "output_dir"
    assert called["title"] == "TestTitle"
    assert called["task_procs"] == 2
    assert called["procs_per_task"] == 3
    assert called["catalog_delim"] == ";"
    assert called["cat_wcs_fits_file"] == "wcs.fits"


def test_serve_command(monkeypatch):
    """Test the serve command prints expected output and calls webbrowser if open_browser."""
    printed = []
    opened = []

    def fake_print(msg):
        printed.append(msg)

    def fake_webbrowser_open(address):
        opened.append(address)

    monkeypatch.setattr("builtins.print", fake_print)
    monkeypatch.setattr("webbrowser.open", fake_webbrowser_open)

    # Patch HTTPServer and ThreadPoolExecutor to avoid actually starting server
    class DummyServer:
        def serve_forever(self):
            pass

    monkeypatch.setattr("http.server.HTTPServer", lambda *a, **kw: DummyServer())

    class DummyPool:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def map(self, func, tasks):
            [func() for func in tasks]

    monkeypatch.setattr(
        "concurrent.futures.ThreadPoolExecutor", lambda max_workers: DummyPool()
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "serve",
            "--out_dir",
            "output_dir",
            "--port",
            "1234",
            "--open_browser",
            "True",
        ],
    )
    assert result.exit_code == 0
    assert any("Starting web server" in msg for msg in printed)
    assert opened and opened[0] == "http://localhost:1234"
