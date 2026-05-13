from pathlib import Path
import pytest
from backend.runtime.file_editor import FileEditor


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    return tmp_path / "repo"


@pytest.fixture()
def editor(repo_root: Path) -> FileEditor:
    repo_root.mkdir()
    return FileEditor(repo_root=str(repo_root))


def test_write_new_file(editor: FileEditor):
    result = editor.write_file("hello.py", "print('hello')\n")
    assert result.existed_before is False
    assert result.original_content is None
    assert result.path.exists()
    assert result.path.read_text() == "print('hello')\n"


def test_overwrite_existing_file(editor: FileEditor):
    editor.write_file("test.py", "v1")
    result = editor.write_file("test.py", "v2")
    assert result.existed_before is True
    assert result.original_content == "v1"
    assert result.updated_content == "v2"


def test_read_file(editor: FileEditor):
    editor.write_file("readme.txt", "hello world")
    assert editor.read_file("readme.txt") == "hello world"


def test_prevent_path_escape(editor: FileEditor):
    with pytest.raises(ValueError):
        editor.write_file("../../../etc/passwd", "bad")


def test_nested_directory_write(editor: FileEditor):
    result = editor.write_file("a/b/c/test.py", "print(1)")
    assert result.path.exists()
    assert result.path.read_text() == "print(1)"


def test_delete_existing_file(editor: FileEditor):
    editor.write_file("del.py", "x = 1")
    result = editor.delete_file("del.py")
    assert result is True
    assert not (editor.repo_root / "del.py").exists()


def test_delete_nonexistent_file(editor: FileEditor):
    assert editor.delete_file("ghost.py") is False


def test_delete_prevents_path_escape(editor: FileEditor):
    with pytest.raises(ValueError):
        editor.delete_file("../../../etc/passwd")