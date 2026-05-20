import pytest
from backend.runtime.patch_parser import PatchParser


# ── extract_code ──────────────────────────────────────────────────────────────

def test_extract_code_block():
    parser = PatchParser()
    text = """
Explanation

```python
print("hello")
```

"""
    result = parser.extract_code(text)
    assert result == 'print("hello")'


def test_extract_plain_fence():
    parser = PatchParser()
    text = "```\ndef hello(): pass\n```"
    result = parser.extract_code(text)
    assert result == "def hello(): pass"

def test_extract_plain_text():
    parser = PatchParser()
    result = parser.extract_code("print('x')")
    assert result == "print('x')"


def test_extract_structured_json_file_content():
    parser = PatchParser()
    text = """
    {
      "summary": "Add typing",
      "reasoning": "Needed",
      "risk": "low",
      "files": {
        "hello.py": "def hello(name: str) -> str:\\n    return name"
      }
    }
    """
    result = parser.extract_file(text, "hello.py")
    assert result == "def hello(name: str) -> str:\n    return name"


def test_extract_empty_returns_empty():
    parser = PatchParser()
    assert parser.extract_code("") == ""
    assert parser.extract_code("   ") == ""


def test_extract_first_block_only():
    parser = PatchParser()
    text = """
```python
first_block = True
```

```python
second_block = True
```
"""
    result = parser.extract_code(text)
    assert "first_block" in result
    assert "second_block" not in result


def test_extract_strips_whitespace():
    parser = PatchParser()
    text = "```python\n\n  def hello():   pass\n\n```"
    result = parser.extract_code(text)
    assert result == "def hello():   pass"


def test_extract_multiline_block():
    parser = PatchParser()
    text = """
```python
def hello(name: str) -> str:
    return "hello " + name
```
"""
    result = parser.extract_code(text)
    assert "def hello" in result
    assert "return" in result


def test_extract_prose_before_code_ignored():
    parser = PatchParser()
    text = (
        "Sure! Here's the updated file:\n\n"
        "```python\n"
        "x = 1\n"
        "```"
    )
    result = parser.extract_code(text)
    assert result == "x = 1"
    assert "Sure" not in result


def test_extract_prose_after_code_ignored():
    parser = PatchParser()
    text = (
        "```python\n"
        "x = 1\n"
        "```\n\n"
        "Let me know if you need changes."
    )
    result = parser.extract_code(text)
    assert result == "x = 1"
    assert "Let me know" not in result


# ── extract_all_blocks ────────────────────────────────────────────────────────

def test_extract_all_blocks_multiple():
    parser = PatchParser()
    text = """
```python
block_one = 1
```

```python
block_two = 2
```
"""
    blocks = parser.extract_all_blocks(text)
    assert len(blocks) == 2
    assert "block_one" in blocks[0]
    assert "block_two" in blocks[1]


def test_extract_all_blocks_empty():
    parser = PatchParser()
    blocks = parser.extract_all_blocks("no code here")
    assert blocks == []


def test_extract_all_blocks_single():
    parser = PatchParser()
    text = "```python\nx = 1\n```"
    blocks = parser.extract_all_blocks(text)
    assert len(blocks) == 1
    assert blocks[0] == "x = 1"


# ── has_code_block ────────────────────────────────────────────────────────────

def test_has_code_block_true():
    parser = PatchParser()
    assert parser.has_code_block("```python\nx = 1\n```") is True


def test_has_code_block_false():
    parser = PatchParser()
    assert parser.has_code_block("just plain text") is False


# ── strip_preamble ────────────────────────────────────────────────────────────

def test_strip_preamble_removes_prose():
    parser = PatchParser()
    text = (
        "Great question! Here is the implementation:\n\n"
        "```python\ndef foo(): pass\n```"
    )
    result = parser.strip_preamble(text)
    assert result == "def foo(): pass"
    assert "Great question" not in result


def test_strip_preamble_no_fence_returns_text():
    parser = PatchParser()
    result = parser.strip_preamble("just plain text")
    assert result == "just plain text"
