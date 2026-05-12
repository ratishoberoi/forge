from backend.llm.decoding import UtfSafeStreamAssembler, decode_token_ids


class FakeTokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        mapping = {
            1: "def",
            2: "Ġhello",
            3: "Ċ",
            4: "return",
            5: "Ġx",
        }
        text = "".join(mapping[token_id] for token_id in token_ids)
        if clean_up_tokenization_spaces:
            text = text.replace("Ġ", " ").replace("Ċ", "\n")
        return text


def test_decode_token_ids_removes_tokenizer_artifacts() -> None:
    tokenizer = FakeTokenizer()
    decoded = decode_token_ids(tokenizer, [1, 2, 3, 4, 5], clean_up_spaces=True)
    assert decoded == "def hello\nreturn x"


def test_utf_safe_stream_assembler_emits_incremental_text() -> None:
    tokenizer = FakeTokenizer()
    assembler = UtfSafeStreamAssembler(tokenizer)
    assert assembler.push([1]) == "def"
    assert assembler.push([1, 2]) == " hello"
    assert assembler.push([1, 2, 3]) == "\n"
