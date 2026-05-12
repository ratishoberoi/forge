"""Symbol and dependency graph engine."""

from __future__ import annotations

import networkx as nx

from backend.repointel.models import CodeSymbol, ParsedFile


class SymbolGraphEngine:
    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()
        self._symbol_index: dict[str, CodeSymbol] = {}
        self._name_index: dict[str, list[str]] = {}

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    def upsert_parsed_file(self, parsed: ParsedFile) -> None:
        file_id = f"file:{parsed.file.path}"
        self._graph.add_node(file_id, type="file", path=parsed.file.path, language=parsed.file.language.value)
        for import_record in parsed.imports:
            import_id = f"import:{parsed.file.path}:{import_record.module}:{import_record.line}"
            self._graph.add_node(import_id, type="import", module=import_record.module)
            self._graph.add_edge(file_id, import_id, type="imports")

        for symbol in parsed.symbols:
            symbol_id = f"symbol:{symbol.id}"
            self._symbol_index[symbol.id] = symbol
            self._name_index.setdefault(symbol.name, []).append(symbol.id)
            self._graph.add_node(symbol_id, type="symbol", kind=symbol.kind.value, path=symbol.file_path, name=symbol.name)
            self._graph.add_edge(file_id, symbol_id, type="owns")
            if symbol.parent_symbol:
                self._graph.add_edge(f"symbol:{symbol.parent_symbol}", symbol_id, type="contains")
            for reference in symbol.references:
                for target_symbol_id in self._name_index.get(reference.name, []):
                    self._graph.add_edge(symbol_id, f"symbol:{target_symbol_id}", type="references")

    def remove_file(self, file_path: str) -> None:
        file_id = f"file:{file_path}"
        stale_symbol_ids = [
            symbol_id for symbol_id, symbol in self._symbol_index.items() if symbol.file_path == file_path
        ]
        for symbol_id in stale_symbol_ids:
            symbol = self._symbol_index.pop(symbol_id)
            if symbol.name in self._name_index:
                self._name_index[symbol.name] = [
                    existing_id for existing_id in self._name_index[symbol.name] if existing_id != symbol_id
                ]
                if not self._name_index[symbol.name]:
                    del self._name_index[symbol.name]
            graph_symbol_id = f"symbol:{symbol_id}"
            if graph_symbol_id in self._graph:
                self._graph.remove_node(graph_symbol_id)
        if file_id in self._graph:
            self._graph.remove_node(file_id)

    def neighbors(self, file_path: str, limit: int = 6) -> list[str]:
        file_id = f"file:{file_path}"
        if file_id not in self._graph:
            return []
        neighbors: list[str] = []
        for node in self._graph.neighbors(file_id):
            node_data = self._graph.nodes[node]
            neighbors.append(node_data.get("path") or node_data.get("name") or str(node))
            if len(neighbors) >= limit:
                break
        return neighbors

    def symbols_for_paths(self, paths: list[str]) -> list[CodeSymbol]:
        path_set = set(paths)
        return [symbol for symbol in self._symbol_index.values() if symbol.file_path in path_set]
