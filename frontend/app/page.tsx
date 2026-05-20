"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Archive,
  Bot,
  Brain,
  CheckCircle2,
  Clock,
  Code2,
  Database,
  FileCode2,
  FolderOpen,
  GitBranch,
  GitCompare,
  ListChecks,
  Network,
  Pause,
  Play,
  RefreshCw,
  Search,
  Square,
  Terminal,
  XCircle
} from "lucide-react";
import {
  ArtifactSummary,
  ApiRequestError,
  ControlSnapshot,
  RepoTreeNode,
  browseWorkspace,
  commitRepository,
  controlRun,
  createRun,
  fetchDashboardHealth,
  fetchRepoFile,
  fetchRepoTree,
  fetchRunReplay,
  fetchSnapshot,
  importRepository,
  refreshRepository,
  rollbackRepository,
  runBenchmarks,
  switchRepository,
  validateRepositoryPath
} from "@/lib/api";
import { Badge, Button, Card, CardHeader, Input, Textarea } from "@/components/ui";
import { cn } from "@/lib/utils";

const defaultRepo = "/home/ratish/Forge";

export default function ControlCenterPage() {
  const [repositoryRoot, setRepositoryRoot] = useState(defaultRepo);
  const [targetFile, setTargetFile] = useState("app.py");
  const [objective, setObjective] = useState("Build a calculator app");
  const [iterations, setIterations] = useState(3);
  const [testCommand, setTestCommand] = useState("pytest -q");
  const [execute, setExecute] = useState(true);
  const [repositoryImportPath, setRepositoryImportPath] = useState(defaultRepo);
  const [workspaceBrowser, setWorkspaceBrowser] = useState<Record<string, unknown> | null>(null);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [commitMessage, setCommitMessage] = useState("Forge autonomous changes");
  const [snapshot, setSnapshot] = useState<ControlSnapshot | null>(null);
  const [tree, setTree] = useState<RepoTreeNode | null>(null);
  const [selectedFile, setSelectedFile] = useState<{ path: string; content: string } | null>(null);
  const [artifactQuery, setArtifactQuery] = useState("");
  const [artifactRole, setArtifactRole] = useState("ALL");
  const [selectedArtifact, setSelectedArtifact] = useState<ArtifactSummary | null>(null);
  const [replayEvents, setReplayEvents] = useState<Array<Record<string, unknown>>>([]);
  const [benchmarkSummary, setBenchmarkSummary] = useState<Record<string, unknown> | null>(null);
  const [health, setHealth] = useState<"checking" | "ok" | "degraded" | "failed">("checking");
  const [error, setError] = useState<string | null>(null);
  const [apiStats, setApiStats] = useState({
    snapshotLatency: 0,
    healthLatency: 0,
    failures: 0,
    lastRefresh: ""
  });
  const activeRun = snapshot?.active_run ?? null;
  const loadedRepositoryRoot = snapshot?.active_repository_root || repositoryRoot;

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let delay = 2500;
    async function load() {
      const started = performance.now();
      try {
        const next = await fetchSnapshot(repositoryRoot);
        if (!cancelled) {
          setSnapshot(next);
          if (next.active_repository_root && next.active_repository_root !== repositoryRoot) {
            setRepositoryRoot(next.active_repository_root);
          }
          setError(null);
          delay = 2500;
          setApiStats((stats) => ({
            ...stats,
            snapshotLatency: Math.round(performance.now() - started),
            failures: 0,
            lastRefresh: new Date().toLocaleTimeString()
          }));
        }
      } catch (err) {
        if (!cancelled) {
          delay = Math.min(delay * 2, 20000);
          const detail = err instanceof ApiRequestError
            ? `${err.message} (${err.diagnostics.duration_ms}ms, attempts ${err.diagnostics.attempts})`
            : err instanceof Error ? err.message : String(err);
          setError(detail);
          setApiStats((stats) => ({
            ...stats,
            snapshotLatency: Math.round(performance.now() - started),
            failures: stats.failures + 1
          }));
        }
      } finally {
        if (!cancelled) timer = window.setTimeout(load, delay);
      }
    }
    load();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [repositoryRoot]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let failures = 0;
    async function loadHealth() {
      const started = performance.now();
      try {
        await fetchDashboardHealth();
        if (!cancelled) {
          failures = 0;
          setHealth("ok");
          setApiStats((stats) => ({ ...stats, healthLatency: Math.round(performance.now() - started) }));
        }
      } catch {
        if (!cancelled) {
          failures += 1;
          setHealth(failures > 2 ? "failed" : "degraded");
          setApiStats((stats) => ({ ...stats, healthLatency: Math.round(performance.now() - started), failures: stats.failures + 1 }));
        }
      } finally {
        if (!cancelled) timer = window.setTimeout(loadHealth, failures ? Math.min(5000 * 2 ** failures, 30000) : 5000);
      }
    }
    loadHealth();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    browseWorkspace(defaultRepo)
      .then((result) => {
        if (!cancelled) {
          setWorkspaceBrowser(result);
          setWorkspaceError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) setWorkspaceError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadTree() {
      try {
        const next = await fetchRepoTree(loadedRepositoryRoot);
        if (!cancelled) setTree(next);
      } catch {
        if (!cancelled) setTree(null);
      }
    }
    loadTree();
    return () => {
      cancelled = true;
    };
  }, [loadedRepositoryRoot]);

  const filteredArtifacts = useMemo(() => {
    const artifacts = snapshot?.artifacts ?? [];
    return artifacts.filter((artifact) => {
      const roleMatch = artifactRole === "ALL" || artifact.role === artifactRole;
      const query = artifactQuery.trim().toLowerCase();
      const queryMatch =
        !query ||
        artifact.content.toLowerCase().includes(query) ||
        artifact.task.toLowerCase().includes(query);
      return roleMatch && queryMatch;
    });
  }, [artifactQuery, artifactRole, snapshot]);

  async function startRun() {
    const command = testCommand.split(" ").filter(Boolean);
    const run = await createRun({
      objective,
      repository_root: repositoryRoot,
      repository_id: stringValue(snapshot?.active_repository?.repository_id) || null,
      target_file: targetFile,
      test_command: command.length ? command : ["pytest", "-q"],
      max_iterations: iterations,
      execute
    });
    setSnapshot(await fetchSnapshot(run.repository_root));
  }

  async function runAction(action: "pause" | "resume" | "stop") {
    if (!activeRun) return;
    await controlRun(activeRun.id, action);
    setSnapshot(await fetchSnapshot(repositoryRoot));
  }

  async function openFile(path: string) {
    const file = await fetchRepoFile(repositoryRoot, path);
    setSelectedFile({ path: file.path, content: file.content });
  }

  async function importCurrentRepository() {
    setWorkspaceError(null);
    const validation = await validateRepositoryPath(repositoryImportPath);
    if (!validation.valid) {
      setWorkspaceError(stringValue(validation.error) || "Repository path is not valid.");
      return;
    }
    const record = await importRepository({ path: repositoryImportPath, set_active: true, refresh_intelligence: true });
    const nextRoot = stringValue(record.repository_path);
    if (nextRoot) {
      setRepositoryRoot(nextRoot);
      setWorkspaceBrowser(await browseWorkspace(nextRoot));
    }
    setSnapshot(await fetchSnapshot(nextRoot || repositoryRoot));
  }

  async function browseDirectory(path?: string) {
    setWorkspaceError(null);
    const result = await browseWorkspace(path);
    setWorkspaceBrowser(result);
    const current = stringValue(result.current);
    if (current) setRepositoryImportPath(current);
  }

  async function selectRepository(repositoryId: string) {
    const record = await switchRepository(repositoryId);
    const nextRoot = stringValue(record.repository_path);
    if (nextRoot) setRepositoryRoot(nextRoot);
    setSnapshot(await fetchSnapshot(nextRoot || repositoryRoot));
  }

  async function refreshActiveRepository() {
    const repositoryId = stringValue(snapshot?.active_repository?.repository_id);
    if (!repositoryId) return;
    await refreshRepository(repositoryId);
    setSnapshot(await fetchSnapshot(repositoryRoot));
  }

  async function commitActiveRepository() {
    const repositoryId = stringValue(snapshot?.active_repository?.repository_id);
    await commitRepository({
      repository_id: repositoryId || undefined,
      repository_root: repositoryId ? undefined : repositoryRoot,
      message: commitMessage
    });
    setSnapshot(await fetchSnapshot(repositoryRoot));
  }

  async function rollbackActiveRepository() {
    const repositoryId = stringValue(snapshot?.active_repository?.repository_id);
    await rollbackRepository({
      repository_id: repositoryId || undefined,
      repository_root: repositoryId ? undefined : repositoryRoot,
      target: "HEAD",
      clean_untracked: false
    });
    setSnapshot(await fetchSnapshot(repositoryRoot));
  }

  async function openReplay(runId: string) {
    const replay = await fetchRunReplay(runId);
    setReplayEvents(replay.events);
  }

  async function runIsolatedBenchmarks() {
    const result = await runBenchmarks({ root: ".forge/benchmarks", cleanup: true });
    setBenchmarkSummary(objectValue(result.summary));
  }

  return (
    <main className="min-h-screen overflow-x-hidden bg-background px-4 py-4">
      <div className="mx-auto flex max-w-[1800px] min-w-0 flex-col gap-5">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-md border border-border bg-panel px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-md bg-cyan-950 text-accent">
              <Terminal size={20} />
            </div>
            <div>
            <h1 className="text-2xl font-semibold tracking-normal">Forge Control Center</h1>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-muted">
              <Badge tone={activeRun ? "accent" : "neutral"}>{activeRun?.status ?? "idle"}</Badge>
              <Badge tone={health === "ok" ? "success" : health === "failed" ? "danger" : "warning"}>
                backend {health}
              </Badge>
              <span>{snapshot?.generated_at ? new Date(snapshot.generated_at).toLocaleTimeString() : "waiting"}</span>
              <span>snapshot {apiStats.snapshotLatency}ms</span>
              <span>health {apiStats.healthLatency}ms</span>
              <span>last {apiStats.lastRefresh || "pending"}</span>
            </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="secondary" onClick={() => fetchSnapshot(repositoryRoot).then(setSnapshot)}>
              <RefreshCw size={16} />
              Refresh
            </Button>
          </div>
        </header>

        {error ? (
          <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-danger">
            {error}
          </div>
        ) : null}

        <div className="grid min-w-0 grid-cols-1 gap-4 xl:grid-cols-[340px_minmax(0,1fr)] 2xl:grid-cols-[360px_minmax(0,1fr)_400px]">
          <section className="flex min-w-0 flex-col gap-5">
            <WorkspaceBrowser
              repositories={snapshot?.repositories ?? []}
              activeRepository={snapshot?.active_repository}
              importPath={repositoryImportPath}
              setImportPath={setRepositoryImportPath}
              browser={workspaceBrowser}
              browserError={workspaceError}
              onBrowse={browseDirectory}
              onImport={importCurrentRepository}
              onSelect={selectRepository}
              onRefresh={refreshActiveRepository}
            />
            <CommandCenter
              objective={objective}
              setObjective={setObjective}
              repositoryRoot={repositoryRoot}
              setRepositoryRoot={setRepositoryRoot}
              targetFile={targetFile}
              setTargetFile={setTargetFile}
              iterations={iterations}
              setIterations={setIterations}
              testCommand={testCommand}
              setTestCommand={setTestCommand}
              execute={execute}
              setExecute={setExecute}
              activeRun={activeRun}
              onStart={startRun}
              onPause={() => runAction("pause")}
              onResume={() => runAction("resume")}
              onStop={() => runAction("stop")}
            />
            <RepositorySummary
              summary={snapshot?.repository_summary}
              architecture={snapshot?.architecture_summary}
              plan={snapshot?.execution_plan}
            />
            <ArchitectureMemoryPanel memory={snapshot?.architecture_memory} compressed={snapshot?.compressed_context} />
            <ProjectBrainPanel brain={snapshot?.project_brain} semantic={snapshot?.semantic_memory} />
            <ConvergencePanel convergence={snapshot?.convergence} />
            <RuntimeMonitor runtime={snapshot?.runtime} />
            <TestResults tests={snapshot?.tests} />
            <ProductionValidationPanel
              bootstrap={snapshot?.bootstrap}
              acceptance={snapshot?.acceptance}
              build={snapshot?.build_validation}
              visual={snapshot?.visual_validation}
              quality={snapshot?.quality_score}
              releaseReport={snapshot?.release_report}
              benchmarkSummary={benchmarkSummary}
              onBenchmark={runIsolatedBenchmarks}
            />
          </section>

          <section className="flex min-w-0 flex-col gap-5">
            <LiveCourtroom roles={snapshot?.courtroom ?? []} />
            <ExecutionTimeline timeline={snapshot?.timeline ?? []} />
            <TaskGraphPanel taskPlan={snapshot?.task_plan} executionGraph={snapshot?.execution_graph} />
            <ContextAssemblyPanel
              assembly={snapshot?.context_assembly}
              repositoryRag={snapshot?.repository_rag}
              knowledgeGraph={snapshot?.knowledge_graph}
              adrs={snapshot?.adrs ?? []}
            />
            <GitPanel
              git={snapshot?.git}
              commitMessage={commitMessage}
              setCommitMessage={setCommitMessage}
              onCommit={commitActiveRepository}
              onRollback={rollbackActiveRepository}
            />
            <PatchViewer patch={snapshot?.patch} />
            <LogsPanel logs={snapshot?.logs ?? []} />
          </section>

          <section className="flex min-w-0 flex-col gap-5">
            <ArtifactExplorer
              artifacts={filteredArtifacts}
              query={artifactQuery}
              setQuery={setArtifactQuery}
              role={artifactRole}
              setRole={setArtifactRole}
              selected={selectedArtifact}
              setSelected={setSelectedArtifact}
            />
            <RunHistoryPanel
              runs={snapshot?.run_history ?? []}
              queued={snapshot?.queued_tasks ?? []}
              replayEvents={replayEvents}
              onReplay={openReplay}
            />
            <ObjectiveMemoryPanel objectives={snapshot?.objective_memory ?? []} />
            <ToolActivityPanel tools={snapshot?.tool_activity} />
            <RepositoryExplorer tree={tree} selectedFile={selectedFile} onOpenFile={openFile} />
            <ConversationView items={snapshot?.conversation ?? []} />
          </section>
        </div>
      </div>
    </main>
  );
}

function WorkspaceBrowser(props: {
  repositories: Array<Record<string, unknown>>;
  activeRepository?: Record<string, unknown> | null;
  importPath: string;
  setImportPath: (value: string) => void;
  browser?: Record<string, unknown> | null;
  browserError?: string | null;
  onBrowse: (path?: string) => void;
  onImport: () => void;
  onSelect: (repositoryId: string) => void;
  onRefresh: () => void;
}) {
  const entries = Array.isArray(props.browser?.entries)
    ? (props.browser.entries as Array<Record<string, unknown>>)
    : [];
  const roots = asList(props.browser?.roots);
  const parent = stringValue(props.browser?.parent);
  return (
    <Card>
      <CardHeader title="Workspace Browser" action={<GitBranch size={18} className="text-accent" />}>
        {stringValue(props.activeRepository?.repository_name) || "no repository selected"}
      </CardHeader>
      <div className="space-y-3 p-4">
        <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto_auto] gap-2">
          <Input value={props.importPath} onChange={(event) => props.setImportPath(event.target.value)} />
          <Button variant="secondary" onClick={() => props.onBrowse(props.importPath)}>
            <FolderOpen size={16} />
            Browse
          </Button>
          <Button variant="secondary" onClick={props.onImport}>
            <Archive size={16} />
            Import
          </Button>
        </div>
        {props.browserError ? (
          <div className="rounded-md border border-red-900 bg-red-950 px-3 py-2 text-xs text-danger">
            {props.browserError}
          </div>
        ) : null}
        <div className="rounded-md border border-border bg-slate-950">
          <div className="flex min-w-0 items-center justify-between gap-2 border-b border-border px-3 py-2 text-xs">
            <span className="truncate text-muted">{stringValue(props.browser?.current) || "Select a local path"}</span>
            <div className="flex shrink-0 items-center gap-2">
              {parent ? (
                <Button variant="ghost" className="h-7 px-2 text-xs" onClick={() => props.onBrowse(parent)}>
                  Up
                </Button>
              ) : null}
              <Badge tone={props.browser?.valid_repository ? "success" : "neutral"}>
                {props.browser?.valid_repository ? "valid" : "folder"}
              </Badge>
            </div>
          </div>
          {roots.length ? (
            <div className="flex gap-2 overflow-x-auto border-b border-border px-3 py-2">
              {roots.map((root) => (
                <Button key={root} variant="ghost" className="h-7 shrink-0 px-2 text-xs" onClick={() => props.onBrowse(root)}>
                  {root}
                </Button>
              ))}
            </div>
          ) : null}
          <div className="max-h-48 overflow-auto">
            {entries.length ? (
              entries.map((entry) => {
                const path = stringValue(entry.path);
                return (
                  <button
                    key={path}
                    className="block w-full border-b border-border px-3 py-2 text-left text-sm last:border-b-0 hover:bg-slate-900"
                    onClick={() => path && props.onBrowse(path)}
                  >
                    <div className="flex min-w-0 items-center justify-between gap-2">
                      <span className="truncate">{stringValue(entry.name)}</span>
                      <Badge tone={entry.is_git_repository || entry.has_app_markers ? "accent" : "neutral"}>
                        {entry.is_git_repository ? "git" : entry.has_app_markers ? "app" : "dir"}
                      </Badge>
                    </div>
                    <div className="mt-1 truncate text-xs text-muted">{path}</div>
                  </button>
                );
              })
            ) : (
              <div className="px-3 py-2 text-sm text-muted">No child directories available.</div>
            )}
          </div>
        </div>
        <div className="max-h-44 overflow-auto rounded-md border border-border bg-slate-950">
          {props.repositories.length ? (
            props.repositories.map((repository) => {
              const repositoryId = stringValue(repository.repository_id);
              const active = repositoryId === stringValue(props.activeRepository?.repository_id);
              return (
                <button
                  key={repositoryId}
                  className={cn(
                    "block w-full border-b border-border px-3 py-2 text-left text-sm last:border-b-0 hover:bg-slate-900",
                    active && "bg-cyan-950"
                  )}
                  onClick={() => repositoryId && props.onSelect(repositoryId)}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-medium">{stringValue(repository.repository_name)}</span>
                    <Badge tone={active ? "accent" : "neutral"}>{stringValue(repository.repository_type)}</Badge>
                  </div>
                  <div className="mt-1 truncate text-xs text-muted">{stringValue(repository.repository_path)}</div>
                </button>
              );
            })
          ) : (
            <div className="px-3 py-2 text-sm text-muted">No repositories registered.</div>
          )}
        </div>
        <Button variant="secondary" onClick={props.onRefresh} disabled={!props.activeRepository}>
          <RefreshCw size={16} />
          Refresh Intelligence
        </Button>
      </div>
    </Card>
  );
}

function RepositorySummary({
  summary,
  architecture,
  plan
}: {
  summary?: Record<string, unknown> | null;
  architecture?: string | null;
  plan?: Record<string, unknown> | null;
}) {
  const filesToCreate = asList(plan?.files_to_create);
  const filesToModify = asList(plan?.files_to_modify);
  const expectedTests = asList(plan?.expected_tests);
  return (
    <Card>
      <CardHeader title="Repository Intelligence" action={<ListChecks size={18} className="text-accent" />}>
        {architecture ?? "scan pending"}
      </CardHeader>
      <div className="space-y-3 p-4 text-sm">
        <dl className="space-y-2">
          <Metric label="language" value={summary?.primary_language ?? "unknown"} />
          <Metric label="frameworks" value={asList(summary?.frameworks).join(", ") || "none"} />
          <Metric label="package manager" value={asList(summary?.package_managers).join(", ") || "none"} />
          <Metric label="tests" value={asList(summary?.test_frameworks).join(", ") || "none"} />
          <Metric label="entrypoints" value={asList(summary?.entrypoints).join(", ") || "none"} />
        </dl>
        <div className="grid gap-2">
          <PlanList title="Files to create" items={filesToCreate} />
          <PlanList title="Files to modify" items={filesToModify} />
          <PlanList title="Expected tests" items={expectedTests} />
        </div>
      </div>
    </Card>
  );
}

function PlanList({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="rounded-md border border-border bg-slate-950 p-2">
      <div className="mb-1 text-xs font-medium text-muted">{title}</div>
      {items.length ? (
        <ul className="space-y-1">
          {items.map((item) => (
            <li key={item} className="truncate text-xs">{item}</li>
          ))}
        </ul>
      ) : (
        <div className="text-xs text-muted">none</div>
      )}
    </div>
  );
}

function MiniList({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="rounded-md border border-border bg-slate-950 p-2">
      <div className="mb-1 text-xs font-medium text-muted">{title}</div>
      {items.length ? (
        <ul className="space-y-1">
          {items.slice(0, 8).map((item, index) => (
            <li key={`${title}-${index}`} className="truncate text-xs">{item}</li>
          ))}
        </ul>
      ) : (
        <div className="text-xs text-muted">none</div>
      )}
    </div>
  );
}

function ArchitectureMemoryPanel({
  memory,
  compressed
}: {
  memory?: Record<string, unknown> | null;
  compressed?: Record<string, unknown> | null;
}) {
  const modules = asList(memory?.important_modules);
  const boundaries = asList(memory?.service_boundaries);
  const modified = asList(memory?.previously_modified_files);
  const selected = asList(compressed?.selected_files);
  return (
    <Card>
      <CardHeader title="Architecture Memory" action={<Activity size={18} className="text-accent" />}>
        {stringValue(memory?.updated_at) || "memory pending"}
      </CardHeader>
      <div className="space-y-3 p-4 text-sm">
        <p className="text-muted">{stringValue(memory?.architecture_summary) || "No architecture memory yet."}</p>
        <div className="grid gap-2">
          <MiniList title="Important Modules" items={modules} />
          <MiniList title="Service Boundaries" items={boundaries} />
          <MiniList title="Previously Modified" items={modified} />
          <MiniList title="Compressed Context Files" items={selected} />
        </div>
        <Metric label="context tokens" value={compressed?.token_estimate ?? 0} />
      </div>
    </Card>
  );
}

function ProjectBrainPanel({
  brain,
  semantic
}: {
  brain?: Record<string, unknown> | null;
  semantic?: Record<string, unknown> | null;
}) {
  const stats = objectValue(semantic?.stats);
  const retrieved = Array.isArray(semantic?.retrieved)
    ? (semantic.retrieved as Array<Record<string, unknown>>)
    : [];
  return (
    <Card>
      <CardHeader title="Project Brain" action={<Brain size={18} className="text-accent" />}>
        {stringValue(brain?.updated_at) || "local memory pending"}
      </CardHeader>
      <div className="space-y-3 p-4 text-sm">
        <p className="text-muted">
          {asList(brain?.architecture_summaries).slice(-1)[0] || "No persisted project brain yet."}
        </p>
        <div className="grid gap-2 md:grid-cols-2">
          <MiniList title="Decisions" items={asList(brain?.decisions)} />
          <MiniList title="Successful Patterns" items={asList(brain?.successful_patterns)} />
          <MiniList title="Failures" items={asList(brain?.failures)} />
          <MiniList title="Repairs" items={asList(brain?.repairs)} />
        </div>
        <dl className="space-y-2">
          <Metric label="semantic items" value={stats.items ?? 0} />
          <Metric label="embedding" value={stats.embedding ?? "local"} />
        </dl>
        <MiniList title="Retrieved Memories" items={retrieved.map((item) => `${item.kind ?? ""}: ${item.text ?? ""}`)} />
      </div>
    </Card>
  );
}

function ContextAssemblyPanel({
  assembly,
  repositoryRag,
  knowledgeGraph,
  adrs
}: {
  assembly?: Record<string, unknown> | null;
  repositoryRag?: Record<string, unknown> | null;
  knowledgeGraph?: Record<string, unknown> | null;
  adrs: Array<Record<string, unknown>>;
}) {
  const usage = objectValue(assembly?.context_usage);
  const ragHits = Array.isArray(repositoryRag?.hits) ? (repositoryRag.hits as Array<Record<string, unknown>>) : [];
  const graphStats = objectValue(knowledgeGraph?.stats);
  const nodes = Array.isArray(knowledgeGraph?.nodes) ? (knowledgeGraph.nodes as Array<Record<string, unknown>>) : [];
  return (
    <Card>
      <CardHeader title="Context Assembly" action={<Network size={18} className="text-accent" />}>
        {`${usage.selected_files ?? 0} files / ${usage.semantic_memories ?? 0} memories`}
      </CardHeader>
      <div className="space-y-3 p-4 text-sm">
        <div className="grid gap-2 md:grid-cols-2">
          <MiniList title="Selected Files" items={asList(assembly?.relevant_files)} />
          <MiniList title="Repository RAG" items={ragHits.map((hit) => `${hit.path ?? ""} (${hit.score ?? 0})`)} />
          <MiniList title="ADR Explorer" items={adrs.map((adr) => `${adr.title ?? ""}: ${adr.decision ?? ""}`)} />
          <MiniList title="Knowledge Nodes" items={nodes.slice(0, 12).map((node) => `${node.kind ?? ""}: ${node.label ?? ""}`)} />
        </div>
        <dl className="space-y-2">
          <Metric label="graph nodes" value={graphStats.nodes ?? 0} />
          <Metric label="graph edges" value={graphStats.edges ?? 0} />
          <Metric label="rag indexed" value={objectValue(repositoryRag?.index).indexed_files ?? 0} />
        </dl>
      </div>
    </Card>
  );
}

function ToolActivityPanel({ tools }: { tools?: Record<string, unknown> | null }) {
  const activities = Array.isArray(tools?.activities)
    ? (tools.activities as Array<Record<string, unknown>>)
    : [];
  return (
    <Card>
      <CardHeader title="Tool Activity" action={<Database size={18} className="text-accent" />}>
        {activities.length ? `${activities.length} local calls` : "idle"}
      </CardHeader>
      <div className="space-y-3 p-4 text-sm">
        <MiniList title="Local Tools" items={asList(tools?.tools)} />
        <div className="max-h-44 overflow-auto rounded-md border border-border bg-slate-950">
          {activities.length ? (
            activities.slice(-10).map((activity, index) => (
              <div key={index} className="border-b border-border px-3 py-2 text-xs last:border-b-0">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium">{stringValue(activity.tool)}</span>
                  <Badge tone={activity.status === "ok" ? "success" : "danger"}>
                    {stringValue(activity.status)}
                  </Badge>
                </div>
                <div className="mt-1 text-muted">{stringValue(activity.action)}</div>
              </div>
            ))
          ) : (
            <div className="px-3 py-2 text-xs text-muted">No local tool calls recorded.</div>
          )}
        </div>
      </div>
    </Card>
  );
}

function TaskGraphPanel({
  taskPlan,
  executionGraph
}: {
  taskPlan?: Record<string, unknown> | null;
  executionGraph?: Record<string, unknown> | null;
}) {
  const tasks = Array.isArray(taskPlan?.tasks) ? (taskPlan.tasks as Array<Record<string, unknown>>) : [];
  const steps = Array.isArray(executionGraph?.steps) ? (executionGraph.steps as Array<Record<string, unknown>>) : [];
  const blocked = asList(executionGraph?.blocked);
  const completed = asList(executionGraph?.completed);
  const failed = asList(executionGraph?.failed);
  const running = stringValue(executionGraph?.running);
  return (
    <Card>
      <CardHeader title="Task Graph" action={<ListChecks size={18} className="text-accent" />}>
        {tasks.length ? `${tasks.length} tasks` : "plan pending"}
      </CardHeader>
      <div className="space-y-3 p-4">
        <div className="grid gap-2">
          {tasks.slice(0, 6).map((task) => (
            <div key={stringValue(task.task_id)} className="rounded-md border border-border bg-slate-950 p-3 text-sm">
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">{stringValue(task.task_id)}</span>
                <Badge tone="neutral">{stringValue(task.status) || "pending"}</Badge>
              </div>
              <div className="mt-1 text-muted">{stringValue(task.goal)}</div>
              <div className="mt-2 truncate text-xs text-muted">{asList(task.affected_files).join(", ")}</div>
            </div>
          ))}
        </div>
        <div className="grid gap-2 md:grid-cols-3">
          <MiniList title="Execution Steps" items={steps.map((step) => {
            const id = stringValue(step.step_id);
            const status = stringValue(step.status) || "PENDING";
            const duration = step.duration_ms != null ? ` ${step.duration_ms}ms` : "";
            return `${status} ${step.kind ?? ""} ${id}${duration}`;
          })} />
          <MiniList title="Completed" items={completed} />
          <MiniList title={running ? `Running: ${running}` : "Running"} items={running ? [running] : []} />
          <MiniList title="Blocked" items={blocked} />
          <MiniList title="Failed" items={failed} />
        </div>
      </div>
    </Card>
  );
}

function ProductionValidationPanel({
  bootstrap,
  acceptance,
  build,
  visual,
  quality,
  releaseReport,
  benchmarkSummary,
  onBenchmark
}: {
  bootstrap?: Record<string, unknown> | null;
  acceptance?: Record<string, unknown> | null;
  build?: Record<string, unknown> | null;
  visual?: Record<string, unknown> | null;
  quality?: Record<string, unknown> | null;
  releaseReport?: Record<string, unknown> | null;
  benchmarkSummary?: Record<string, unknown> | null;
  onBenchmark: () => void;
}) {
  return (
    <Card>
      <CardHeader title="Production Readiness" action={<CheckCircle2 size={18} className="text-accent" />}>
        score {String(quality?.overall ?? "pending")}
      </CardHeader>
      <div className="space-y-3 p-4 text-sm">
        <dl className="space-y-2">
          <Metric label="bootstrap" value={bootstrap?.applied ? "applied" : bootstrap?.reason ?? "pending"} />
          <Metric label="acceptance" value={acceptance?.passed === undefined ? "pending" : acceptance.passed ? "passed" : "failed"} />
          <Metric label="build" value={build?.passed === undefined ? "pending" : build.passed ? "passed" : "failed"} />
          <Metric label="visual" value={visual?.passed === undefined ? "pending" : visual.passed ? "passed" : "failed"} />
          <Metric label="report" value={releaseReport?.created_at ?? "pending"} />
        </dl>
        <div className="grid gap-2 md:grid-cols-2">
          <MiniList title="Acceptance Errors" items={asList(acceptance?.errors)} />
          <MiniList title="Build Errors" items={asList(build?.errors)} />
        </div>
        <Button variant="secondary" onClick={onBenchmark}>
          <Activity size={16} />
          Run Isolated Benchmarks
        </Button>
        {benchmarkSummary ? (
          <div className="rounded-md border border-border bg-slate-950 p-2 text-xs">
            success {String(benchmarkSummary.success_rate ?? 0)} / completion {String(benchmarkSummary.completion_rate ?? 0)}
          </div>
        ) : null}
      </div>
    </Card>
  );
}

function ConvergencePanel({ convergence }: { convergence?: Record<string, unknown> }) {
  const history = Array.isArray(convergence?.history)
    ? (convergence.history as Array<Record<string, unknown>>)
    : [];
  const passRate = Number(convergence?.test_pass_rate ?? 0);
  const status = String(convergence?.status ?? "idle");
  return (
    <Card>
      <CardHeader title="Convergence" action={<ListChecks size={18} className="text-accent" />}>
        {String(convergence?.stop_reason ?? "waiting")}
      </CardHeader>
      <div className="space-y-3 p-4 text-sm">
        <dl className="space-y-2">
          <Metric label="phase" value={convergence?.current_phase ?? "idle"} />
          <Metric label="status" value={status} />
          <Metric
            label="repair attempt"
            value={`${convergence?.current_repair_attempt ?? 0} / ${convergence?.repair_limit ?? 0}`}
          />
          <Metric label="failure" value={convergence?.failure_category ?? "none"} />
          <Metric label="last failing test" value={convergence?.last_failing_test ?? "none"} />
          <Metric label="pass rate" value={`${Math.round(passRate * 100)}%`} />
        </dl>
        <div className="h-2 rounded bg-slate-950">
          <div
            className={cn("h-2 rounded", status === "converged" ? "bg-emerald-500" : "bg-cyan-500")}
            style={{ width: `${Math.max(0, Math.min(100, passRate * 100))}%` }}
          />
        </div>
        <div className="max-h-40 overflow-auto rounded-md border border-border bg-slate-950">
          {history.length ? (
            history.slice(-6).map((entry, index) => (
              <div key={index} className="border-b border-border px-3 py-2 text-xs last:border-b-0">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium">{String(entry.phase)}</span>
                  <span className="text-muted">attempt {String(entry.attempt)}</span>
                </div>
                <div className="mt-1 truncate text-muted">
                  {String(entry.failure_category ?? entry.message ?? "ok")}
                </div>
              </div>
            ))
          ) : (
            <div className="px-3 py-2 text-xs text-muted">No repair attempts yet.</div>
          )}
        </div>
      </div>
    </Card>
  );
}

function CommandCenter(props: {
  objective: string;
  setObjective: (value: string) => void;
  repositoryRoot: string;
  setRepositoryRoot: (value: string) => void;
  targetFile: string;
  setTargetFile: (value: string) => void;
  iterations: number;
  setIterations: (value: number) => void;
  testCommand: string;
  setTestCommand: (value: string) => void;
  execute: boolean;
  setExecute: (value: boolean) => void;
  activeRun: ControlSnapshot["active_run"];
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
}) {
  return (
    <Card>
      <CardHeader title="Command Center">
        {props.activeRun ? props.activeRun.id : "ready"}
      </CardHeader>
      <div className="space-y-3 p-4">
        <label className="block text-xs font-medium text-muted">Objective</label>
        <Textarea value={props.objective} onChange={(event) => props.setObjective(event.target.value)} />
        <label className="block text-xs font-medium text-muted">Target Repository</label>
        <Input value={props.repositoryRoot} onChange={(event) => props.setRepositoryRoot(event.target.value)} />
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="mb-1 block text-xs font-medium text-muted">Target File</label>
            <Input value={props.targetFile} onChange={(event) => props.setTargetFile(event.target.value)} />
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-muted">Iterations</label>
            <Input
              type="number"
              min={1}
              max={20}
              value={props.iterations}
              onChange={(event) => props.setIterations(Number(event.target.value))}
            />
          </div>
        </div>
        <label className="block text-xs font-medium text-muted">Test Command</label>
        <Input value={props.testCommand} onChange={(event) => props.setTestCommand(event.target.value)} />
            <div className="flex items-center justify-between rounded-md border border-border bg-slate-950 px-3 py-2">
          <span className="text-sm">Execute autonomous run</span>
          <input
            type="checkbox"
            checked={props.execute}
            onChange={(event) => props.setExecute(event.target.checked)}
            className="h-4 w-4 accent-cyan-700"
          />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Button onClick={props.onStart}>
            <Play size={16} />
            Run
          </Button>
          <Button variant="secondary" onClick={props.onPause} disabled={!props.activeRun}>
            <Pause size={16} />
            Pause
          </Button>
          <Button variant="secondary" onClick={props.onResume} disabled={!props.activeRun}>
            <Play size={16} />
            Resume
          </Button>
          <Button variant="danger" onClick={props.onStop} disabled={!props.activeRun}>
            <Square size={16} />
            Stop
          </Button>
        </div>
      </div>
    </Card>
  );
}

function LiveCourtroom({ roles }: { roles: Array<Record<string, unknown>> }) {
  return (
    <Card>
      <CardHeader title="Live Courtroom" action={<Bot size={18} className="text-accent" />} />
      <div className="grid grid-cols-1 gap-3 p-4 md:grid-cols-3">
        {roles.map((role) => (
            <div key={String(role.role)} className="rounded-md border border-border bg-slate-950 p-3">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm font-semibold">{String(role.role)}</h3>
              <Badge tone={role.status === "active" ? "accent" : "neutral"}>{String(role.status)}</Badge>
            </div>
            <dl className="mt-3 space-y-2 text-xs">
              <Metric label="model" value={role.model} />
              <Metric label="runtime" value={role.runtime} />
              <Metric label="tokens" value={role.token_count ?? "pending"} />
              <Metric label="latency" value={role.inference_time_seconds ?? "pending"} />
              <Metric label="health" value={role.health} />
            </dl>
          </div>
        ))}
      </div>
    </Card>
  );
}

function ExecutionTimeline({ timeline }: { timeline: Array<{ name: string; status: string }> }) {
  return (
    <Card>
      <CardHeader title="Execution Timeline" action={<Activity size={18} className="text-accent" />} />
      <div className="grid grid-cols-2 gap-2 p-4 md:grid-cols-5">
        {timeline.map((stage) => (
          <div key={stage.name} className="flex items-center gap-2 rounded-md border border-border px-3 py-2">
            {stage.status === "completed" ? (
              <CheckCircle2 size={16} className="text-success" />
            ) : stage.status === "active" ? (
              <Clock size={16} className="text-accent" />
            ) : (
              <Clock size={16} className="text-muted" />
            )}
            <span className="truncate text-sm">{stage.name}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

function GitPanel(props: {
  git?: Record<string, unknown> | null;
  commitMessage: string;
  setCommitMessage: (value: string) => void;
  onCommit: () => void;
  onRollback: () => void;
}) {
  const status = objectValue(props.git?.status);
  const changedFiles = Array.isArray(props.git?.changed_files)
    ? (props.git.changed_files as Array<Record<string, unknown>>)
    : [];
  const history = Array.isArray(props.git?.history)
    ? (props.git.history as Array<Record<string, unknown>>)
    : [];
  return (
    <Card>
      <CardHeader title="Git Safety" action={<GitBranch size={18} className="text-accent" />}>
        {stringValue(status.branch) || "not a git repository"}
      </CardHeader>
      <div className="space-y-3 p-4 text-sm">
        <dl className="space-y-2">
          <Metric label="branch" value={status.branch ?? "unknown"} />
          <Metric label="dirty" value={status.is_dirty ? "yes" : "no"} />
          <Metric label="modified" value={asList(status.modified_files).length} />
          <Metric label="untracked" value={asList(status.untracked_files).length} />
        </dl>
        <div className="grid grid-cols-[1fr_auto_auto] gap-2">
          <Input value={props.commitMessage} onChange={(event) => props.setCommitMessage(event.target.value)} />
          <Button variant="secondary" onClick={props.onCommit}>
            <CheckCircle2 size={16} />
            Commit
          </Button>
          <Button variant="danger" onClick={props.onRollback}>
            <XCircle size={16} />
            Rollback
          </Button>
        </div>
        <div className="grid gap-2 md:grid-cols-2">
          <MiniList title="Changed Files" items={changedFiles.map((file) => `${file.status ?? "M"} ${file.path ?? ""}`)} />
          <MiniList title="Commit History" items={history.map((commit) => `${String(commit.sha).slice(0, 7)} ${commit.subject ?? ""}`)} />
        </div>
      </div>
    </Card>
  );
}

function ArtifactExplorer(props: {
  artifacts: ArtifactSummary[];
  query: string;
  setQuery: (value: string) => void;
  role: string;
  setRole: (value: string) => void;
  selected: ArtifactSummary | null;
  setSelected: (artifact: ArtifactSummary) => void;
}) {
  return (
    <Card>
      <CardHeader title="Artifact Explorer" action={<Archive size={18} className="text-accent" />} />
      <div className="space-y-3 p-4">
        <div className="grid grid-cols-[1fr_130px] gap-2">
          <div className="relative">
            <Search className="absolute left-2 top-2.5 text-muted" size={15} />
            <Input className="pl-8" value={props.query} onChange={(event) => props.setQuery(event.target.value)} />
          </div>
          <select
            value={props.role}
            onChange={(event) => props.setRole(event.target.value)}
            className="h-9 rounded-md border border-border bg-slate-950 px-2 text-sm"
          >
            <option>ALL</option>
            <option>PRIMARY_CODER</option>
            <option>DEEPSEEK_SYNTH</option>
            <option>JUDGE</option>
          </select>
        </div>
        <div className="max-h-72 overflow-auto rounded-md border border-border">
          {props.artifacts.map((artifact) => (
            <button
              key={artifact.artifact_id}
              onClick={() => props.setSelected(artifact)}
              className={cn(
                "block w-full border-b border-border px-3 py-2 text-left text-sm last:border-b-0 hover:bg-slate-900",
                props.selected?.artifact_id === artifact.artifact_id && "bg-cyan-950"
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">{artifact.role}</span>
                <span className="text-xs text-muted">round {artifact.round_id}</span>
              </div>
              <div className="mt-1 truncate text-xs text-muted">{artifact.task}</div>
            </button>
          ))}
        </div>
        <pre className="max-h-64 overflow-auto rounded-md bg-slate-950 p-3 text-xs leading-5 text-slate-100">
          {props.selected?.content ?? "No artifact selected."}
        </pre>
      </div>
    </Card>
  );
}

function RunHistoryPanel(props: {
  runs: Array<Record<string, unknown>>;
  queued: Array<Record<string, unknown>>;
  replayEvents: Array<Record<string, unknown>>;
  onReplay: (runId: string) => void;
}) {
  return (
    <Card>
      <CardHeader title="Run History" action={<Clock size={18} className="text-accent" />}>
        {props.queued.length ? `${props.queued.length} queued` : "queue empty"}
      </CardHeader>
      <div className="space-y-3 p-4">
        <div className="rounded-md border border-border bg-slate-950">
          {props.runs.length ? (
            props.runs.slice(0, 8).map((run) => {
              const runId = stringValue(run.run_id);
              return (
                <button
                  key={runId}
                  className="block w-full border-b border-border px-3 py-2 text-left text-sm last:border-b-0 hover:bg-slate-900"
                  onClick={() => runId && props.onReplay(runId)}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-medium">{stringValue(run.objective)}</span>
                    <Badge tone={run.status === "completed" ? "success" : run.status === "failed" ? "danger" : "neutral"}>
                      {stringValue(run.status)}
                    </Badge>
                  </div>
                  <div className="mt-1 truncate text-xs text-muted">{stringValue(run.branch) || stringValue(run.repository_path)}</div>
                </button>
              );
            })
          ) : (
            <div className="px-3 py-2 text-sm text-muted">No persisted runs yet.</div>
          )}
        </div>
        <MiniList
          title="Queued Tasks"
          items={props.queued.map((task) => `${task.id ?? ""} ${task.objective ?? ""}`)}
        />
        <div className="max-h-48 overflow-auto rounded-md border border-border bg-slate-950">
          {props.replayEvents.length ? (
            props.replayEvents.map((event, index) => (
              <div key={index} className="border-b border-border px-3 py-2 text-xs last:border-b-0">
                <div className="font-medium">{stringValue(event.stage)}</div>
                <div className="mt-1 truncate text-muted">{stringValue(event.message)}</div>
              </div>
            ))
          ) : (
            <div className="px-3 py-2 text-xs text-muted">Select a run to replay stored telemetry.</div>
          )}
        </div>
      </div>
    </Card>
  );
}

function ObjectiveMemoryPanel({ objectives }: { objectives: Array<Record<string, unknown>> }) {
  return (
    <Card>
      <CardHeader title="Run Knowledge Base" action={<Archive size={18} className="text-accent" />}>
        {objectives.length ? `${objectives.length} objectives` : "empty"}
      </CardHeader>
      <div className="max-h-72 overflow-auto p-4">
        {objectives.length ? (
          objectives.slice(0, 8).map((objective, index) => (
            <div key={index} className="mb-2 rounded-md border border-border bg-slate-950 p-3 text-sm last:mb-0">
              <div className="flex items-center justify-between gap-2">
                <span className="truncate font-medium">{stringValue(objective.objective)}</span>
                <Badge tone={objective.outcome === "passed" ? "success" : objective.outcome === "failed" ? "danger" : "neutral"}>
                  {stringValue(objective.outcome)}
                </Badge>
              </div>
              <div className="mt-1 truncate text-xs text-muted">
                failures: {asList(objective.failures).join(", ") || "none"}
              </div>
            </div>
          ))
        ) : (
          <div className="text-sm text-muted">No objective memory recorded yet.</div>
        )}
      </div>
    </Card>
  );
}

function PatchViewer({ patch }: { patch?: ControlSnapshot["patch"] }) {
  const diffLines = (patch?.diff ?? "").split("\n");
  return (
    <Card>
      <CardHeader title="Patch Viewer" action={<GitCompare size={18} className="text-accent" />}>
        {(patch?.changed_files ?? []).join(", ") || "clean"}
      </CardHeader>
      <div className="p-4">
        <pre className="max-h-96 overflow-auto rounded-md border border-border bg-slate-950 p-3 text-xs leading-5 text-slate-100">
          {diffLines.length && diffLines[0]
            ? diffLines.map((line, index) => (
                <div
                  key={`${index}-${line}`}
                  className={cn(
                    line.startsWith("+") && "text-emerald-300",
                    line.startsWith("-") && "text-red-300",
                    line.startsWith("@@") && "text-cyan-300"
                  )}
                >
                  {line}
                </div>
              ))
            : "No patch detected."}
        </pre>
      </div>
    </Card>
  );
}

function RepositoryExplorer({
  tree,
  selectedFile,
  onOpenFile
}: {
  tree: RepoTreeNode | null;
  selectedFile: { path: string; content: string } | null;
  onOpenFile: (path: string) => void;
}) {
  return (
    <Card>
      <CardHeader title="Repository Explorer" action={<FileCode2 size={18} className="text-accent" />} />
      <div className="grid gap-3 p-4">
        <div className="max-h-60 overflow-auto rounded-md border border-border bg-slate-950 p-2 text-sm">
          {tree ? <TreeNode node={tree} onOpenFile={onOpenFile} /> : <span className="text-muted">No repository loaded.</span>}
        </div>
        <pre className="max-h-64 overflow-auto rounded-md bg-slate-950 p-3 text-xs leading-5 text-slate-100">
          {selectedFile ? selectedFile.content : "No file selected."}
        </pre>
      </div>
    </Card>
  );
}

function TreeNode({ node, onOpenFile }: { node: RepoTreeNode; onOpenFile: (path: string) => void }) {
  return (
    <div className="pl-2">
      <button
        className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-slate-800"
        onClick={() => node.type === "file" && onOpenFile(node.path)}
      >
        {node.type === "file" ? <Code2 size={14} /> : <GitBranch size={14} />}
        <span className="truncate">{node.name}</span>
      </button>
      {node.children?.length ? (
        <div className="border-l border-border pl-3">
          {node.children.map((child) => (
            <TreeNode key={child.path} node={child} onOpenFile={onOpenFile} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function RuntimeMonitor({ runtime }: { runtime?: Record<string, unknown> }) {
  const diagnostics = (runtime?.diagnostics ?? {}) as Record<string, unknown>;
  return (
    <Card>
      <CardHeader title="Runtime Monitor" action={<Terminal size={18} className="text-accent" />} />
      <dl className="space-y-2 p-4 text-sm">
        <Metric label="active runtime" value={runtime?.active_runtime ?? "idle"} />
        <Metric label="model" value={runtime?.active_model ?? "none"} />
        <Metric label="load status" value={diagnostics.load_status ?? "idle"} />
        <Metric label="fallback" value={diagnostics.fallback_status ?? "none"} />
        <Metric label="vram required" value={formatMegabytes(diagnostics.vram_required)} />
        <Metric label="vram available" value={formatMegabytes(diagnostics.free_vram)} />
        <Metric label="pid" value={runtime?.pid ?? "pending"} />
        <Metric label="pgid" value={runtime?.pgid ?? "pending"} />
        <Metric label="swap count" value={runtime?.swap_count ?? "pending"} />
        <Metric label="health" value={runtime?.health ?? "unknown"} />
        <Metric label="vram" value={formatVram(runtime?.vram)} />
      </dl>
    </Card>
  );
}

function LogsPanel({ logs }: { logs: string[] }) {
  return (
    <Card>
      <CardHeader title="Logs Panel" action={<Terminal size={18} className="text-accent" />} />
      <pre className="max-h-72 overflow-auto p-4 text-xs leading-5">
        {logs.length ? logs.join("\n") : "[READY] waiting for runtime events"}
      </pre>
    </Card>
  );
}

function TestResults({ tests }: { tests?: Record<string, unknown> }) {
  const status = String(tests?.status ?? "idle");
  return (
    <Card>
      <CardHeader title="Test Results" action={status === "failed" ? <XCircle size={18} className="text-danger" /> : <CheckCircle2 size={18} className="text-success" />} />
      <dl className="space-y-2 p-4 text-sm">
        <Metric label="status" value={status} />
        <Metric label="passing" value={tests?.passing ?? 0} />
        <Metric label="failing" value={tests?.failing ?? 0} />
        <Metric label="retries" value={tests?.retries ?? 0} />
        <Metric label="repairs" value={tests?.repair_attempts ?? 0} />
      </dl>
    </Card>
  );
}

function ConversationView({ items }: { items: Array<Record<string, unknown>> }) {
  return (
    <Card>
      <CardHeader title="Conversation View" action={<Bot size={18} className="text-accent" />} />
      <div className="space-y-2 p-4">
        {items.length ? (
          items.map((item, index) => (
            <div key={index} className="rounded-md border border-border bg-slate-950 p-3 text-sm">
              <div className="mb-1 flex items-center justify-between">
                <span className="font-medium">{String(item.role)}</span>
                <span className="text-xs text-muted">round {String(item.round_id)}</span>
              </div>
              <p className="text-sm text-muted">{String(item.summary)}</p>
            </div>
          ))
        ) : (
          <div className="text-sm text-muted">No summaries available.</div>
        )}
      </div>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="text-muted">{label}</dt>
      <dd className="truncate text-right font-medium">{String(value ?? "unknown")}</dd>
    </div>
  );
}

function formatVram(value: unknown) {
  if (!value || typeof value !== "object") return "unknown";
  const vram = value as Record<string, unknown>;
  if (!vram.available) return "unavailable";
  return `${vram.memory_used_mb} / ${vram.memory_total_mb} MB`;
}

function formatMegabytes(value: unknown) {
  return typeof value === "number" ? `${value} MB` : "unknown";
}

function asList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item));
}

function objectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}
