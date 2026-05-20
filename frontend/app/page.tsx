"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Archive,
  Bot,
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  Circle,
  Clock,
  Code2,
  Command,
  Database,
  FileCode2,
  Folder,
  FolderOpen,
  GitBranch,
  GitCommit,
  GitCompare,
  History,
  LayoutDashboard,
  ListChecks,
  Loader2,
  MemoryStick,
  Network,
  Pause,
  Play,
  RefreshCw,
  Search,
  Settings,
  Sparkles,
  Square,
  Terminal,
  TestTube2,
  X,
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
import { Button, Input, Textarea } from "@/components/ui";
import { cn } from "@/lib/utils";

const defaultRepo = "/home/ratish/Forge";

const phases = [
  "REPOSITORY_SCAN",
  "PLANNING",
  "CODER",
  "SYNTH",
  "JUDGE",
  "PATCH",
  "TESTS",
  "REPAIR"
];

const promptTemplates = [
  "Build a complete FastAPI Todo application.",
  "Add OAuth login with tests.",
  "Create an admin dashboard.",
  "Fix all failing tests.",
  "Refactor authentication into service modules."
];

type RightTab = "plan" | "files" | "memory" | "logs";
type DockTab = "logs" | "tests" | "runtime" | "git";

export default function ControlCenterPage() {
  const [repositoryRoot, setRepositoryRoot] = useState(defaultRepo);
  const [targetFile, setTargetFile] = useState("app.py");
  const [objective, setObjective] = useState("Build a complete FastAPI Todo application.");
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
  const [selectedArtifact, setSelectedArtifact] = useState<ArtifactSummary | null>(null);
  const [replayEvents, setReplayEvents] = useState<Array<Record<string, unknown>>>([]);
  const [benchmarkSummary, setBenchmarkSummary] = useState<Record<string, unknown> | null>(null);
  const [health, setHealth] = useState<"checking" | "ok" | "degraded" | "failed">("checking");
  const [error, setError] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState<RightTab>("plan");
  const [dockTab, setDockTab] = useState<DockTab>("logs");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [dockHeight, setDockHeight] = useState(300);
  const [apiStats, setApiStats] = useState({
    snapshotLatency: 0,
    healthLatency: 0,
    failures: 0,
    lastRefresh: ""
  });

  const activeRun = snapshot?.active_run ?? null;
  const loadedRepositoryRoot = snapshot?.active_repository_root || repositoryRoot;
  const plan = snapshot?.generated_plan ?? snapshot?.execution_plan ?? null;
  const phase = activeRun?.phase || (activeRun?.status === "running" ? "CODER" : "IDLE");

  useEffect(() => {
    const storedSidebar = window.localStorage.getItem("forge.sidebarCollapsed");
    const storedDock = window.localStorage.getItem("forge.dockHeight");
    if (storedSidebar) setSidebarCollapsed(storedSidebar === "true");
    if (storedDock) setDockHeight(Math.max(220, Math.min(520, Number(storedDock))));
  }, []);

  useEffect(() => {
    window.localStorage.setItem("forge.sidebarCollapsed", String(sidebarCollapsed));
  }, [sidebarCollapsed]);

  useEffect(() => {
    window.localStorage.setItem("forge.dockHeight", String(dockHeight));
  }, [dockHeight]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((open) => !open);
      }
      if (event.key === "Escape") setPaletteOpen(false);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

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

  const latestArtifact = useMemo(() => {
    const artifacts = snapshot?.artifacts ?? [];
    return selectedArtifact ?? artifacts[artifacts.length - 1] ?? null;
  }, [selectedArtifact, snapshot?.artifacts]);

  const generatedFiles = useMemo(() => generatedFileNames(snapshot?.artifacts ?? [], plan), [snapshot?.artifacts, plan]);

  async function startRun() {
    const command = testCommand.split(" ").filter(Boolean);
    const activeRepository = snapshot?.active_repository;
    const activeRepositoryPath = stringValue(activeRepository?.repository_path);
    const activeRepositoryId = stringValue(activeRepository?.repository_id);
    const repositoryId = activeRepositoryPath === repositoryRoot ? activeRepositoryId : null;
    const run = await createRun({
      objective,
      repository_root: repositoryRoot,
      repository_id: repositoryId,
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
    setRightTab("files");
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
    <main className="forge-shell">
      <CommandPalette
        open={paletteOpen}
        objective={objective}
        setObjective={setObjective}
        repositoryRoot={repositoryRoot}
        setRepositoryRoot={setRepositoryRoot}
        onClose={() => setPaletteOpen(false)}
        onRun={startRun}
        onOpenLogs={() => {
          setDockTab("logs");
          setPaletteOpen(false);
        }}
      />

      <LeftSidebar
        collapsed={sidebarCollapsed}
        setCollapsed={setSidebarCollapsed}
        repositories={snapshot?.repositories ?? []}
        activeRepository={snapshot?.active_repository}
        importPath={repositoryImportPath}
        setImportPath={setRepositoryImportPath}
        browser={workspaceBrowser}
        browserError={workspaceError}
        runHistory={snapshot?.run_history ?? []}
        objectiveMemory={snapshot?.objective_memory ?? []}
        knowledgeGraph={snapshot?.knowledge_graph}
        benchmarkSummary={benchmarkSummary}
        onBrowse={browseDirectory}
        onImport={importCurrentRepository}
        onSelect={selectRepository}
        onRefresh={refreshActiveRepository}
        onReplay={openReplay}
        onBenchmark={runIsolatedBenchmarks}
      />

      <section className="workspace-main">
        <TopBar
          health={health}
          activeRun={activeRun}
          phase={phase}
          repositoryRoot={loadedRepositoryRoot}
          apiStats={apiStats}
          error={error}
          onRefresh={() => fetchSnapshot(repositoryRoot).then(setSnapshot)}
          onPalette={() => setPaletteOpen(true)}
        />

        <div className="workspace-grid">
          <section className="center-stage">
            <ObjectiveComposer
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
              classification={snapshot?.objective_classification}
              onStart={startRun}
              onPause={() => runAction("pause")}
              onResume={() => runAction("resume")}
              onStop={() => runAction("stop")}
            />
            <ActiveRunPanel activeRun={activeRun} phase={phase} snapshot={snapshot} />
            <VisualTimeline timeline={snapshot?.timeline ?? []} phase={phase} />
            <LiveCourtroom roles={snapshot?.courtroom ?? []} runtime={snapshot?.runtime} />
            <LiveCodeGeneration artifacts={snapshot?.artifacts ?? []} generatedFiles={generatedFiles} onSelect={setSelectedArtifact} />
            <DiffViewer patch={snapshot?.patch} />
          </section>

          <RightInspector
            tab={rightTab}
            setTab={setRightTab}
            plan={plan}
            summary={snapshot?.repository_summary}
            architecture={snapshot?.architecture_summary}
            tree={tree}
            selectedFile={selectedFile}
            onOpenFile={openFile}
            projectBrain={snapshot?.project_brain}
            architectureMemory={snapshot?.architecture_memory}
            semanticMemory={snapshot?.semantic_memory}
            repositoryRag={snapshot?.repository_rag}
            knowledgeGraph={snapshot?.knowledge_graph}
            adrs={snapshot?.adrs ?? []}
            logs={snapshot?.logs ?? []}
            latestArtifact={latestArtifact}
            artifacts={snapshot?.artifacts ?? []}
            onArtifactSelect={setSelectedArtifact}
          />
        </div>

        <BottomDock
          tab={dockTab}
          setTab={setDockTab}
          height={dockHeight}
          setHeight={setDockHeight}
          logs={snapshot?.logs ?? []}
          tests={snapshot?.tests}
          convergence={snapshot?.convergence}
          runtime={snapshot?.runtime}
          git={snapshot?.git}
          commitMessage={commitMessage}
          setCommitMessage={setCommitMessage}
          onCommit={commitActiveRepository}
          onRollback={rollbackActiveRepository}
          replayEvents={replayEvents}
          releaseReport={snapshot?.release_report}
          acceptance={snapshot?.acceptance}
          build={snapshot?.build_validation}
          visual={snapshot?.visual_validation}
          quality={snapshot?.quality_score}
        />
      </section>
    </main>
  );
}

function TopBar({
  health,
  activeRun,
  phase,
  repositoryRoot,
  apiStats,
  error,
  onRefresh,
  onPalette
}: {
  health: "checking" | "ok" | "degraded" | "failed";
  activeRun: ControlSnapshot["active_run"];
  phase: string;
  repositoryRoot: string;
  apiStats: { snapshotLatency: number; healthLatency: number; failures: number; lastRefresh: string };
  error: string | null;
  onRefresh: () => void;
  onPalette: () => void;
}) {
  return (
    <header className="topbar">
      <div className="min-w-0">
        <div className="brand-row">
          <div className="brand-mark"><Sparkles size={18} /></div>
          <div className="min-w-0">
            <h1>Forge</h1>
            <p>{repositoryRoot}</p>
          </div>
        </div>
      </div>
      <div className="topbar-status">
        <StatusPill tone={health === "ok" ? "success" : health === "failed" ? "danger" : "warning"}>
          backend {health}
        </StatusPill>
        <StatusPill tone={activeRun ? "accent" : "neutral"}>{phase}</StatusPill>
        <span className="latency">snapshot {apiStats.snapshotLatency}ms</span>
        <span className="latency">health {apiStats.healthLatency}ms</span>
        <span className="latency">last {apiStats.lastRefresh || "pending"}</span>
        <Button variant="ghost" className="h-8 px-2" onClick={onPalette}>
          <Command size={15} />
          Ctrl+K
        </Button>
        <Button variant="secondary" className="h-8 px-2" onClick={onRefresh}>
          <RefreshCw size={15} />
        </Button>
      </div>
      {error ? <div className="api-alert">{error}</div> : null}
    </header>
  );
}

function LeftSidebar({
  collapsed,
  setCollapsed,
  repositories,
  activeRepository,
  importPath,
  setImportPath,
  browser,
  browserError,
  runHistory,
  objectiveMemory,
  knowledgeGraph,
  benchmarkSummary,
  onBrowse,
  onImport,
  onSelect,
  onRefresh,
  onReplay,
  onBenchmark
}: {
  collapsed: boolean;
  setCollapsed: (value: boolean) => void;
  repositories: Array<Record<string, unknown>>;
  activeRepository?: Record<string, unknown> | null;
  importPath: string;
  setImportPath: (value: string) => void;
  browser?: Record<string, unknown> | null;
  browserError?: string | null;
  runHistory: Array<Record<string, unknown>>;
  objectiveMemory: Array<Record<string, unknown>>;
  knowledgeGraph?: Record<string, unknown> | null;
  benchmarkSummary?: Record<string, unknown> | null;
  onBrowse: (path?: string) => void;
  onImport: () => void;
  onSelect: (repositoryId: string) => void;
  onRefresh: () => void;
  onReplay: (runId: string) => void;
  onBenchmark: () => void;
}) {
  const entries = Array.isArray(browser?.entries) ? (browser.entries as Array<Record<string, unknown>>) : [];
  const roots = asList(browser?.roots);
  const parent = stringValue(browser?.parent);
  const graphStats = objectValue(knowledgeGraph?.stats);

  return (
    <aside className={cn("left-sidebar", collapsed && "is-collapsed")}>
      <div className="sidebar-head">
        <button className="icon-button" onClick={() => setCollapsed(!collapsed)} aria-label="Toggle sidebar">
          {collapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
        </button>
        {!collapsed ? <span>Workspace</span> : null}
      </div>
      <nav className="rail-icons" aria-label="Workspace navigation">
        <RailIcon icon={<FolderOpen size={17} />} label="Repositories" collapsed={collapsed} />
        <RailIcon icon={<History size={17} />} label="Run History" collapsed={collapsed} />
        <RailIcon icon={<Brain size={17} />} label="Memories" collapsed={collapsed} />
        <RailIcon icon={<Network size={17} />} label="Knowledge Graph" collapsed={collapsed} />
        <RailIcon icon={<Activity size={17} />} label="Benchmarks" collapsed={collapsed} />
        <RailIcon icon={<Settings size={17} />} label="Settings" collapsed={collapsed} />
      </nav>
      {collapsed ? null : (
        <div className="sidebar-scroll">
          <SidebarSection title="Repositories" icon={<FolderOpen size={14} />}>
            <div className="path-import">
              <Input value={importPath} onChange={(event) => setImportPath(event.target.value)} />
              <div className="path-actions">
                <Button variant="secondary" className="h-8 px-2" onClick={() => onBrowse(importPath)}>Browse</Button>
                <Button variant="secondary" className="h-8 px-2" onClick={onImport}>Import</Button>
              </div>
            </div>
            {browserError ? <div className="inline-error">{browserError}</div> : null}
            <div className="browser-card">
              <div className="browser-current">
                <span>{stringValue(browser?.current) || "Select a folder"}</span>
                {parent ? <button onClick={() => onBrowse(parent)}>Up</button> : null}
              </div>
              {roots.length ? (
                <div className="root-list">
                  {roots.map((root) => <button key={root} onClick={() => onBrowse(root)}>{root}</button>)}
                </div>
              ) : null}
              <div className="dir-list">
                {entries.slice(0, 18).map((entry) => {
                  const path = stringValue(entry.path);
                  return (
                    <button key={path} onClick={() => path && onBrowse(path)}>
                      <Folder size={14} />
                      <span>{stringValue(entry.name)}</span>
                      <small>{entry.is_git_repository ? "git" : entry.has_app_markers ? "app" : ""}</small>
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="repo-list">
              {repositories.length ? repositories.slice(0, 10).map((repo) => {
                const id = stringValue(repo.repository_id);
                const active = id === stringValue(activeRepository?.repository_id);
                return (
                  <button key={id} className={cn(active && "active")} onClick={() => id && onSelect(id)}>
                    <GitBranch size={14} />
                    <span>{stringValue(repo.repository_name) || "repository"}</span>
                    <small>{stringValue(repo.branch) || stringValue(repo.repository_type)}</small>
                  </button>
                );
              }) : <p className="empty">No repositories registered.</p>}
            </div>
            <Button variant="secondary" className="w-full" disabled={!activeRepository} onClick={onRefresh}>
              <RefreshCw size={15} />
              Refresh Intelligence
            </Button>
          </SidebarSection>

          <SidebarSection title="Run History" icon={<History size={14} />}>
            <div className="history-list">
              {runHistory.length ? runHistory.slice(0, 8).map((run) => {
                const id = stringValue(run.run_id);
                return (
                  <button key={id} onClick={() => id && onReplay(id)}>
                    <span>{stringValue(run.objective) || "Autonomous run"}</span>
                    <StatusPill tone={run.status === "completed" ? "success" : run.status === "failed" ? "danger" : "neutral"}>
                      {stringValue(run.status) || "stored"}
                    </StatusPill>
                  </button>
                );
              }) : <p className="empty">No persisted runs yet.</p>}
            </div>
          </SidebarSection>

          <SidebarSection title="Memory" icon={<Brain size={14} />}>
            <StatGrid
              items={[
                ["objectives", objectiveMemory.length],
                ["nodes", graphStats.nodes ?? 0],
                ["edges", graphStats.edges ?? 0],
                ["bench", benchmarkSummary?.success_rate ?? "idle"]
              ]}
            />
          </SidebarSection>

          <Button variant="secondary" className="w-full" onClick={onBenchmark}>
            <Activity size={15} />
            Run Isolated Benchmarks
          </Button>
        </div>
      )}
    </aside>
  );
}

function ObjectiveComposer({
  objective,
  setObjective,
  repositoryRoot,
  setRepositoryRoot,
  targetFile,
  setTargetFile,
  iterations,
  setIterations,
  testCommand,
  setTestCommand,
  execute,
  setExecute,
  activeRun,
  classification,
  onStart,
  onPause,
  onResume,
  onStop
}: {
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
  classification?: string | null;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
}) {
  return (
    <section className="hero-composer">
      <div className="composer-head">
        <div>
          <p className="eyebrow">Autonomous objective</p>
          <h2>What should Forge build, repair, or refactor?</h2>
        </div>
        <StatusPill tone={classification ? "accent" : "neutral"}>{classification || "unclassified"}</StatusPill>
      </div>
      <Textarea
        className="objective-box"
        value={objective}
        onChange={(event) => setObjective(event.target.value)}
        placeholder="Describe the engineering outcome. Forge will scan, plan, code, validate, repair, and judge."
      />
      <div className="template-row">
        {promptTemplates.map((template) => (
          <button key={template} onClick={() => setObjective(template)}>{template}</button>
        ))}
      </div>
      <div className="composer-grid">
        <Field label="Repository"><Input value={repositoryRoot} onChange={(event) => setRepositoryRoot(event.target.value)} /></Field>
        <Field label="Target file"><Input value={targetFile} onChange={(event) => setTargetFile(event.target.value)} /></Field>
        <Field label="Iterations"><Input type="number" min={1} max={20} value={iterations} onChange={(event) => setIterations(Number(event.target.value))} /></Field>
        <Field label="Tests"><Input value={testCommand} onChange={(event) => setTestCommand(event.target.value)} /></Field>
      </div>
      <div className="composer-actions">
        <label className="execute-toggle">
          <input type="checkbox" checked={execute} onChange={(event) => setExecute(event.target.checked)} />
          Execute immediately
        </label>
        <div className="run-controls">
          <Button onClick={onStart}><Play size={16} /> Run</Button>
          <Button variant="secondary" onClick={onPause} disabled={!activeRun}><Pause size={16} /> Pause</Button>
          <Button variant="secondary" onClick={onResume} disabled={!activeRun}><Play size={16} /> Resume</Button>
          <Button variant="danger" onClick={onStop} disabled={!activeRun}><Square size={16} /> Stop</Button>
        </div>
      </div>
    </section>
  );
}

function ActiveRunPanel({
  activeRun,
  phase,
  snapshot
}: {
  activeRun: ControlSnapshot["active_run"];
  phase: string;
  snapshot: ControlSnapshot | null;
}) {
  const changedFiles = snapshot?.patch?.changed_files ?? [];
  const tests = snapshot?.tests ?? {};
  const runtime = snapshot?.runtime ?? {};
  return (
    <section className="active-run-panel">
      <div className="run-summary">
        <div>
          <p className="eyebrow">Active run</p>
          <h2>{activeRun?.objective || snapshot?.active_objective || "No autonomous run active"}</h2>
        </div>
        <StatusPill tone={activeRun?.status === "failed" ? "danger" : activeRun ? "accent" : "neutral"}>
          {activeRun?.status || "idle"}
        </StatusPill>
      </div>
      <div className="phase-meter">
        <div className="phase-meter-fill" style={{ width: `${phasePercent(phase)}%` }} />
      </div>
      <div className="run-metrics">
        <MetricCard label="phase" value={phase} icon={<Activity size={16} />} />
        <MetricCard label="model" value={runtime.active_model ?? "none"} icon={<Bot size={16} />} />
        <MetricCard label="changed files" value={changedFiles.length} icon={<GitCompare size={16} />} />
        <MetricCard label="tests" value={tests.status ?? "idle"} icon={<TestTube2 size={16} />} />
      </div>
    </section>
  );
}

function VisualTimeline({ timeline, phase }: { timeline: Array<{ name: string; status: string }>; phase: string }) {
  const normalized = phases.map((name) => {
    const item = timeline.find((stage) => stage.name === name || stage.name?.toUpperCase?.() === name);
    const status = item?.status || inferStageStatus(name, phase);
    return { name, status };
  });

  return (
    <section className="timeline-card">
      <div className="section-title">
        <Activity size={16} />
        <span>Execution timeline</span>
      </div>
      <div className="execution-timeline">
        {normalized.map((stage, index) => (
          <div key={stage.name} className={cn("timeline-step", stage.status)}>
            <div className="timeline-dot">
              {stage.status === "completed" ? <Check size={13} /> : stage.status === "active" ? <Play size={12} /> : <Circle size={10} />}
            </div>
            <div>
              <strong>{formatPhase(stage.name)}</strong>
              <span>{stage.status === "active" ? "running" : stage.status}</span>
            </div>
            {index < normalized.length - 1 ? <div className="timeline-line" /> : null}
          </div>
        ))}
      </div>
    </section>
  );
}

function LiveCourtroom({ roles, runtime }: { roles: Array<Record<string, unknown>>; runtime?: Record<string, unknown> }) {
  const fallback = [
    { role: "PRIMARY_CODER", status: "waiting" },
    { role: "DEEPSEEK_SYNTH", status: "waiting" },
    { role: "JUDGE", status: "waiting" }
  ];
  const data = roles.length ? roles : fallback;
  return (
    <section className="courtroom-panel">
      <div className="section-title">
        <Bot size={16} />
        <span>Live courtroom</span>
      </div>
      <div className="model-grid">
        {data.map((role) => {
          const roleData = role as Record<string, unknown>;
          const status = stringValue(roleData.status) || "waiting";
          const active = status === "active" || status === "running";
          return (
            <article key={String(roleData.role)} className={cn("model-card", active && "active")}>
              <div className="model-card-head">
                <div className="model-avatar"><Bot size={17} /></div>
                <div>
                  <h3>{String(roleData.role)}</h3>
                  <p>{stringValue(roleData.model) || String(runtime?.active_model ?? "local runtime")}</p>
                </div>
                <StatusPill tone={active ? "accent" : status === "failed" ? "danger" : status === "completed" ? "success" : "neutral"}>
                  {status}
                </StatusPill>
              </div>
              <div className="model-stats">
                <MetricInline label="tokens" value={roleData.token_count ?? "pending"} />
                <MetricInline label="latency" value={roleData.inference_time_seconds ?? "pending"} />
                <MetricInline label="health" value={roleData.health ?? runtime?.health ?? "unknown"} />
                <MetricInline label="output" value={roleData.output_size ?? "pending"} />
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function LiveCodeGeneration({
  artifacts,
  generatedFiles,
  onSelect
}: {
  artifacts: ArtifactSummary[];
  generatedFiles: string[];
  onSelect: (artifact: ArtifactSummary) => void;
}) {
  return (
    <section className="codegen-panel">
      <div className="section-title">
        <Code2 size={16} />
        <span>Live code generation</span>
      </div>
      <div className="codegen-grid">
        <div className="generated-file-list">
          {generatedFiles.length ? generatedFiles.slice(0, 14).map((file) => (
            <div key={file} className="generated-file">
              <FileCode2 size={14} />
              <span>{file}</span>
              <small>tracked</small>
            </div>
          )) : <p className="empty">Generated files will appear as PRIMARY_CODER artifacts arrive.</p>}
        </div>
        <div className="artifact-stream">
          {artifacts.length ? artifacts.slice(-4).reverse().map((artifact) => (
            <button key={artifact.artifact_id} onClick={() => onSelect(artifact)}>
              <span>{artifact.role}</span>
              <strong>{artifact.content.length.toLocaleString()} chars</strong>
              <small>{artifact.task}</small>
            </button>
          )) : <p className="empty">No model output yet.</p>}
        </div>
      </div>
    </section>
  );
}

function DiffViewer({ patch }: { patch?: ControlSnapshot["patch"] }) {
  const diffLines = (patch?.diff ?? "").split("\n").filter(Boolean);
  const added = diffLines.filter((line) => line.startsWith("+") && !line.startsWith("+++")).length;
  const removed = diffLines.filter((line) => line.startsWith("-") && !line.startsWith("---")).length;
  return (
    <section className="diff-panel">
      <div className="section-title">
        <GitCompare size={16} />
        <span>File changes</span>
        <small>{(patch?.changed_files ?? []).length} files</small>
        <small className="additions">+{added}</small>
        <small className="deletions">-{removed}</small>
      </div>
      <div className="changed-files">
        {(patch?.changed_files ?? []).map((file) => <span key={file}>{file}</span>)}
      </div>
      <pre className="diff-viewer">
        {diffLines.length ? diffLines.map((line, index) => (
          <div key={`${index}-${line}`} className={cn(
            "diff-line",
            line.startsWith("+") && "plus",
            line.startsWith("-") && "minus",
            line.startsWith("@@") && "hunk",
            line.startsWith("diff --git") && "file"
          )}>
            <span>{String(index + 1).padStart(4, " ")}</span>
            <code>{line}</code>
          </div>
        )) : <div className="diff-empty">No repository diff yet.</div>}
      </pre>
    </section>
  );
}

function RightInspector({
  tab,
  setTab,
  plan,
  summary,
  architecture,
  tree,
  selectedFile,
  onOpenFile,
  projectBrain,
  architectureMemory,
  semanticMemory,
  repositoryRag,
  knowledgeGraph,
  adrs,
  logs,
  latestArtifact,
  artifacts,
  onArtifactSelect
}: {
  tab: RightTab;
  setTab: (tab: RightTab) => void;
  plan?: Record<string, unknown> | null;
  summary?: Record<string, unknown> | null;
  architecture?: string | null;
  tree: RepoTreeNode | null;
  selectedFile: { path: string; content: string } | null;
  onOpenFile: (path: string) => void;
  projectBrain?: Record<string, unknown> | null;
  architectureMemory?: Record<string, unknown> | null;
  semanticMemory?: Record<string, unknown> | null;
  repositoryRag?: Record<string, unknown> | null;
  knowledgeGraph?: Record<string, unknown> | null;
  adrs: Array<Record<string, unknown>>;
  logs: string[];
  latestArtifact: ArtifactSummary | null;
  artifacts: ArtifactSummary[];
  onArtifactSelect: (artifact: ArtifactSummary) => void;
}) {
  return (
    <aside className="right-inspector">
      <div className="inspector-tabs">
        <TabButton active={tab === "plan"} onClick={() => setTab("plan")}>Plan</TabButton>
        <TabButton active={tab === "files"} onClick={() => setTab("files")}>Files</TabButton>
        <TabButton active={tab === "memory"} onClick={() => setTab("memory")}>Memory</TabButton>
        <TabButton active={tab === "logs"} onClick={() => setTab("logs")}>Logs</TabButton>
      </div>
      <div className="inspector-body">
        {tab === "plan" ? <PlanInspector plan={plan} summary={summary} architecture={architecture} latestArtifact={latestArtifact} artifacts={artifacts} onArtifactSelect={onArtifactSelect} /> : null}
        {tab === "files" ? <FileInspector tree={tree} selectedFile={selectedFile} onOpenFile={onOpenFile} /> : null}
        {tab === "memory" ? (
          <MemoryInspector
            projectBrain={projectBrain}
            architectureMemory={architectureMemory}
            semanticMemory={semanticMemory}
            repositoryRag={repositoryRag}
            knowledgeGraph={knowledgeGraph}
            adrs={adrs}
          />
        ) : null}
        {tab === "logs" ? <LogStream logs={logs} compact /> : null}
      </div>
    </aside>
  );
}

function PlanInspector({
  plan,
  summary,
  architecture,
  latestArtifact,
  artifacts,
  onArtifactSelect
}: {
  plan?: Record<string, unknown> | null;
  summary?: Record<string, unknown> | null;
  architecture?: string | null;
  latestArtifact: ArtifactSummary | null;
  artifacts: ArtifactSummary[];
  onArtifactSelect: (artifact: ArtifactSummary) => void;
}) {
  return (
    <div className="inspector-stack">
      <PanelTitle icon={<ListChecks size={15} />} title="Execution plan" subtitle={stringValue(plan?.objective_type) || "waiting"} />
      <p className="muted-copy">{architecture || "Repository architecture summary will appear after scan."}</p>
      <StatGrid
        items={[
          ["language", summary?.primary_language ?? "unknown"],
          ["frameworks", asList(summary?.frameworks).join(", ") || "none"],
          ["tests", asList(summary?.test_frameworks).join(", ") || "none"],
          ["package", asList(summary?.package_managers).join(", ") || "none"]
        ]}
      />
      <FileList title="Files to create" files={asList(plan?.files_to_create)} />
      <FileList title="Files to modify" files={asList(plan?.files_to_modify)} />
      <FileList title="Expected tests" files={asList(plan?.expected_tests)} />
      <PanelTitle icon={<Archive size={15} />} title="Artifacts" subtitle={latestArtifact?.role ?? "none"} />
      <div className="artifact-list">
        {artifacts.length ? artifacts.slice(-8).reverse().map((artifact) => (
          <button key={artifact.artifact_id} onClick={() => onArtifactSelect(artifact)}>
            <span>{artifact.role}</span>
            <small>{artifact.content.length.toLocaleString()} chars</small>
          </button>
        )) : <p className="empty">No artifacts yet.</p>}
      </div>
      <pre className="artifact-preview">{latestArtifact?.content ?? "Select an artifact to inspect model output."}</pre>
    </div>
  );
}

function FileInspector({ tree, selectedFile, onOpenFile }: { tree: RepoTreeNode | null; selectedFile: { path: string; content: string } | null; onOpenFile: (path: string) => void }) {
  const [query, setQuery] = useState("");
  return (
    <div className="inspector-stack">
      <div className="search-field">
        <Search size={14} />
        <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search files" />
      </div>
      <div className="repo-tree">
        {tree ? <TreeNode node={tree} query={query.toLowerCase()} onOpenFile={onOpenFile} /> : <p className="empty">No repository loaded.</p>}
      </div>
      <div className="file-preview-head">
        <FileCode2 size={15} />
        <span>{selectedFile?.path || "No file selected"}</span>
      </div>
      <pre className="code-preview">{selectedFile?.content ?? "Open a file from the repository tree."}</pre>
    </div>
  );
}

function MemoryInspector({
  projectBrain,
  architectureMemory,
  semanticMemory,
  repositoryRag,
  knowledgeGraph,
  adrs
}: {
  projectBrain?: Record<string, unknown> | null;
  architectureMemory?: Record<string, unknown> | null;
  semanticMemory?: Record<string, unknown> | null;
  repositoryRag?: Record<string, unknown> | null;
  knowledgeGraph?: Record<string, unknown> | null;
  adrs: Array<Record<string, unknown>>;
}) {
  const stats = objectValue(knowledgeGraph?.stats);
  const nodes = Array.isArray(knowledgeGraph?.nodes) ? (knowledgeGraph.nodes as Array<Record<string, unknown>>) : [];
  const semanticStats = objectValue(semanticMemory?.stats);
  const ragIndex = objectValue(repositoryRag?.index);
  return (
    <div className="inspector-stack">
      <PanelTitle icon={<Brain size={15} />} title="Project brain" subtitle={stringValue(projectBrain?.updated_at) || "local"} />
      <div className="memory-card">
        <strong>Architecture</strong>
        <p>{asList(projectBrain?.architecture_summaries).slice(-1)[0] || stringValue(architectureMemory?.architecture_summary) || "No architecture memory recorded yet."}</p>
      </div>
      <div className="memory-grid">
        <MemoryMetric label="semantic" value={semanticStats.items ?? 0} />
        <MemoryMetric label="rag files" value={ragIndex.indexed_files ?? 0} />
        <MemoryMetric label="graph nodes" value={stats.nodes ?? 0} />
        <MemoryMetric label="adrs" value={adrs.length} />
      </div>
      <KnowledgeGraphPreview nodes={nodes} />
      <FileList title="Decisions" files={asList(projectBrain?.decisions)} />
      <FileList title="Repairs" files={asList(projectBrain?.repairs)} />
      <FileList title="Important modules" files={asList(architectureMemory?.important_modules)} />
      <FileList title="ADR Explorer" files={adrs.map((adr) => `${adr.title ?? "ADR"}: ${adr.decision ?? ""}`)} />
    </div>
  );
}

function KnowledgeGraphPreview({ nodes }: { nodes: Array<Record<string, unknown>> }) {
  const visible = nodes.slice(0, 10);
  return (
    <div className="graph-preview">
      <svg viewBox="0 0 320 190" role="img" aria-label="Knowledge graph preview">
        <line x1="160" y1="94" x2="70" y2="52" />
        <line x1="160" y1="94" x2="258" y2="54" />
        <line x1="160" y1="94" x2="82" y2="145" />
        <line x1="160" y1="94" x2="245" y2="142" />
        <circle cx="160" cy="94" r="25" className="node-core" />
        <circle cx="70" cy="52" r="16" />
        <circle cx="258" cy="54" r="16" />
        <circle cx="82" cy="145" r="16" />
        <circle cx="245" cy="142" r="16" />
      </svg>
      <div className="graph-node-list">
        {visible.length ? visible.map((node, index) => <span key={index}>{stringValue(node.label) || stringValue(node.kind) || `node-${index}`}</span>) : <span>No graph nodes yet</span>}
      </div>
    </div>
  );
}

function BottomDock({
  tab,
  setTab,
  height,
  setHeight,
  logs,
  tests,
  convergence,
  runtime,
  git,
  commitMessage,
  setCommitMessage,
  onCommit,
  onRollback,
  replayEvents,
  releaseReport,
  acceptance,
  build,
  visual,
  quality
}: {
  tab: DockTab;
  setTab: (tab: DockTab) => void;
  height: number;
  setHeight: (height: number) => void;
  logs: string[];
  tests?: Record<string, unknown>;
  convergence?: Record<string, unknown>;
  runtime?: Record<string, unknown>;
  git?: Record<string, unknown> | null;
  commitMessage: string;
  setCommitMessage: (value: string) => void;
  onCommit: () => void;
  onRollback: () => void;
  replayEvents: Array<Record<string, unknown>>;
  releaseReport?: Record<string, unknown> | null;
  acceptance?: Record<string, unknown> | null;
  build?: Record<string, unknown> | null;
  visual?: Record<string, unknown> | null;
  quality?: Record<string, unknown> | null;
}) {
  return (
    <section className="bottom-dock" style={{ height }}>
      <div className="dock-tabs">
        <DockButton active={tab === "logs"} onClick={() => setTab("logs")} icon={<Terminal size={14} />}>Logs</DockButton>
        <DockButton active={tab === "tests"} onClick={() => setTab("tests")} icon={<TestTube2 size={14} />}>Tests</DockButton>
        <DockButton active={tab === "runtime"} onClick={() => setTab("runtime")} icon={<MemoryStick size={14} />}>Runtime</DockButton>
        <DockButton active={tab === "git"} onClick={() => setTab("git")} icon={<GitBranch size={14} />}>Git</DockButton>
        <input
          aria-label="Dock height"
          className="dock-resizer"
          type="range"
          min={220}
          max={520}
          value={height}
          onChange={(event) => setHeight(Number(event.target.value))}
        />
      </div>
      <div className="dock-content">
        {tab === "logs" ? <LogStream logs={logs} /> : null}
        {tab === "tests" ? <TestDock tests={tests} convergence={convergence} acceptance={acceptance} build={build} visual={visual} quality={quality} releaseReport={releaseReport} /> : null}
        {tab === "runtime" ? <RuntimeDock runtime={runtime} /> : null}
        {tab === "git" ? <GitDock git={git} commitMessage={commitMessage} setCommitMessage={setCommitMessage} onCommit={onCommit} onRollback={onRollback} replayEvents={replayEvents} /> : null}
      </div>
    </section>
  );
}

function LogStream({ logs, compact = false }: { logs: string[]; compact?: boolean }) {
  const prioritized = useMemo(() => {
    const execution = logs.filter((line) => /^\d|^\[/.test(line) && /\[(RUN_START|REPOSITORY_SCAN|PLANNING|CODER|SYNTH|JUDGE|PATCH|TEST|REPAIR|CONVERGED|FAILED|SCHEMA|INFER|MODEL_READY|SWAP)\]/.test(line));
    const rest = logs.filter((line) => !execution.includes(line));
    return [...execution.slice(-120), ...rest.slice(-60)];
  }, [logs]);
  return (
    <pre className={cn("log-stream", compact && "compact")}>
      {prioritized.length ? prioritized.join("\n") : "[READY] waiting for Forge execution events"}
    </pre>
  );
}

function TestDock({
  tests,
  convergence,
  acceptance,
  build,
  visual,
  quality,
  releaseReport
}: {
  tests?: Record<string, unknown>;
  convergence?: Record<string, unknown>;
  acceptance?: Record<string, unknown> | null;
  build?: Record<string, unknown> | null;
  visual?: Record<string, unknown> | null;
  quality?: Record<string, unknown> | null;
  releaseReport?: Record<string, unknown> | null;
}) {
  return (
    <div className="dock-grid">
      <MetricCard label="test status" value={tests?.status ?? "idle"} icon={<TestTube2 size={16} />} />
      <MetricCard label="passing" value={tests?.passing ?? 0} icon={<Check size={16} />} />
      <MetricCard label="failing" value={tests?.failing ?? 0} icon={<XCircle size={16} />} />
      <MetricCard label="repairs" value={convergence?.current_repair_attempt ?? tests?.repair_attempts ?? 0} icon={<RefreshCw size={16} />} />
      <MetricCard label="acceptance" value={formatPass(acceptance?.passed)} icon={<ListChecks size={16} />} />
      <MetricCard label="build" value={formatPass(build?.passed)} icon={<Terminal size={16} />} />
      <MetricCard label="visual" value={formatPass(visual?.passed)} icon={<LayoutDashboard size={16} />} />
      <MetricCard label="quality" value={quality?.overall ?? "pending"} icon={<Sparkles size={16} />} />
      <pre className="release-report">{releaseReport ? JSON.stringify(releaseReport, null, 2) : "Release report will appear after validation."}</pre>
    </div>
  );
}

function RuntimeDock({ runtime }: { runtime?: Record<string, unknown> }) {
  const diagnostics = objectValue(runtime?.diagnostics);
  return (
    <div className="dock-grid">
      <MetricCard label="active runtime" value={runtime?.active_runtime ?? "idle"} icon={<Terminal size={16} />} />
      <MetricCard label="model" value={runtime?.active_model ?? "none"} icon={<Bot size={16} />} />
      <MetricCard label="load status" value={diagnostics.load_status ?? "idle"} icon={<Activity size={16} />} />
      <MetricCard label="fallback" value={diagnostics.fallback_status ?? "none"} icon={<RefreshCw size={16} />} />
      <MetricCard label="VRAM required" value={formatMegabytes(diagnostics.vram_required)} icon={<MemoryStick size={16} />} />
      <MetricCard label="VRAM free" value={formatMegabytes(diagnostics.free_vram)} icon={<MemoryStick size={16} />} />
      <MetricCard label="PID" value={runtime?.pid ?? "pending"} icon={<Code2 size={16} />} />
      <MetricCard label="health" value={runtime?.health ?? "unknown"} icon={<Check size={16} />} />
      <pre className="runtime-json">{JSON.stringify(runtime ?? {}, null, 2)}</pre>
    </div>
  );
}

function GitDock({
  git,
  commitMessage,
  setCommitMessage,
  onCommit,
  onRollback,
  replayEvents
}: {
  git?: Record<string, unknown> | null;
  commitMessage: string;
  setCommitMessage: (value: string) => void;
  onCommit: () => void;
  onRollback: () => void;
  replayEvents: Array<Record<string, unknown>>;
}) {
  const status = objectValue(git?.status);
  const changedFiles = Array.isArray(git?.changed_files) ? (git.changed_files as Array<Record<string, unknown>>) : [];
  const history = Array.isArray(git?.history) ? (git.history as Array<Record<string, unknown>>) : [];
  return (
    <div className="git-dock">
      <div className="git-actions">
        <MetricCard label="branch" value={status.branch ?? "unknown"} icon={<GitBranch size={16} />} />
        <MetricCard label="dirty" value={status.is_dirty ? "yes" : "no"} icon={<GitCompare size={16} />} />
        <Input value={commitMessage} onChange={(event) => setCommitMessage(event.target.value)} />
        <Button variant="secondary" onClick={onCommit}><GitCommit size={15} /> Commit</Button>
        <Button variant="danger" onClick={onRollback}><XCircle size={15} /> Rollback</Button>
      </div>
      <div className="git-columns">
        <FileList title="Changed files" files={changedFiles.map((file) => `${file.status ?? "M"} ${file.path ?? ""}`)} />
        <FileList title="Commit history" files={history.map((commit) => `${String(commit.sha).slice(0, 7)} ${commit.subject ?? ""}`)} />
        <FileList title="Replay events" files={replayEvents.map((event) => `${event.stage ?? ""}: ${event.message ?? ""}`)} />
      </div>
    </div>
  );
}

function CommandPalette({
  open,
  objective,
  setObjective,
  repositoryRoot,
  setRepositoryRoot,
  onClose,
  onRun,
  onOpenLogs
}: {
  open: boolean;
  objective: string;
  setObjective: (value: string) => void;
  repositoryRoot: string;
  setRepositoryRoot: (value: string) => void;
  onClose: () => void;
  onRun: () => void;
  onOpenLogs: () => void;
}) {
  if (!open) return null;
  return (
    <div className="palette-backdrop" onClick={onClose}>
      <div className="command-palette" onClick={(event) => event.stopPropagation()}>
        <div className="palette-search">
          <Command size={17} />
          <input value={objective} onChange={(event) => setObjective(event.target.value)} autoFocus />
          <button onClick={onClose}><X size={16} /></button>
        </div>
        <div className="palette-section">
          <small>Repository</small>
          <Input value={repositoryRoot} onChange={(event) => setRepositoryRoot(event.target.value)} />
        </div>
        <div className="palette-actions">
          <button onClick={onRun}><Play size={15} /> Run objective</button>
          <button onClick={onOpenLogs}><Terminal size={15} /> Open logs</button>
          {promptTemplates.map((template) => <button key={template} onClick={() => setObjective(template)}><Sparkles size={15} /> {template}</button>)}
        </div>
      </div>
    </div>
  );
}

function TreeNode({ node, query, onOpenFile }: { node: RepoTreeNode; query: string; onOpenFile: (path: string) => void }) {
  const [open, setOpen] = useState(true);
  const visible = !query || node.name.toLowerCase().includes(query) || node.path.toLowerCase().includes(query) || node.children.some((child) => child.path.toLowerCase().includes(query));
  if (!visible) return null;
  return (
    <div className="tree-node">
      <button onClick={() => node.type === "file" ? onOpenFile(node.path) : setOpen(!open)}>
        {node.type === "directory" ? (open ? <ChevronDown size={13} /> : <ChevronRight size={13} />) : <FileCode2 size={13} />}
        <span>{node.name}</span>
      </button>
      {node.type === "directory" && open ? (
        <div className="tree-children">
          {node.children.map((child) => <TreeNode key={child.path} node={child} query={query} onOpenFile={onOpenFile} />)}
        </div>
      ) : null}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function SidebarSection({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="sidebar-section">
      <h3>{icon}{title}</h3>
      {children}
    </section>
  );
}

function RailIcon({ icon, label, collapsed }: { icon: React.ReactNode; label: string; collapsed: boolean }) {
  return <div className="rail-icon">{icon}{collapsed ? null : <span>{label}</span>}</div>;
}

function PanelTitle({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle?: string }) {
  return (
    <div className="panel-title">
      <div>{icon}<strong>{title}</strong></div>
      {subtitle ? <span>{subtitle}</span> : null}
    </div>
  );
}

function FileList({ title, files }: { title: string; files: string[] }) {
  return (
    <div className="file-list">
      <div className="file-list-title">{title}<span>{files.length}</span></div>
      {files.length ? files.slice(0, 24).map((file, index) => <div key={`${title}-${index}-${file}`}><FileCode2 size={13} /> <span>{file}</span></div>) : <p className="empty">None</p>}
    </div>
  );
}

function MetricCard({ label, value, icon }: { label: string; value: unknown; icon: React.ReactNode }) {
  return (
    <div className="metric-card">
      <div>{icon}<span>{label}</span></div>
      <strong>{String(value ?? "unknown")}</strong>
    </div>
  );
}

function MetricInline({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="metric-inline">
      <span>{label}</span>
      <strong>{String(value ?? "unknown")}</strong>
    </div>
  );
}

function MemoryMetric({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="memory-metric">
      <strong>{String(value ?? 0)}</strong>
      <span>{label}</span>
    </div>
  );
}

function StatGrid({ items }: { items: Array<[string, unknown]> }) {
  return (
    <div className="stat-grid">
      {items.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{String(value ?? "unknown")}</strong>
        </div>
      ))}
    </div>
  );
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return <button className={cn(active && "active")} onClick={onClick}>{children}</button>;
}

function DockButton({ active, onClick, icon, children }: { active: boolean; onClick: () => void; icon: React.ReactNode; children: React.ReactNode }) {
  return <button className={cn(active && "active")} onClick={onClick}>{icon}{children}</button>;
}

function StatusPill({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "success" | "warning" | "danger" | "accent" }) {
  return <span className={cn("status-pill", tone)}>{children}</span>;
}

function generatedFileNames(artifacts: ArtifactSummary[], plan?: Record<string, unknown> | null): string[] {
  const files = new Set(asList(plan?.files_to_create));
  artifacts.forEach((artifact) => {
    try {
      const parsed = JSON.parse(artifact.content) as Record<string, unknown>;
      const generated = objectValue(parsed.files);
      Object.keys(generated).forEach((path) => files.add(path));
    } catch {
      // Artifacts are sometimes summaries; non-JSON output is already visible in preview.
    }
  });
  return Array.from(files);
}

function phasePercent(phase: string) {
  const index = phases.indexOf(phase);
  if (phase === "CONVERGED") return 100;
  if (phase === "FAILED") return 100;
  return index >= 0 ? Math.round(((index + 1) / phases.length) * 100) : 3;
}

function inferStageStatus(name: string, phase: string) {
  const current = phases.indexOf(phase);
  const index = phases.indexOf(name);
  if (phase === "CONVERGED") return "completed";
  if (phase === "FAILED" && index === current) return "failed";
  if (index < current) return "completed";
  if (index === current) return "active";
  return "pending";
}

function formatPhase(value: string) {
  return value.toLowerCase().replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatPass(value: unknown) {
  return value === undefined || value === null ? "pending" : value ? "passed" : "failed";
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
