"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Archive,
  Bot,
  CheckCircle2,
  Clock,
  Code2,
  FileCode2,
  GitBranch,
  GitCompare,
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
  ControlSnapshot,
  RepoTreeNode,
  controlRun,
  createRun,
  fetchRepoFile,
  fetchRepoTree,
  fetchSnapshot
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
  const [execute, setExecute] = useState(false);
  const [snapshot, setSnapshot] = useState<ControlSnapshot | null>(null);
  const [tree, setTree] = useState<RepoTreeNode | null>(null);
  const [selectedFile, setSelectedFile] = useState<{ path: string; content: string } | null>(null);
  const [artifactQuery, setArtifactQuery] = useState("");
  const [artifactRole, setArtifactRole] = useState("ALL");
  const [selectedArtifact, setSelectedArtifact] = useState<ArtifactSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const activeRun = snapshot?.active_run ?? null;

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const next = await fetchSnapshot(repositoryRoot);
        if (!cancelled) {
          setSnapshot(next);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
    }
    load();
    const interval = window.setInterval(load, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [repositoryRoot]);

  useEffect(() => {
    let cancelled = false;
    async function loadTree() {
      try {
        const next = await fetchRepoTree(repositoryRoot);
        if (!cancelled) setTree(next);
      } catch {
        if (!cancelled) setTree(null);
      }
    }
    loadTree();
    return () => {
      cancelled = true;
    };
  }, [repositoryRoot]);

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

  return (
    <main className="min-h-screen px-5 py-5">
      <div className="mx-auto flex max-w-[1800px] flex-col gap-5">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-normal">Forge Control Center</h1>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-muted">
              <Badge tone={activeRun ? "accent" : "neutral"}>{activeRun?.status ?? "idle"}</Badge>
              <span>{snapshot?.generated_at ? new Date(snapshot.generated_at).toLocaleTimeString() : "waiting"}</span>
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

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[420px_1fr_420px]">
          <section className="flex flex-col gap-5">
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
            <RuntimeMonitor runtime={snapshot?.runtime} />
            <TestResults tests={snapshot?.tests} />
          </section>

          <section className="flex flex-col gap-5">
            <LiveCourtroom roles={snapshot?.courtroom ?? []} />
            <ExecutionTimeline timeline={snapshot?.timeline ?? []} />
            <PatchViewer patch={snapshot?.patch} />
            <LogsPanel logs={snapshot?.logs ?? []} />
          </section>

          <section className="flex flex-col gap-5">
            <ArtifactExplorer
              artifacts={filteredArtifacts}
              query={artifactQuery}
              setQuery={setArtifactQuery}
              role={artifactRole}
              setRole={setArtifactRole}
              selected={selectedArtifact}
              setSelected={setSelectedArtifact}
            />
            <RepositoryExplorer tree={tree} selectedFile={selectedFile} onOpenFile={openFile} />
            <ConversationView items={snapshot?.conversation ?? []} />
          </section>
        </div>
      </div>
    </main>
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
        <div className="flex items-center justify-between rounded-md border border-border px-3 py-2">
          <span className="text-sm">Execute autonomous run</span>
          <input
            type="checkbox"
            checked={props.execute}
            onChange={(event) => props.setExecute(event.target.checked)}
            className="h-4 w-4 accent-cyan-700"
          />
        </div>
        <div className="grid grid-cols-4 gap-2">
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
          <div key={String(role.role)} className="rounded-md border border-border p-3">
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
            className="h-9 rounded-md border border-border bg-white px-2 text-sm"
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
                "block w-full border-b border-border px-3 py-2 text-left text-sm last:border-b-0 hover:bg-slate-50",
                props.selected?.artifact_id === artifact.artifact_id && "bg-cyan-50"
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
        <div className="max-h-60 overflow-auto rounded-md border border-border p-2 text-sm">
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
        className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-slate-100"
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
  return (
    <Card>
      <CardHeader title="Runtime Monitor" action={<Terminal size={18} className="text-accent" />} />
      <dl className="space-y-2 p-4 text-sm">
        <Metric label="active runtime" value={runtime?.active_runtime ?? "idle"} />
        <Metric label="model" value={runtime?.active_model ?? "none"} />
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
            <div key={index} className="rounded-md border border-border p-3 text-sm">
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
