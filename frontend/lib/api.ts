export type RunStatus =
  | "queued"
  | "running"
  | "paused"
  | "stopping"
  | "completed"
  | "failed"
  | "cancelled";

export type RunSummary = {
  id: string;
  objective: string;
  repository_root: string;
  target_file: string;
  test_command: string[];
  max_iterations: number;
  status: RunStatus;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  result: Record<string, unknown> | null;
};

export type ArtifactSummary = {
  artifact_id: string;
  role: string;
  round_id: number;
  task: string;
  content: string;
  created_at: string;
  metadata: Record<string, unknown>;
};

export type RepoTreeNode = {
  name: string;
  path: string;
  type: "file" | "directory";
  size_bytes: number | null;
  children: RepoTreeNode[];
};

export type ControlSnapshot = {
  generated_at: string;
  active_run: RunSummary | null;
  runs: RunSummary[];
  courtroom: Array<Record<string, unknown>>;
  timeline: Array<{ name: string; status: string }>;
  runtime: Record<string, unknown>;
  artifacts: ArtifactSummary[];
  patch: {
    diff: string;
    stat: string;
    changed_files: string[];
    repository_root: string;
    error?: string;
  };
  tests: Record<string, unknown>;
  logs: string[];
  conversation: Array<Record<string, unknown>>;
};

const API_BASE = process.env.NEXT_PUBLIC_FORGE_API_URL ?? "http://localhost:8000";

export async function fetchSnapshot(repositoryRoot?: string): Promise<ControlSnapshot> {
  const params = new URLSearchParams();
  if (repositoryRoot) params.set("repository_root", repositoryRoot);
  const response = await fetch(`${API_BASE}/api/control/snapshot?${params}`, {
    cache: "no-store"
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function fetchRepoTree(root: string): Promise<RepoTreeNode> {
  const params = new URLSearchParams({ root, depth: "4" });
  const response = await fetch(`${API_BASE}/api/control/repository/tree?${params}`, {
    cache: "no-store"
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function fetchRepoFile(root: string, path: string) {
  const params = new URLSearchParams({ root, path });
  const response = await fetch(`${API_BASE}/api/control/repository/file?${params}`, {
    cache: "no-store"
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{ path: string; content: string; size_bytes: number }>;
}

export async function createRun(payload: {
  objective: string;
  repository_root: string;
  target_file: string;
  test_command: string[];
  max_iterations: number;
  execute: boolean;
}) {
  const response = await fetch(`${API_BASE}/api/control/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<RunSummary>;
}

export async function controlRun(runId: string, action: "pause" | "resume" | "stop") {
  const response = await fetch(`${API_BASE}/api/control/runs/${runId}/${action}`, {
    method: "POST"
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<RunSummary>;
}
