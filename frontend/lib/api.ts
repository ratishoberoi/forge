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
  convergence: Record<string, unknown>;
  logs: string[];
  conversation: Array<Record<string, unknown>>;
  repository_summary: Record<string, unknown> | null;
  architecture_summary: string | null;
  execution_plan: Record<string, unknown> | null;
  repositories: Array<Record<string, unknown>>;
  active_repository: Record<string, unknown> | null;
  git: Record<string, unknown> | null;
  run_history: Array<Record<string, unknown>>;
  queued_tasks: Array<Record<string, unknown>>;
  architecture_memory: Record<string, unknown> | null;
  task_plan: Record<string, unknown> | null;
  execution_graph: Record<string, unknown> | null;
  compressed_context: Record<string, unknown> | null;
  objective_memory: Array<Record<string, unknown>>;
  bootstrap: Record<string, unknown> | null;
  acceptance: Record<string, unknown> | null;
  build_validation: Record<string, unknown> | null;
  visual_validation: Record<string, unknown> | null;
  quality_score: Record<string, unknown> | null;
  release_report: Record<string, unknown> | null;
};

const API_BASE = process.env.NEXT_PUBLIC_FORGE_API_URL ?? "";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const started = performance.now();
  try {
    const response = await fetch(url, {
      cache: "no-store",
      ...init,
      headers: {
        ...(init?.headers ?? {})
      }
    });
    const duration = Math.round(performance.now() - started);
    console.info("[Forge API]", init?.method ?? "GET", url, response.status, `${duration}ms`);
    if (!response.ok) throw new Error(await response.text());
    return response.json() as Promise<T>;
  } catch (error) {
    console.error("[Forge API] request failed", init?.method ?? "GET", url, error);
    throw error;
  }
}

export async function fetchSnapshot(repositoryRoot?: string): Promise<ControlSnapshot> {
  const params = new URLSearchParams();
  if (repositoryRoot) params.set("repository_root", repositoryRoot);
  return requestJson<ControlSnapshot>(`/api/control/snapshot?${params}`);
}

export async function fetchRepoTree(root: string): Promise<RepoTreeNode> {
  const params = new URLSearchParams({ root, depth: "4" });
  return requestJson<RepoTreeNode>(`/api/control/repository/tree?${params}`);
}

export async function fetchRepoFile(root: string, path: string) {
  const params = new URLSearchParams({ root, path });
  return requestJson<{ path: string; content: string; size_bytes: number }>(
    `/api/control/repository/file?${params}`
  );
}

export async function createRun(payload: {
  objective: string;
  repository_root: string;
  repository_id?: string | null;
  target_file: string;
  test_command: string[];
  max_iterations: number;
  execute: boolean;
}) {
  return requestJson<RunSummary>("/api/control/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function controlRun(runId: string, action: "pause" | "resume" | "stop") {
  return requestJson<RunSummary>(`/api/control/runs/${runId}/${action}`, {
    method: "POST"
  });
}

export async function fetchDashboardHealth() {
  return requestJson<{ status: string; backend: string; generated_at: string }>(
    "/api/control/health"
  );
}

export async function importRepository(payload: {
  path: string;
  repository_name?: string;
  set_active?: boolean;
}) {
  return requestJson<Record<string, unknown>>("/api/control/workspaces/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function cloneRepository(payload: {
  source: string;
  repository_name?: string;
  set_active?: boolean;
}) {
  return requestJson<Record<string, unknown>>("/api/control/workspaces/clone", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function switchRepository(repositoryId: string) {
  return requestJson<Record<string, unknown>>(
    `/api/control/workspaces/repositories/${repositoryId}/switch`,
    { method: "POST" }
  );
}

export async function refreshRepository(repositoryId: string) {
  return requestJson<Record<string, unknown>>(
    `/api/control/workspaces/repositories/${repositoryId}/refresh`,
    { method: "POST" }
  );
}

export async function commitRepository(payload: {
  repository_root?: string;
  repository_id?: string;
  message: string;
}) {
  return requestJson<Record<string, unknown>>("/api/control/git/commit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function rollbackRepository(payload: {
  repository_root?: string;
  repository_id?: string;
  target?: string;
  clean_untracked?: boolean;
}) {
  return requestJson<Record<string, unknown>>("/api/control/git/rollback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function fetchRunReplay(runId: string) {
  return requestJson<{ run_id: string; events: Array<Record<string, unknown>> }>(
    `/api/control/history/runs/${runId}/replay`
  );
}

export async function runBenchmarks(payload: {
  root?: string;
  cleanup?: boolean;
  cases?: Array<Record<string, unknown>>;
}) {
  return requestJson<Record<string, unknown>>("/api/control/benchmarks/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}
