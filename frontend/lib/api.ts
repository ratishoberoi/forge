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
  repository_id?: string | null;
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
  phase?: string;
  telemetry?: string[];
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
  active_objective?: string | null;
  objective_source?: string;
  objective_classification?: string | null;
  generated_plan?: Record<string, unknown> | null;
  active_repository_id: string | null;
  active_repository_root: string | null;
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
  completed_nodes?: Array<Record<string, unknown>>;
  running_node?: Record<string, unknown> | null;
  blocked_nodes?: Array<Record<string, unknown>>;
  failed_nodes?: Array<Record<string, unknown>>;
  compressed_context: Record<string, unknown> | null;
  objective_memory: Array<Record<string, unknown>>;
  bootstrap: Record<string, unknown> | null;
  acceptance: Record<string, unknown> | null;
  build_validation: Record<string, unknown> | null;
  visual_validation: Record<string, unknown> | null;
  quality_score: Record<string, unknown> | null;
  release_report: Record<string, unknown> | null;
  project_brain: Record<string, unknown> | null;
  semantic_memory: Record<string, unknown> | null;
  repository_rag: Record<string, unknown> | null;
  context_assembly: Record<string, unknown> | null;
  knowledge_graph: Record<string, unknown> | null;
  adrs: Array<Record<string, unknown>>;
  tool_activity: Record<string, unknown> | null;
  runtime_diagnostics?: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
};

const API_BASE = process.env.NEXT_PUBLIC_FORGE_API_URL ?? "";

export type ApiDiagnostics = {
  method: string;
  url: string;
  status?: number;
  duration_ms: number;
  attempts: number;
};

export class ApiRequestError extends Error {
  diagnostics: ApiDiagnostics;

  constructor(message: string, diagnostics: ApiDiagnostics) {
    super(message);
    this.name = "ApiRequestError";
    this.diagnostics = diagnostics;
  }
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function requestJson<T>(path: string, init?: RequestInit, retries = 2): Promise<T> {
  const url = `${API_BASE}${path}`;
  const method = init?.method ?? "GET";
  let lastError: unknown = null;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const started = performance.now();
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), method === "GET" ? 15000 : 30000);
    try {
      const response = await fetch(url, {
        cache: "no-store",
        ...init,
        signal: controller.signal,
        headers: {
          ...(init?.headers ?? {})
        }
      });
      const duration = Math.round(performance.now() - started);
      console.info("[Forge API]", method, url, response.status, `${duration}ms`, `attempt=${attempt + 1}`);
      if (!response.ok) {
        const text = await response.text();
        throw new ApiRequestError(text || response.statusText, {
          method,
          url,
          status: response.status,
          duration_ms: duration,
          attempts: attempt + 1
        });
      }
      return response.json() as Promise<T>;
    } catch (error) {
      const duration = Math.round(performance.now() - started);
      lastError = error;
      console.error("[Forge API] request failed", method, url, error, `attempt=${attempt + 1}`);
      if (attempt >= retries) {
        throw error instanceof ApiRequestError
          ? error
          : new ApiRequestError(error instanceof Error ? error.message : String(error), {
              method,
              url,
              duration_ms: duration,
              attempts: attempt + 1
            });
      }
      await sleep(300 * 2 ** attempt);
    } finally {
      window.clearTimeout(timeout);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
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
  return requestJson<Record<string, unknown>>(
    "/api/control/health"
  );
}

export async function importRepository(payload: {
  path: string;
  repository_name?: string;
  set_active?: boolean;
  refresh_intelligence?: boolean;
}) {
  return requestJson<Record<string, unknown>>("/api/control/workspaces/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function validateRepositoryPath(path: string) {
  return requestJson<Record<string, unknown>>("/api/control/workspaces/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path })
  });
}

export async function browseWorkspace(path?: string) {
  const params = new URLSearchParams();
  if (path) params.set("path", path);
  return requestJson<Record<string, unknown>>(`/api/control/workspaces/browse?${params}`);
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
