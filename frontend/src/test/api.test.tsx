import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  api,
  post,
  storedToken,
  saveToken,
  installAuthPrompt,
  useAction,
} from "@/lib/api";
import { AuthDialog } from "@/components/auth";
import { Button, Message } from "@/components/common";
const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status });
describe("shared API and administrator authentication", () => {
  it("does not prompt on background 401 and removes the rejected credential", async () => {
    saveToken("expired");
    const prompt = vi.fn();
    const remove = installAuthPrompt(prompt);
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(response({ detail: "Invalid token" }, 401))),
    );
    await expect(api("/api/v1/setup")).rejects.toThrow("Invalid token");
    expect(storedToken()).toBe("");
    expect(prompt).not.toHaveBeenCalled();
    remove();
  });
  it("retries the same action after token validation without exposing its token to the URL", async () => {
    saveToken("expired");
    const prompt = vi.fn(async () => {
      saveToken("accepted");
      return "accepted";
    });
    const remove = installAuthPrompt(prompt);
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(response({}, 401))
      .mockResolvedValueOnce(response({ ok: true }));
    vi.stubGlobal("fetch", fetch);
    await post("/api/v1/pool/load", {
      model_id: "owner/model",
      cluster_id: "pool-2",
    });
    expect(prompt).toHaveBeenCalledOnce();
    expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({
      model_id: "owner/model",
      cluster_id: "pool-2",
    });
    expect(
      new Headers(fetch.mock.calls[1][1].headers).get("Authorization"),
    ).toBe("Bearer accepted");
    expect(fetch.mock.calls[1][0]).toBe("/api/v1/pool/load");
    remove();
  });
  it("preserves explicit Headers credentials during verification", async () => {
    saveToken("old");
    const fetch = vi.fn(async (..._args: [RequestInfo | URL, RequestInit?]) =>
      response({ ok: true }),
    );
    vi.stubGlobal("fetch", fetch);
    await api("/api/v1/controller/verify", {
      headers: new Headers({
        Authorization: "Bearer candidate",
        "X-Custom": "present",
      }),
    });
    expect(
      new Headers(fetch.mock.calls[0][1]?.headers).get("Authorization"),
    ).toBe("Bearer candidate");
    expect(new Headers(fetch.mock.calls[0][1]?.headers).get("X-Custom")).toBe(
      "present",
    );
  });
  it("keeps a verified token for the session when browser storage is unavailable", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("disabled");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("disabled");
    });
    saveToken("session-only");
    expect(storedToken()).toBe("session-only");
  });
  it("reports structured backend blockers", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response(
          {
            detail: {
              message: "Runtime lacks required memory",
              fit: { fits: false },
            },
          },
          409,
        ),
      ),
    );
    await expect(post("/api/v1/launches", {})).rejects.toThrow(
      "Runtime lacks required memory",
    );
  });
  it("collects concurrent administrator requests in one dialog and stores only accepted tokens", async () => {
    const user = userEvent.setup();
    const fetch = vi.fn(async (path: string, options: RequestInit) => {
      const authorization = new Headers(options.headers).get("Authorization");
      return response(
        authorization === "Bearer valid"
          ? { ok: true }
          : { detail: "Invalid model administration token" },
        authorization === "Bearer valid" ? 200 : 401,
      );
    });
    vi.stubGlobal("fetch", fetch);
    function Harness() {
      const a = useAction();
      return (
        <>
          <Button
            onClick={() =>
              void a.run(
                () => Promise.all([post("/one"), post("/two")]),
                "Actions completed",
              )
            }
          >
            Run
          </Button>
          <Message error={a.error}>{a.message}</Message>
          <AuthDialog />
        </>
      );
    }
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    render(
      <QueryClientProvider client={client}>
        <Harness />
      </QueryClientProvider>,
    );
    await user.click(screen.getByText("Run"));
    await screen.findByRole("dialog");
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    await user.type(screen.getByLabelText("Administration token"), "wrong");
    await user.click(screen.getByText("Continue"));
    await screen.findByRole("alert");
    expect(storedToken()).toBe("");
    await user.clear(screen.getByLabelText("Administration token"));
    await user.type(screen.getByLabelText("Administration token"), "valid");
    await user.click(screen.getByText("Continue"));
    await screen.findByText("Actions completed");
    expect(storedToken()).toBe("valid");
    expect(window.localStorage.getItem("spark-cluster.admin-token")).toBe(
      "valid",
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(fetch.mock.calls.filter((c) => c[0] === "/one")).toHaveLength(2);
    expect(fetch.mock.calls.filter((c) => c[0] === "/two")).toHaveLength(2);
  });
});
