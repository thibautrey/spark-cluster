import { useEffect, useRef, useState } from "react";
import { installAuthPrompt, api, saveToken, clearToken } from "@/lib/api";
import { errorMessage } from "@/lib/utils";
import { Sheet, Field, Input, Button, Message } from "./common";
export function AuthDialog() {
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const waiters = useRef<
    { resolve: (token: string) => void; reject: (e: Error) => void }[]
  >([]);
  useEffect(
    () =>
      installAuthPrompt(
        (message) =>
          new Promise((resolve, reject) => {
            waiters.current.push({ resolve, reject });
            setMessage(message);
            setError("");
            setOpen(true);
          }),
      ),
    [],
  );
  function close() {
    if (busy) return;
    waiters.current
      .splice(0)
      .forEach((w) =>
        w.reject(new Error("Administrator authentication cancelled.")),
      );
    setToken("");
    setOpen(false);
  }
  return (
    <Sheet
      open={open}
      onOpenChange={(v) => {
        if (!v) close();
      }}
      title="Controller token required"
      description={message}
    >
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          setError("");
          try {
            await api("/api/v1/controller/verify", {
              headers: { Authorization: `Bearer ${token.trim()}` },
            });
            saveToken(token.trim());
            waiters.current.splice(0).forEach((w) => w.resolve(token.trim()));
            setToken("");
            setOpen(false);
          } catch (e) {
            clearToken();
            setError(errorMessage(e));
          } finally {
            setBusy(false);
          }
        }}
      >
        <Field label="Administration token">
          <Input
            type="password"
            autoComplete="off"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            required
          />
        </Field>
        <Message error={error} />
        <div className="actions">
          <Button
            type="button"
            variant="outline"
            onClick={close}
            disabled={busy}
          >
            Cancel
          </Button>
          <Button disabled={busy || !token.trim()}>Continue</Button>
        </div>
      </form>
    </Sheet>
  );
}
