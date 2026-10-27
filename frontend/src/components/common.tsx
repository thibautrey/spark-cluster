import { useRef, useState, type ReactNode, type ComponentProps } from "react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from "./ui/dialog";
import { cn, errorMessage } from "@/lib/utils";
export { Button } from "./ui/button";
export { Input, Textarea } from "./ui/input";
export function Panel({
  title,
  actions,
  children,
  className,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("panel", className)}>
      {title && (
        <header className="panel-head">
          <h2>{title}</h2>
          <div className="actions">{actions}</div>
        </header>
      )}
      {children}
    </section>
  );
}
export function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}
export function Select({ className, ...props }: ComponentProps<"select">) {
  return (
    <select
      className={cn(
        "h-9 rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        className,
      )}
      {...props}
    />
  );
}
export function Badge({
  children,
  good = false,
}: {
  children: ReactNode;
  good?: boolean;
}) {
  return <span className={cn("badge", good && "good")}>{children}</span>;
}
export function Message({
  error,
  children,
}: {
  error?: unknown;
  children?: ReactNode;
}) {
  return error ? (
    <p role="alert" className="error">
      {errorMessage(error)}
    </p>
  ) : children ? (
    <p role="status" className="note">
      {children}
    </p>
  ) : null;
}
export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
export function Meter({ value, label }: { value: number; label: string }) {
  return (
    <div
      className="meter"
      role="progressbar"
      aria-label={label}
      aria-valuenow={Math.round(value)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <i style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}
export function Sheet({
  open,
  onOpenChange,
  title,
  description,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  children: ReactNode;
}) {
  const opener = useRef<HTMLElement | null>(null);
  if (open && !opener.current) {
    opener.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
  }
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          if (opener.current?.isConnected) {
            opener.current.focus({ preventScroll: true });
          }
          opener.current = null;
        }}
      >
        <DialogTitle className="text-xl font-semibold pr-8">
          {title}
        </DialogTitle>
        <DialogDescription className={description ? "note" : "sr-only"}>
          {description || title}
        </DialogDescription>
        {children}
      </DialogContent>
    </Dialog>
  );
}
export function Copy({
  value,
  label = "Copy",
}: {
  value: string;
  label?: string;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="outline"
      type="button"
      onClick={async () => {
        await navigator.clipboard.writeText(value);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? "Copied" : label}
    </Button>
  );
}
export function Facts({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="facts">
      {items.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}
export function Chart({
  values,
  label,
}: {
  values: (number | null)[];
  label: string;
}) {
  const max = Math.max(1, ...values.filter((v): v is number => v != null));
  let previous = false;
  const path = values
    .map((v, i) => {
      if (v == null) {
        previous = false;
        return "";
      }
      const p = `${previous ? "L" : "M"}${20 + (i / Math.max(1, values.length - 1)) * 660},${150 - (v / max) * 125}`;
      previous = true;
      return p;
    })
    .join(" ");
  return (
    <figure className="chart">
      <figcaption>{label}</figcaption>
      {values.some((v) => v != null) ? (
        <svg viewBox="0 0 700 175" role="img" aria-label={label}>
          <path d="M20 20V150H680" stroke="var(--border)" fill="none" />
          <path d={path} stroke="var(--primary)" strokeWidth="2" fill="none" />
        </svg>
      ) : (
        <Empty>No recorded measurements</Empty>
      )}
    </figure>
  );
}
