import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { clearToken } from "@/lib/api";
class MemoryStorage implements Storage {
  private values = new Map<string, string>();
  get length() {
    return this.values.size;
  }
  clear() {
    this.values.clear();
  }
  getItem(key: string) {
    return this.values.get(key) ?? null;
  }
  key(index: number) {
    return [...this.values.keys()][index] ?? null;
  }
  removeItem(key: string) {
    this.values.delete(key);
  }
  setItem(key: string, value: string) {
    this.values.set(key, String(value));
  }
}
const storage = new MemoryStorage();
Object.defineProperty(window, "localStorage", {
  value: storage,
  configurable: true,
});
Object.defineProperty(globalThis, "localStorage", {
  value: storage,
  configurable: true,
});
afterEach(() => {
  cleanup();
  clearToken();
  storage.clear();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
Object.defineProperty(window, "scrollTo", { value: () => {}, writable: true });
Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
  value: () => {},
  writable: true,
});
Object.defineProperty(HTMLElement.prototype, "hasPointerCapture", {
  value: () => false,
  writable: true,
});
Object.defineProperty(HTMLElement.prototype, "setPointerCapture", {
  value: () => {},
  writable: true,
});
Object.defineProperty(HTMLElement.prototype, "releasePointerCapture", {
  value: () => {},
  writable: true,
});
