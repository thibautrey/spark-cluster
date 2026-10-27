import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));
export const number = (value: number | null | undefined, suffix = '', digits = 1) => value == null ? '—' : `${value.toLocaleString(undefined, {maximumFractionDigits:digits})}${suffix}`;
export const bytes = (value: number) => value >= 2 ** 30 ? `${number(value / 2 ** 30)} GiB` : value >= 2 ** 20 ? `${number(value / 2 ** 20)} MiB` : `${number(value / 1024)} KiB`;
export const percent = (used: number, total: number) => total ? Math.max(0, Math.min(100, used / total * 100)) : 0;
export const repoPath = (id: string) => id.split('/').map(encodeURIComponent).join('/');
export const fresh = (time: string) => Date.now() - Date.parse(time) <= 15000;
export const errorMessage = (error: unknown) => error instanceof Error ? error.message : String(error);
