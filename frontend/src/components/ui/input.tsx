import type { ComponentProps } from 'react';
import { cn } from '@/lib/utils';
export function Input({className,type,...props}:ComponentProps<'input'>){return <input type={type} className={cn('flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50',className)} {...props}/>;}
export function Textarea({className,...props}:ComponentProps<'textarea'>){return <textarea className={cn('min-h-24 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',className)} {...props}/>;}
