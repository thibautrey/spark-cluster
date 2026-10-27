import * as Menu from '@radix-ui/react-dropdown-menu';
import type { ComponentProps } from 'react';
import { cn } from '@/lib/utils';
export const DropdownMenu=Menu.Root;
export const DropdownMenuTrigger=Menu.Trigger;
export function DropdownMenuContent({className,...props}:ComponentProps<typeof Menu.Content>){return <Menu.Portal><Menu.Content sideOffset={6} className={cn('z-50 min-w-48 rounded-md border border-border bg-popover p-1 text-popover-foreground shadow-md',className)} {...props}/></Menu.Portal>;}
export function DropdownMenuItem({className,...props}:ComponentProps<typeof Menu.Item>){return <Menu.Item className={cn('relative flex cursor-default select-none items-center rounded-sm px-3 py-2 text-sm outline-none focus:bg-accent data-[disabled]:pointer-events-none data-[disabled]:opacity-50',className)} {...props}/>;}
export function DropdownMenuCheckboxItem({className,...props}:ComponentProps<typeof Menu.CheckboxItem>){return <Menu.CheckboxItem className={cn('flex cursor-default items-center gap-2 rounded-sm px-3 py-2 text-sm outline-none focus:bg-accent',className)} {...props}/>;}
