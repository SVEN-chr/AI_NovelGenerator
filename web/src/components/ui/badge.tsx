import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  tone?: "success" | "warning" | "neutral";
}

export function Badge({ className, tone = "neutral", ...props }: BadgeProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold",
        tone === "success" && "bg-emerald-500/20 text-emerald-200",
        tone === "warning" && "bg-amber-500/20 text-amber-200",
        tone === "neutral" && "bg-slate-500/20 text-slate-200",
        className
      )}
      {...props}
    />
  );
}
