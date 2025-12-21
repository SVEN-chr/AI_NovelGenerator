import { Card } from "@/components/ui/card";

interface StatCardProps {
  label: string;
  value: string;
  helper: string;
}

export function StatCard({ label, value, helper }: StatCardProps) {
  return (
    <Card className="space-y-3">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="text-3xl font-semibold text-white">{value}</p>
      <p className="text-xs text-slate-400">{helper}</p>
    </Card>
  );
}
