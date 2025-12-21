import { Card } from "@/components/ui/card";

interface SectionCardProps {
  title: string;
  description: string;
}

export function SectionCard({ title, description }: SectionCardProps) {
  return (
    <Card className="space-y-3">
      <p className="text-sm text-slate-400">{title}</p>
      <p className="text-base text-white">{description}</p>
    </Card>
  );
}
