import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import type { NovelProject } from "@/types/novel";

const statusMap: Record<NovelProject["status"], { label: string; tone: "success" | "warning" | "neutral" }> = {
  draft: { label: "草稿", tone: "neutral" },
  writing: { label: "写作中", tone: "success" },
  paused: { label: "暂停", tone: "warning" },
  completed: { label: "已完成", tone: "success" }
};

export function ProjectCard({ project }: { project: NovelProject }) {
  const status = statusMap[project.status];
  return (
    <Card className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-lg font-semibold text-white">{project.title}</p>
          <p className="text-xs text-slate-400">{project.genre} · {project.style}</p>
        </div>
        <Badge tone={status.tone}>{status.label}</Badge>
      </div>
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span>目标章节: {project.targetChapters}</span>
          <span>更新于 {project.updatedAt}</span>
        </div>
        <div className="h-2 rounded-full bg-slate-800">
          <div
            className="h-2 rounded-full bg-accent"
            style={{ width: `${project.progress}%` }}
          />
        </div>
        <p className="text-xs text-slate-400">完成度 {project.progress}%</p>
      </div>
    </Card>
  );
}
