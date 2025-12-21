"use client";

import { ProjectCard } from "@/components/dashboard/project-card";
import { StatCard } from "@/components/dashboard/stat-card";
import { Button } from "@/components/ui/button";
import { useNovelStore } from "@/store/useNovelStore";

export default function HomePage() {
  const projects = useNovelStore((state) => state.projects);

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">项目仪表板</h2>
          <p className="text-sm text-slate-400">
            管理小说项目，查看进度和关键指标
          </p>
        </div>
        <Button>新建项目</Button>
      </header>

      <section className="grid gap-4 md:grid-cols-3">
        <StatCard label="活跃项目" value="2" helper="草稿与写作中项目" />
        <StatCard label="已完成章节" value="18" helper="累计生成章节" />
        <StatCard label="AI配置" value="2" helper="OpenAI / Gemini" />
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        {projects.map((project) => (
          <ProjectCard key={project.id} project={project} />
        ))}
      </section>
    </div>
  );
}
