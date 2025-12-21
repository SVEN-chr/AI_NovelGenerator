"use client";

import { Button } from "@/components/ui/button";
import { SectionCard } from "@/components/novel/section-card";
import { useNovelStore } from "@/store/useNovelStore";

export default function WorldbuildingPage() {
  const world = useNovelStore((state) => state.worldbuilding);

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">世界观管理</h2>
          <p className="text-sm text-slate-400">
            结构化记录世界观，支持 AI 重新生成与手动编辑
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary">手动编辑</Button>
          <Button>AI 重新生成</Button>
        </div>
      </header>

      <section className="grid gap-4 md:grid-cols-2">
        <SectionCard title="时代背景" description={world.era} />
        <SectionCard title="权力/科技体系" description={world.powerSystem} />
        <SectionCard title="文化习俗" description={world.culture} />
        <SectionCard title="地理环境" description={world.geography} />
        <SectionCard title="历史事件" description={world.history} />
        <SectionCard title="补充说明" description={world.notes} />
      </section>
    </div>
  );
}
