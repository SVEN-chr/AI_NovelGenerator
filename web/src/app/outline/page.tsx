"use client";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useNovelStore } from "@/store/useNovelStore";

export default function OutlinePage() {
  const outline = useNovelStore((state) => state.outline);

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">大纲管理</h2>
          <p className="text-sm text-slate-400">
            管理三幕式或起承转合结构，标注高潮与伏笔
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary">切换结构</Button>
          <Button>AI 生成大纲</Button>
        </div>
      </header>

      <section className="grid gap-4">
        {outline.map((beat) => (
          <Card key={beat.id} className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-400">{beat.act}</p>
                <p className="text-lg font-semibold text-white">{beat.title}</p>
              </div>
              <Button variant="ghost">编辑</Button>
            </div>
            <p className="text-sm text-slate-300">{beat.summary}</p>
            <div className="grid gap-3 md:grid-cols-2 text-xs text-slate-400">
              <div>
                <p className="font-semibold text-slate-300">剧情亮点</p>
                <ul className="list-disc pl-4">
                  {beat.highlights.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="font-semibold text-slate-300">伏笔与回收</p>
                <ul className="list-disc pl-4">
                  {beat.foreshadowing.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          </Card>
        ))}
      </section>
    </div>
  );
}
