"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useNovelStore } from "@/store/useNovelStore";

const statusMap = {
  pending: "待生成",
  draft: "草稿",
  review: "审校中",
  final: "已定稿"
};

export default function ChaptersPage() {
  const chapters = useNovelStore((state) => state.chapters);

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">章节管理</h2>
          <p className="text-sm text-slate-400">
            章节生成、进度展示、字数统计与批量生成
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary">批量生成</Button>
          <Button>生成新章节</Button>
        </div>
      </header>

      <section className="grid gap-4">
        {chapters.map((chapter) => (
          <Card key={chapter.id} className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-lg font-semibold text-white">{chapter.title}</p>
                <p className="text-xs text-slate-400">{chapter.wordCount} 字</p>
              </div>
              <Badge>{statusMap[chapter.status]}</Badge>
            </div>
            <p className="text-sm text-slate-300">{chapter.summary}</p>
            <div className="flex gap-2">
              <Button variant="secondary">编辑</Button>
              <Button variant="ghost">预览</Button>
            </div>
          </Card>
        ))}
      </section>
    </div>
  );
}
