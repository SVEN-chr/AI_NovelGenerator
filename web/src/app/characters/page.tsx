"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useNovelStore } from "@/store/useNovelStore";

const roleMap = {
  protagonist: "主角",
  supporting: "配角",
  antagonist: "反派",
  minor: "次要角色"
};

export default function CharactersPage() {
  const characters = useNovelStore((state) => state.characters);

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">角色管理</h2>
          <p className="text-sm text-slate-400">
            AI 生成角色档案，维护角色弧光与关系图谱
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary">导入角色关系</Button>
          <Button>AI 生成角色</Button>
        </div>
      </header>

      <section className="grid gap-4 lg:grid-cols-2">
        {characters.map((character) => (
          <Card key={character.id} className="space-y-4">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-lg font-semibold text-white">{character.name}</p>
                <p className="text-xs text-slate-400">年龄 {character.age}</p>
              </div>
              <Badge>{roleMap[character.role]}</Badge>
            </div>
            <div className="space-y-2 text-sm text-slate-300">
              <p>外貌：{character.appearance}</p>
              <p>性格：{character.personality}</p>
              <p>背景：{character.background}</p>
              <p>能力：{character.skills.join("、")}</p>
              <p>人际关系：{character.relationships}</p>
              <p>角色弧光：{character.arc}</p>
              <p>说话风格：{character.voice}</p>
            </div>
          </Card>
        ))}
      </section>
    </div>
  );
}
