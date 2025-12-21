"use client";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useNovelStore } from "@/store/useNovelStore";

export default function SettingsPage() {
  const { aiConfigs, setActiveConfig } = useNovelStore((state) => ({
    aiConfigs: state.aiConfigs,
    setActiveConfig: state.setActiveConfig
  }));

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">AI 配置管理</h2>
          <p className="text-sm text-slate-400">
            支持 OpenAI 与 Gemini API，支持多配置切换
          </p>
        </div>
        <Button>新增配置</Button>
      </header>

      <section className="grid gap-4 md:grid-cols-2">
        {aiConfigs.map((config) => (
          <Card key={config.id} className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-slate-400">{config.provider.toUpperCase()}</p>
                <p className="text-lg font-semibold text-white">{config.model}</p>
              </div>
              <Button
                variant={config.isActive ? "secondary" : "ghost"}
                onClick={() => setActiveConfig(config.id)}
              >
                {config.isActive ? "当前使用" : "切换"}
              </Button>
            </div>
            <div className="grid gap-2 text-sm text-slate-300">
              <p>Temperature: {config.temperature}</p>
              <p>Max Tokens: {config.maxTokens}</p>
            </div>
          </Card>
        ))}
      </section>
    </div>
  );
}
