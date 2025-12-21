import Link from "next/link";

const links = [
  { href: "/", label: "项目概览" },
  { href: "/worldbuilding", label: "世界观" },
  { href: "/characters", label: "角色" },
  { href: "/outline", label: "大纲" },
  { href: "/chapters", label: "章节" },
  { href: "/settings", label: "AI配置" }
];

export function Sidebar() {
  return (
    <aside className="flex h-full w-64 flex-col gap-4 border-r border-muted bg-muted/40 p-6">
      <div>
        <p className="text-xs uppercase text-slate-400">AI自动写小说系统</p>
        <h1 className="text-lg font-semibold">创作工作台</h1>
      </div>
      <nav className="flex flex-col gap-2">
        {links.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className="rounded-lg px-3 py-2 text-sm text-slate-200 transition hover:bg-muted"
          >
            {link.label}
          </Link>
        ))}
      </nav>
      <div className="mt-auto rounded-xl bg-slate-900/60 p-4 text-xs text-slate-300">
        <p>版本: 0.1.0</p>
        <p>支持 OpenAI / Gemini</p>
      </div>
    </aside>
  );
}
