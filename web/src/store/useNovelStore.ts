import { create } from "zustand";
import type {
  AIConfig,
  ChapterPlan,
  CharacterProfile,
  NovelProject,
  OutlineBeat,
  WorldbuildingProfile
} from "@/types/novel";

interface NovelState {
  projects: NovelProject[];
  worldbuilding: WorldbuildingProfile;
  characters: CharacterProfile[];
  outline: OutlineBeat[];
  chapters: ChapterPlan[];
  aiConfigs: AIConfig[];
  setActiveConfig: (id: string) => void;
}

export const useNovelStore = create<NovelState>((set) => ({
  projects: [
    {
      id: "project-1",
      title: "雾隐城纪事",
      genre: "奇幻",
      style: "史诗 + 悬疑",
      targetChapters: 48,
      status: "writing",
      updatedAt: "2024-03-18",
      progress: 62
    },
    {
      id: "project-2",
      title: "星轨边境",
      genre: "科幻",
      style: "硬科幻 + 成长",
      targetChapters: 36,
      status: "draft",
      updatedAt: "2024-03-12",
      progress: 18
    }
  ],
  worldbuilding: {
    era: "蒸汽与秘术并行的第二帝国时期",
    powerSystem: "符文矩阵 + 以太驱动装置",
    culture: "贵族议会与工坊联盟并存，强调契约与血统",
    geography: "巨型浮空群岛与深渊矿脉共存",
    history: "旧王朝覆灭后，新议会与军工财团分权",
    notes: "世界观说明支持结构化 JSON 与富文本描述。"
  },
  characters: [
    {
      id: "char-1",
      name: "林岚",
      age: "24",
      appearance: "银发短发，左眼带符文单片镜",
      personality: "理性冷静但内心执着",
      background: "工坊联盟的前侦察官",
      skills: ["符文破解", "机巧战术", "谈判"],
      relationships: "与反派家族有隐秘的血缘线索",
      arc: "从怀疑体制到重建秩序",
      voice: "言语简洁，偏向命令式",
      role: "protagonist"
    },
    {
      id: "char-2",
      name: "席恩",
      age: "31",
      appearance: "深色长发，披风缀有旧王朝徽章",
      personality: "强势、战略思维",
      background: "旧王朝遗民领袖",
      skills: ["政治博弈", "剑术", "情报网络"],
      relationships: "与主角保持互相利用的脆弱同盟",
      arc: "复辟信念与个人救赎的拉扯",
      voice: "用词优雅，带暗示性",
      role: "antagonist"
    }
  ],
  outline: [
    {
      id: "beat-1",
      act: "第一幕",
      title: "雾隐城的信号",
      summary: "主角收到失踪导师留下的密电，触发调查。",
      highlights: ["城市停摆", "导师留言"],
      foreshadowing: ["浮空群岛失衡", "议会阴影"]
    },
    {
      id: "beat-2",
      act: "第二幕",
      title: "深渊矿脉的交易",
      summary: "主角深入矿脉，与地下势力达成交易。",
      highlights: ["权力体系冲突", "盟友背叛"],
      foreshadowing: ["核心装置曝光"]
    }
  ],
  chapters: [
    {
      id: "chapter-1",
      title: "序章：雾隐的钟声",
      status: "final",
      wordCount: 3200,
      summary: "引出世界观与主角的现状。"
    },
    {
      id: "chapter-2",
      title: "第一章：碎裂的符文",
      status: "draft",
      wordCount: 1800,
      summary: "主角调查符文异常，获取初步线索。"
    },
    {
      id: "chapter-3",
      title: "第二章：暗影交易",
      status: "pending",
      wordCount: 0,
      summary: "为揭示地下势力铺垫。"
    }
  ],
  aiConfigs: [
    {
      id: "openai-1",
      provider: "openai",
      model: "gpt-4o-mini",
      temperature: 0.7,
      maxTokens: 4096,
      isActive: true
    },
    {
      id: "gemini-1",
      provider: "gemini",
      model: "gemini-1.5-pro",
      temperature: 0.6,
      maxTokens: 2048,
      isActive: false
    }
  ],
  setActiveConfig: (id) =>
    set((state) => ({
      aiConfigs: state.aiConfigs.map((config) => ({
        ...config,
        isActive: config.id === id
      }))
    }))
}));
