export type ProjectStatus = "draft" | "writing" | "paused" | "completed";

export interface NovelProject {
  id: string;
  title: string;
  genre: string;
  style: string;
  targetChapters: number;
  status: ProjectStatus;
  updatedAt: string;
  progress: number;
}

export interface WorldbuildingProfile {
  era: string;
  powerSystem: string;
  culture: string;
  geography: string;
  history: string;
  notes: string;
}

export type CharacterRole = "protagonist" | "supporting" | "antagonist" | "minor";

export interface CharacterProfile {
  id: string;
  name: string;
  age: string;
  appearance: string;
  personality: string;
  background: string;
  skills: string[];
  relationships: string;
  arc: string;
  voice: string;
  role: CharacterRole;
}

export interface OutlineBeat {
  id: string;
  act: string;
  title: string;
  summary: string;
  highlights: string[];
  foreshadowing: string[];
}

export interface ChapterPlan {
  id: string;
  title: string;
  status: "pending" | "draft" | "review" | "final";
  wordCount: number;
  summary: string;
}

export interface AIConfig {
  id: string;
  provider: "openai" | "gemini";
  model: string;
  temperature: number;
  maxTokens: number;
  isActive: boolean;
}
