export type Indicator = "great" | "ok" | "bad";

export function indicatorFromNumber(value: number): Indicator {
  if (value === 1) return "great";
  if (value === 0) return "ok";
  return "bad";
}

export const pelletConfig: Record<
  Indicator,
  { color: string; translationKey: string }
> = {
  great: {
    color: "var(--background-color-indicator-great)",
    translationKey: "indicator_message_great",
  },
  ok: {
    color: "var(--background-color-indicator-ok)",
    translationKey: "indicator_message_ok",
  },
  bad: {
    color: "var(--background-color-indicator-bad)",
    translationKey: "indicator_message_bad",
  },
};
