function hsvToRgb(h: number, s: number, v: number): string {
  let r = 0, g = 0, b = 0;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    case 5: r = v; g = p; b = q; break;
  }
  return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
}

let cachedColors: Map<string, string> | null = null;

export function buildClusterColors(icons: { cluster: string }[]): Map<string, string> {
  if (cachedColors) return cachedColors;
  const themes = new Map<string, number>();
  for (const icon of icons) {
    if (icon.cluster) {
      themes.set(icon.cluster, (themes.get(icon.cluster) ?? 0) + 1);
    }
  }
  const sorted = [...themes.entries()].sort((a, b) => b[1] - a[1]);
  const n = sorted.length;
  cachedColors = new Map();
  sorted.forEach(([theme], i) => {
    cachedColors!.set(theme, hsvToRgb(i / n, 0.65, 0.85));
  });
  return cachedColors;
}

export function clusterColor(colors: Map<string, string>, cluster: string): string {
  return colors.get(cluster) || '#888';
}

export function clusterList(colors: Map<string, string>, icons: { cluster: string }[]): { theme: string; count: number; color: string }[] {
  const counts = new Map<string, number>();
  for (const icon of icons) {
    if (icon.cluster) {
      counts.set(icon.cluster, (counts.get(icon.cluster) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([theme, count]) => ({ theme, count, color: colors.get(theme) || '#888' }));
}
