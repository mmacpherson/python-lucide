<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import type { IconsManifest, IconData } from "../lib/types";
  import { loadEmbeddings, search } from "../lib/embeddings";
  import { clusterColor, clusterList } from "../lib/clusters";

  const BASE = import.meta.env.BASE_URL;

  let {
    manifest,
    clusterColors,
  }: {
    manifest: IconsManifest;
    clusterColors: Map<string, string>;
  } = $props();

  let canvas: HTMLCanvasElement;

  let coords: Record<string, [number, number]> = {};
  let loaded = $state(false);
  let loadError = $state("");

  let clusters = $derived(clusterList(clusterColors, manifest.icons));

  let selectedIcons: Set<string> = $state(new Set());
  let neighborIcons: Set<string> = $state(new Set());
  let clusterHighlight: Set<string> = $state(new Set());
  let hoveredIcon: IconData | null = $state(null);
  let hoveredPos = $state({ x: 0, y: 0 });

  let embeddingsMatrix: Float32Array | null = null;
  let embeddingDim = 0;

  let transform = $state({ x: 0, y: 0, scale: 1 });
  let dragging = false;

  let points: { icon: IconData; x: number; y: number; idx: number }[] = [];
  const ICON_SIZE = 18;

  const iconImageCache = new Map<string, HTMLImageElement>();

  function svgToImage(svgStr: string, color: string, size: number): Promise<HTMLImageElement> {
    const key = `${svgStr}:${color}`;
    const cached = iconImageCache.get(key);
    if (cached) return Promise.resolve(cached);

    const colored = svgStr
      .replace(/width="[^"]*"/, `width="${size}"`)
      .replace(/height="[^"]*"/, `height="${size}"`)
      .replace(/currentColor/g, color);
    const blob = new Blob([colored], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        iconImageCache.set(key, img);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.src = url;
    });
  }

  let iconImages: Map<string, HTMLImageElement> = new Map();

  // Glyphs stream in behind the colored-dot fallback in draw(); each
  // arrival coalesces into at most one redraw per frame.
  function prerenderIcons() {
    for (const icon of manifest.icons) {
      const color = clusterColor(clusterColors, icon.cluster);
      void svgToImage(icon.svg, color, ICON_SIZE * 2).then((img) => {
        iconImages.set(icon.name, img);
        scheduleDraw();
      });
    }
  }

  // Width/height are CSS pixels: the draw context is dpr-scaled, so laying
  // out in device pixels would shift everything by dpr× and clip the plot
  // on high-density (phone) screens.
  function layoutPoints(width: number, height: number) {
    if (!Object.keys(coords).length) return;

    const xs = Object.values(coords).map((c) => c[0]);
    const ys = Object.values(coords).map((c) => c[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    const pad = 40;
    const w = width - pad * 2;
    const h = height - pad * 2;

    points = manifest.icons
      .map((icon, idx) => {
        const c = coords[icon.name];
        if (!c) return null;
        return {
          icon,
          x: pad + ((c[0] - minX) / rangeX) * w,
          y: pad + ((c[1] - minY) / rangeY) * h,
          idx,
        };
      })
      .filter((p): p is NonNullable<typeof p> => p !== null);
  }

  let drawScheduled = false;
  function scheduleDraw() {
    if (drawScheduled) return;
    drawScheduled = true;
    requestAnimationFrame(() => {
      drawScheduled = false;
      draw();
    });
  }

  function draw() {
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    layoutPoints(rect.width, rect.height);

    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);

    const hasSelection = selectedIcons.size > 0;
    const hasCluster = clusterHighlight.size > 0;
    const dimmed = hasSelection || hasCluster;

    if (hasSelection) {
      ctx.strokeStyle = "rgba(109, 108, 245, 0.15)";
      ctx.lineWidth = 0.5;
      for (const p of points) {
        if (!selectedIcons.has(p.icon.name)) continue;
        for (const np of points) {
          if (!neighborIcons.has(np.icon.name)) continue;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(np.x, np.y);
          ctx.stroke();
        }
      }
    }

    const iconDrawSize = ICON_SIZE / transform.scale;
    const half = iconDrawSize / 2;

    for (const p of points) {
      const isSelected = selectedIcons.has(p.icon.name);
      const isNeighbor = neighborIcons.has(p.icon.name);
      const isClusterHit = clusterHighlight.has(p.icon.name);

      let alpha = dimmed ? 0.1 : 0.7;
      let scale = 1;

      if (isSelected) { alpha = 1; scale = 1.5; }
      else if (isNeighbor) { alpha = 0.85; scale = 1.2; }
      else if (isClusterHit) { alpha = 0.95; scale = 1.3; }

      ctx.globalAlpha = alpha;

      const img = iconImages.get(p.icon.name);
      if (img && img.complete && img.naturalWidth > 0) {
        const s = half * scale;
        ctx.drawImage(img, p.x - s, p.y - s, s * 2, s * 2);
      } else {
        ctx.fillStyle = clusterColor(clusterColors, p.icon.cluster);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3 / transform.scale, 0, Math.PI * 2);
        ctx.fill();
      }

      if (isSelected) {
        ctx.strokeStyle = "rgb(109, 108, 245)";
        ctx.lineWidth = 2 / transform.scale;
        ctx.beginPath();
        ctx.arc(p.x, p.y, half * scale + 2 / transform.scale, 0, Math.PI * 2);
        ctx.stroke();
      } else if (isClusterHit) {
        ctx.strokeStyle = "rgb(255, 200, 60)";
        ctx.lineWidth = 1.5 / transform.scale;
        ctx.beginPath();
        ctx.arc(p.x, p.y, half * scale + 2 / transform.scale, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    ctx.restore();
    ctx.globalAlpha = 1;
  }

  function screenToCanvas(ex: number, ey: number): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (ex - rect.left - transform.x) / transform.scale,
      y: (ey - rect.top - transform.y) / transform.scale,
    };
  }

  function findIconAt(ex: number, ey: number): typeof points[0] | null {
    const { x, y } = screenToCanvas(ex, ey);
    const hitRadius = (ICON_SIZE / 2 + 2) / transform.scale;
    let best: typeof points[0] | null = null;
    let bestDist = hitRadius * hitRadius;
    for (const p of points) {
      const dx = p.x - x;
      const dy = p.y - y;
      const d = dx * dx + dy * dy;
      if (d < bestDist) { bestDist = d; best = p; }
    }
    return best;
  }

  function computeNeighbors() {
    if (!embeddingsMatrix || selectedIcons.size === 0) {
      neighborIcons = new Set();
      return;
    }
    const allNeighbors = new Set<string>();
    for (const name of selectedIcons) {
      const idx = manifest.icons.findIndex((i) => i.name === name);
      if (idx < 0) continue;
      const vec = embeddingsMatrix.slice(idx * embeddingDim, (idx + 1) * embeddingDim);
      const hits = search(vec, embeddingsMatrix, embeddingDim, 16);
      for (const h of hits) {
        const n = manifest.icons[h.index].name;
        if (!selectedIcons.has(n)) allNeighbors.add(n);
      }
    }
    neighborIcons = allNeighbors;
  }

  function handleClick(e: MouseEvent) {
    const hit = findIconAt(e.clientX, e.clientY);
    if (!hit) {
      selectedIcons = new Set();
      neighborIcons = new Set();
      draw();
      return;
    }
    const name = hit.icon.name;
    const next = new Set(selectedIcons);
    if (e.shiftKey || e.metaKey) {
      if (next.has(name)) next.delete(name); else next.add(name);
    } else {
      if (next.size === 1 && next.has(name)) next.clear();
      else { next.clear(); next.add(name); }
    }
    selectedIcons = next;
    computeNeighbors();
    draw();
  }

  function handleMouseMove(e: MouseEvent) {
    if (dragging) {
      transform = { ...transform, x: transform.x + e.movementX, y: transform.y + e.movementY };
      scheduleDraw();
      return;
    }
    const hit = findIconAt(e.clientX, e.clientY);
    if (hit) {
      hoveredIcon = hit.icon;
      hoveredPos = { x: e.clientX, y: e.clientY };
      canvas.style.cursor = "pointer";
    } else {
      hoveredIcon = null;
      canvas.style.cursor = "grab";
    }
  }

  function handleMouseDown(e: MouseEvent) {
    if (e.button !== 0) return;
    const hit = findIconAt(e.clientX, e.clientY);
    if (!hit) { dragging = true; canvas.style.cursor = "grabbing"; }
  }

  function handleMouseUp() {
    dragging = false;
    canvas.style.cursor = hoveredIcon ? "pointer" : "grab";
  }

  function handleWheel(e: WheelEvent) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.3, Math.min(10, transform.scale * factor));
    const ratio = newScale / transform.scale;
    transform = { x: mx - (mx - transform.x) * ratio, y: my - (my - transform.y) * ratio, scale: newScale };
    scheduleDraw();
  }

  // Radio-style cluster selection: one cluster at a time, clicking the
  // active one (or the chip above the list) clears it
  let activeCluster: string | null = $state(null);

  function toggleCluster(theme: string) {
    if (activeCluster === theme) { clearCluster(); return; }
    activeCluster = theme;
    clusterHighlight = new Set(
      manifest.icons.filter((i) => i.cluster === theme).map((i) => i.name),
    );
    draw();
  }

  function clearCluster() {
    activeCluster = null;
    clusterHighlight = new Set();
    draw();
  }

  async function loadData() {
    try {
      const resp = await fetch(`${BASE}data/umap-coords.json?v=${manifest.version}`);
      if (!resp.ok) throw new Error(`Failed to fetch umap-coords.json: HTTP ${resp.status}`);
      coords = await resp.json();

      const model = manifest.models.find((m) => m.clusterSource) ?? manifest.models[0];
      embeddingDim = model.dim;
      embeddingsMatrix = await loadEmbeddings(
        `${BASE}data/${model.file}?v=${manifest.version}`, manifest.icons.length, model.dim,
      );

      loaded = true;
      prerenderIcons();
    } catch (e) {
      loadError = `Failed to load explore data: ${e}`;
    }
  }

  async function retryLoad() {
    loadError = "";
    await loadData();
    draw();
  }

  let resizeObserver: ResizeObserver | null = null;

  onMount(async () => {
    await loadData();
    draw();
    resizeObserver = new ResizeObserver(() => draw());
    resizeObserver.observe(canvas);
  });

  onDestroy(() => {
    resizeObserver?.disconnect();
  });
</script>

<div class="explore">
  <div class="ex-list">
    {#if activeCluster}
      {@const ac = clusters.find((c) => c.theme === activeCluster)}
      <button class="active-chip" onclick={clearCluster} aria-label="Clear cluster selection">
        <span class="dot" style="background: {ac?.color}"></span>
        <span class="chip-name">{activeCluster}</span>
        <span class="ex-count">{clusterHighlight.size}</span>
        <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
      </button>
    {/if}

    <div class="ex-h mono">{clusters.length} clusters &middot; click to highlight</div>

    {#if selectedIcons.size > 0}
      <div class="sel-panel">
        <div class="sel-h">Selected ({selectedIcons.size})</div>
        <div class="sel-list">
          {#each [...selectedIcons] as name}
            {@const icon = manifest.icons.find((i) => i.name === name)}
            {#if icon}
              <div class="sel-item">
                <span class="sel-icon">{@html icon.svg}</span>
                <span class="sel-name">{icon.name}</span>
              </div>
            {/if}
          {/each}
        </div>
        {#if neighborIcons.size > 0}
          <div class="sel-info">{neighborIcons.size} neighbors shown</div>
        {/if}
        <button class="sel-clear" onclick={() => { selectedIcons = new Set(); neighborIcons = new Set(); draw(); }}>
          Clear
        </button>
      </div>
    {/if}

    <div class="cluster-scroll">
      {#each clusters as cl}
        <button
          class="ex-item"
          class:active={activeCluster === cl.theme}
          onclick={() => toggleCluster(cl.theme)}
        >
          <span class="dot" style="background: {cl.color}"></span>
          <span class="ex-name">{cl.theme}</span>
          <span class="ex-n mono">{cl.count}</span>
        </button>
      {/each}
    </div>
  </div>

  <div class="ex-canvas">
    <canvas
      bind:this={canvas}
      onclick={handleClick}
      onmousemove={handleMouseMove}
      onmousedown={handleMouseDown}
      onmouseup={handleMouseUp}
      onmouseleave={handleMouseUp}
      onwheel={handleWheel}
    ></canvas>
    {#if loadError}
      <div class="canvas-loading">
        <div class="canvas-error">
          <span>{loadError}</span>
          <button class="retry" onclick={retryLoad}>Retry</button>
        </div>
      </div>
    {:else if !loaded}
      <div class="canvas-loading">Loading embedding space...</div>
    {/if}
    <div class="ex-hint">Click to select &middot; Shift+click multi-select &middot; Scroll to zoom &middot; Drag to pan</div>
  </div>
</div>

{#if hoveredIcon}
  <div class="tooltip" style="left: {hoveredPos.x + 16}px; top: {hoveredPos.y - 8}px">
    <div class="tooltip-hdr">
      <span class="tooltip-svg">{@html hoveredIcon.svg}</span>
      <strong>{hoveredIcon.name}</strong>
    </div>
    <p>{hoveredIcon.description}</p>
    <span class="tooltip-cluster" style="color: {clusterColor(clusterColors, hoveredIcon.cluster)}">
      {hoveredIcon.cluster}
    </span>
  </div>
{/if}

<style>
  .explore {
    flex: 1; display: flex; position: relative; z-index: 1; min-height: 0;
    /* Cap at the viewport so the absolutely positioned canvas and the
       scrollable cluster list stay bounded. max-height, not height: flex:1
       sets flex-basis:0%, which makes height ignored on the main axis.
       dvh tracks mobile browser chrome where supported. */
    max-height: calc(100vh - var(--hdr-h));
    max-height: calc(100dvh - var(--hdr-h));
  }

  /* ── Sidebar ─────────────────────────────────────────────────── */
  .ex-list {
    width: 260px; flex: 0 0 auto;
    border-right: 1px solid var(--bd);
    padding: 18px 14px; overflow: auto;
    display: flex; flex-direction: column;
  }
  .active-chip {
    display: flex; align-items: center; gap: 9px;
    background: var(--surf); border: 1px solid var(--bd);
    border-radius: 9px; padding: 9px 11px; margin-bottom: 16px;
    cursor: pointer; color: var(--tx3); font-family: var(--font);
    width: 100%; text-align: left;
  }
  .active-chip:hover { border-color: var(--bd2); color: var(--tx); }
  .chip-name {
    flex: 1; font-size: 12.5px; color: var(--tx);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .ex-count {
    font-family: var(--font-mono); font-size: 11px;
    color: var(--ac); background: var(--acs);
    padding: 1px 6px; border-radius: 4px;
  }
  .ex-h {
    font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase;
    color: var(--tx3); margin: 0 0 10px 4px;
  }
  .mono { font-family: var(--font-mono); }

  .cluster-scroll {
    display: flex; flex-direction: column; gap: 1px;
    overflow-y: auto; flex: 1; min-height: 0;
  }
  .ex-item {
    display: flex; align-items: center; gap: 9px;
    width: 100%; padding: 7px 8px; border-radius: 7px;
    border: none; background: transparent;
    cursor: pointer; color: var(--tx2); transition: background .12s;
    font-family: var(--font); text-align: left;
  }
  .ex-item:hover { background: var(--surf2); color: var(--tx); }
  .ex-item.active { background: var(--acs); color: var(--tx); }
  .dot { width: 8px; height: 8px; border-radius: 8px; flex: 0 0 auto; }
  .ex-name {
    flex: 1; font-size: 12.5px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .ex-n { font-size: 11px; color: var(--tx3); }

  /* ── Selection panel ─────────────────────────────────────────── */
  .sel-panel {
    background: var(--surf); border: 1px solid var(--bd);
    border-radius: 9px; padding: 10px; margin-bottom: 12px;
  }
  .sel-h { font-size: 11px; font-weight: 600; color: var(--tx2); margin-bottom: 8px; }
  .sel-list { display: flex; flex-direction: column; gap: 4px; }
  .sel-item { display: flex; align-items: center; gap: 8px; }
  .sel-icon { width: 20px; height: 20px; color: var(--ic); display: flex; }
  .sel-icon :global(svg) { width: 18px; height: 18px; }
  .sel-name { font-family: var(--font-mono); font-size: 11px; color: var(--tx); }
  .sel-info { font-size: 10px; color: var(--tx3); margin-top: 6px; }
  .sel-clear {
    margin-top: 8px; padding: 4px 10px;
    background: transparent; border: 1px solid var(--bd);
    border-radius: 6px; color: var(--tx2);
    font-size: 11px; cursor: pointer; font-family: var(--font);
  }
  .sel-clear:hover { border-color: var(--bd2); color: var(--tx); }

  /* ── Canvas ──────────────────────────────────────────────────── */
  .ex-canvas {
    flex: 1; position: relative; overflow: hidden;
    background: radial-gradient(circle at 50% 42%, var(--hl), transparent 60%);
  }
  canvas {
    /* Absolute, not height:100%: in the stacked (column) phone layout the
       flexed parent height isn't "definite", so a percentage height falls
       back to the buffer's intrinsic aspect and squashes the plot */
    position: absolute; inset: 0; width: 100%; height: 100%; display: block;
    cursor: grab;
  }
  .canvas-loading {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    color: var(--tx2); pointer-events: none;
  }
  .canvas-error {
    pointer-events: auto;
    display: flex; flex-direction: column; align-items: center; gap: 10px;
    color: #ff6b6b; font-size: 13px; text-align: center; max-width: 360px;
  }
  .retry {
    border: 1px solid #ff6b6b; background: transparent; color: #ff6b6b;
    border-radius: 6px; padding: 5px 14px; cursor: pointer;
    font-family: var(--font); font-size: 12.5px;
  }
  .retry:hover { background: #ff6b6b; color: #fff; }
  .ex-hint {
    position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%);
    font-size: 11.5px; color: var(--tx3); white-space: nowrap; pointer-events: none;
  }

  /* ── Tooltip ─────────────────────────────────────────────────── */
  .tooltip {
    position: fixed; z-index: 100;
    background: var(--bg2); border: 1px solid var(--bd);
    border-radius: 10px; padding: 10px; max-width: 300px;
    pointer-events: none; box-shadow: var(--shadow);
  }
  .tooltip-hdr { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .tooltip-svg { width: 22px; height: 22px; color: var(--ic); display: flex; }
  .tooltip-svg :global(svg) { width: 22px; height: 22px; }
  .tooltip strong { font-family: var(--font-mono); font-size: 13px; }
  .tooltip p { font-size: 12px; color: var(--tx2); line-height: 1.4; margin: 0; }
  .tooltip-cluster { display: inline-block; margin-top: 4px; font-size: 11px; }

  /* Portrait phones: plot on top, cluster list scrolls below it.
     column-reverse keeps the DOM order (list first) for keyboard users.
     Must come after the base rules above to override them. */
  @media (max-width: 700px) {
    .explore { flex-direction: column-reverse; }
    .ex-list {
      width: 100%; max-height: 34vh;
      border-right: none; border-top: 1px solid var(--bd);
    }
    .ex-canvas { min-height: 52vh; }
    .ex-hint { display: none; }
  }
</style>
