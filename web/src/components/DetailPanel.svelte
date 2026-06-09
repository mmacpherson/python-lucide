<script lang="ts">
  import type { IconData } from "../lib/types";

  let {
    icon,
    clusterColors,
    similarIcons = [],
    onclose,
    oncopy,
    onpick,
  }: {
    icon: IconData | null;
    clusterColors: Map<string, string>;
    similarIcons: IconData[];
    onclose: () => void;
    oncopy: (value: string, label: string) => void;
    onpick: (icon: IconData) => void;
  } = $props();

  let stroke = $state(1.75);
  let size = $state(72);

  $effect(() => {
    if (icon) {
      stroke = 1.75;
      size = 72;
    }
  });

  function clusterCol(cluster: string): string {
    return clusterColors.get(cluster) || "#888";
  }

  function renderSvg(svg: string, s: number, sw: number): string {
    return svg
      .replace(/width="[^"]*"/, `width="${s}"`)
      .replace(/height="[^"]*"/, `height="${s}"`)
      .replace(/stroke-width="[^"]*"/, `stroke-width="${sw}"`);
  }

  function getSvgForCopy(): string {
    if (!icon) return "";
    return renderSvg(icon.svg, size, stroke);
  }

  function toJsxName(name: string): string {
    return name.split("-").map((s) => s[0].toUpperCase() + s.slice(1)).join("");
  }

  function copyFormats(name: string) {
    return [
      { label: "Name", hint: "click to copy", value: name },
      { label: "python-lucide", hint: "FastHTML / Flask / Django", value: `lucide_icon("${name}")` },
      { label: "JSX", hint: "lucide-react", value: `<${toJsxName(name)} />` },
      { label: "HTML", hint: "data attribute", value: `<i data-lucide="${name}"></i>` },
    ];
  }
</script>

<div class="scrim" class:open={!!icon} onclick={onclose} role="presentation"></div>
<aside class="panel" class:open={!!icon}>
  {#if icon}
    <div class="panel-in">
      <div class="panel-head">
        <span class="mono panel-name">{icon.name}</span>
        <button class="iconbtn" onclick={onclose} title="Close">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--tx2)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
        </button>
      </div>

      <div
        class="preview"
        onclick={() => oncopy(getSvgForCopy(), "Copied SVG")}
        title="Click to copy SVG"
        role="button"
        tabindex="0"
        onkeydown={(e) => e.key === "Enter" && oncopy(getSvgForCopy(), "Copied SVG")}
      >
        {@html renderSvg(icon.svg, size, stroke)}
      </div>

      <div class="ctrls">
        <label class="ctrl">
          <span>Stroke<b class="mono">{stroke.toFixed(2)}</b></span>
          <input type="range" min="0.5" max="3" step="0.25" bind:value={stroke} />
        </label>
        <label class="ctrl">
          <span>Size<b class="mono">{size}px</b></span>
          <input type="range" min="16" max="96" step="2" bind:value={size} />
        </label>
      </div>

      <div class="meta-block">
        <div class="meta-row">
          <span class="dot" style="background: {clusterCol(icon.cluster)}"></span>
          <span class="meta-cat">{icon.cluster}</span>
        </div>
        <p class="meta-desc">{icon.description}</p>
      </div>

      <button class="svgbtn" onclick={() => oncopy(getSvgForCopy(), "Copied SVG")}>
        <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
        Copy SVG
      </button>

      <div class="formats">
        {#each copyFormats(icon.name) as fmt}
          <button class="fmt" onclick={() => oncopy(fmt.value, `Copied ${fmt.label}`)}>
            <span class="fmt-l">
              <span class="fmt-label">{fmt.label}</span>
              <span class="fmt-hint">{fmt.hint}</span>
            </span>
            <span class="fmt-v mono">{fmt.value}</span>
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--tx2)" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
          </button>
        {/each}
      </div>

      {#if similarIcons.length > 0}
        <div class="similar">
          <div class="similar-h">More in {icon.cluster}</div>
          <div class="similar-row">
            {#each similarIcons as sim}
              <button class="sim" title={sim.name} onclick={() => onpick(sim)}>
                <span class="sim-icon">{@html sim.svg}</span>
              </button>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</aside>

<style>
  .scrim {
    position: fixed; inset: 0;
    background: rgba(0,0,0,.4);
    opacity: 0; pointer-events: none;
    transition: opacity .25s; z-index: 55;
  }
  .scrim.open { opacity: 1; pointer-events: auto; }

  .panel {
    position: fixed; top: 0; right: 0; height: 100vh;
    width: 404px; max-width: 92vw;
    background: var(--bg2);
    border-left: 1px solid var(--bd);
    transform: translateX(101%);
    transition: transform .3s cubic-bezier(.4,0,.2,1);
    z-index: 60; box-shadow: var(--shadow);
    overflow: auto;
  }
  .panel.open { transform: none; }
  .panel-in { display: flex; flex-direction: column; }

  .panel-head {
    position: sticky; top: 0; background: var(--bg2);
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 18px; border-bottom: 1px solid var(--bd); z-index: 1;
  }
  .panel-name { font-size: 15px; color: var(--tx); }
  .mono { font-family: var(--font-mono); }

  .iconbtn {
    width: 32px; height: 32px; border-radius: 8px;
    border: 1px solid transparent; background: transparent;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background .15s;
  }
  .iconbtn:hover { background: var(--surf2); }

  .preview {
    margin: 18px 18px 0; height: 170px;
    border-radius: 14px; border: 1px solid var(--bd);
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    color: var(--tx);
    background:
      var(--bg),
      conic-gradient(from 90deg at 1px 1px, transparent 90deg, var(--hl) 0) 0 0 / 16px 16px;
  }
  .preview:hover { border-color: var(--bd2); }

  .ctrls { padding: 16px 18px 4px; display: flex; flex-direction: column; gap: 14px; }
  .ctrl span {
    display: flex; justify-content: space-between;
    font-size: 12px; color: var(--tx2); margin-bottom: 8px;
  }
  .ctrl b { color: var(--tx); font-weight: 500; }

  .meta-block { padding: 16px 18px; }
  .meta-row { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
  .dot {
    width: 6px; height: 6px; border-radius: 6px; flex: 0 0 auto;
  }
  .meta-cat { font-size: 12.5px; color: var(--tx2); white-space: nowrap; }
  .meta-desc { margin: 0; font-size: 13px; line-height: 1.5; color: var(--tx2); }

  .svgbtn {
    margin: 0 18px;
    display: flex; align-items: center; justify-content: center; gap: 8px;
    padding: 11px; border-radius: 10px;
    border: 1px solid var(--bd2); background: var(--surf);
    color: var(--tx); font-size: 13px; font-weight: 600;
    cursor: pointer; transition: .14s;
  }
  .svgbtn:hover { background: var(--surf2); border-color: var(--ac); }

  .formats { padding: 12px 18px 0; display: flex; flex-direction: column; gap: 8px; }
  .fmt {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 12px; border-radius: 10px;
    border: 1px solid var(--bd); background: var(--surf);
    cursor: pointer; text-align: left; transition: .14s;
  }
  .fmt:hover { border-color: var(--ac); background: var(--surf2); }
  .fmt-l { display: flex; flex-direction: column; gap: 2px; flex: 0 0 auto; }
  .fmt-label { font-size: 11px; font-weight: 700; color: var(--tx); }
  .fmt-hint { font-size: 10px; color: var(--tx3); }
  .fmt-v {
    flex: 1; min-width: 0; font-size: 11.5px; color: var(--tx2);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: right;
  }

  .similar { padding: 18px; }
  .similar-h { font-size: 11px; font-weight: 700; color: var(--tx2); margin-bottom: 10px; }
  .similar-row { display: flex; gap: 7px; flex-wrap: wrap; }
  .sim {
    width: 42px; height: 42px; border-radius: 9px;
    border: 1px solid var(--bd); background: var(--surf);
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: .12s; color: var(--ic);
  }
  .sim:hover { border-color: var(--ac); background: var(--surf2); transform: translateY(-1px); }
  .sim-icon { display: flex; align-items: center; justify-content: center; }
  .sim-icon :global(svg) { width: 22px; height: 22px; }
</style>
