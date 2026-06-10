<script lang="ts">
  import type { IconsManifest, IconData, ModelConfig, SearchResult } from "../lib/types";
  import { loadEmbeddings, search } from "../lib/embeddings";
  import { loadModel, embed } from "../lib/model-loader";
  import { clusterColor } from "../lib/clusters";
  import DetailPanel from "../components/DetailPanel.svelte";

  const BASE = import.meta.env.BASE_URL;

  let {
    manifest,
    clusterColors,
    showToast,
  }: {
    manifest: IconsManifest;
    clusterColors: Map<string, string>;
    showToast: (msg: string) => void;
  } = $props();

  let results: SearchResult[] = $state([]);
  let query = $state("");
  let activeModelId = $state("");
  let modelProgress = $state(-1);
  let embeddingsMatrix: Float32Array | null = $state(null);
  let ready = $state(false);
  let loadError = $state("");
  let dense = $state(true);
  let selectedIcon: IconData | null = $state(null);
  let inputRef: HTMLInputElement | null = $state(null);

  let isLoading = $derived(modelProgress >= 0);

  let similarIcons = $derived.by(() => {
    if (!selectedIcon) return [];
    const cluster = selectedIcon.cluster;
    return manifest.icons
      .filter((i) => i.name !== selectedIcon!.name && i.cluster === cluster)
      .slice(0, 8);
  });

  // Monotonic counter so a superseded load or search can't apply stale state
  let generation = 0;

  async function switchModel(config: ModelConfig) {
    const gen = ++generation;
    ready = false;
    modelProgress = 0;
    activeModelId = config.id;

    try {
      const [emb] = await Promise.all([
        loadEmbeddings(`${BASE}data/${config.file}?v=${manifest.version}`, manifest.icons.length, config.dim),
        loadModel(config, (p) => { if (gen === generation) modelProgress = p; }),
      ]);
      if (gen !== generation) return;
      embeddingsMatrix = emb;
      modelProgress = -1;
      ready = true;
      if (query) await doSearch(query);
    } catch (e) {
      if (gen !== generation) return;
      loadError = `Failed to load model: ${e}`;
      modelProgress = -1;
    }
  }

  function retryLoad() {
    loadError = "";
    const config = manifest.models.find((m) => m.id === activeModelId) ?? manifest.models[0];
    switchModel(config);
  }

  async function doSearch(q: string) {
    if (!embeddingsMatrix || !ready) return;
    const config = manifest.models.find((m) => m.id === activeModelId);
    if (!config) return;

    const gen = ++generation;
    try {
      const queryVec = await embed(q, config);
      if (gen !== generation) return;
      const hits = search(queryVec, embeddingsMatrix, config.dim, 50);
      results = hits.map((h) => ({
        icon: manifest.icons[h.index],
        score: h.score,
        index: h.index,
      }));
    } catch (e) {
      if (gen === generation) console.error("Search failed:", e);
    }
  }

  function handleInput() {
    if (searchTimeout) clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      if (query.trim()) {
        doSearch(query);
      } else {
        results = [];
      }
    }, 150);
  }

  let searchTimeout: ReturnType<typeof setTimeout> | null = null;

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Escape") {
      query = "";
      results = [];
    }
  }

  function clearSearch() {
    query = "";
    results = [];
    inputRef?.focus();
  }

  function copyText(value: string, label: string) {
    const done = () => showToast(label);
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(value).then(done, done);
    } else {
      done();
    }
  }

  function activeModelLabel(): string {
    const config = manifest.models.find((m) => m.id === activeModelId);
    return config?.hfModel.split("/").pop() ?? "";
  }

  // Curated with scripts/eval-prompts.mjs: each retrieves clearly relevant
  // icons in the top results on BOTH models, and shows that descriptive
  // phrases beat single keywords
  const EXAMPLES = [
    "waiting for something to finish loading",
    "the weather is getting colder",
    "notifications are turned off",
    "celebrate a big achievement",
    "secure login with a password",
  ];

  // Per-model overrides, curated the same way. The multilingual set mixes
  // languages to demonstrate the model's coverage — these would retrieve
  // junk on the English-only models, so they only appear here.
  const EXAMPLES_BY_MODEL: Record<string, string[]> = {
    multilingual: [
      "viajar a algún lugar en avión",
      "das Wetter wird kälter",
      "パスワードで安全にログイン",
      "庆祝一个重大成就",
      "notifications are turned off",
    ],
  };

  const examples = $derived(EXAMPLES_BY_MODEL[activeModelId] ?? EXAMPLES);

  function runExample(q: string) {
    query = q;
    doSearch(q);
  }

  let modelInfoOpen = $state(false);

  function handleWindowClick() {
    modelInfoOpen = false;
  }

  switchModel(manifest.models.find((m) => m.default) ?? manifest.models[0]);
</script>

<svelte:window onclick={handleWindowClick} />

<main class="main">
  {#if loadError}
    <div class="error">
      <span>{loadError}</span>
      <button class="retry" onclick={retryLoad}>Retry</button>
    </div>
  {:else}
    <div class="searchbar">
      <svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="var(--ac)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
      <input
        bind:this={inputRef}
        bind:value={query}
        oninput={handleInput}
        onkeydown={handleKeydown}
        placeholder="Describe an icon by meaning..."
        disabled={!ready}
      />
      {#if query}
        <button class="clear" onclick={clearSearch} aria-label="Clear search">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--tx3)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
        </button>
      {/if}
    </div>

    {#if !query && ready}
      <div class="examples">
        <span class="ex-label">Try</span>
        {#each examples as example}
          <button class="chip" onclick={() => runExample(example)}>{example}</button>
        {/each}
      </div>
    {/if}

    <div class="meta">
      <div class="meta-l">
        {#if results.length > 0}
          <span><b>{results.length}</b> results <span class="dim">&middot; ranked by meaning</span></span>
        {:else if !query && ready}
          <span><b>{manifest.icons.length.toLocaleString()}</b> icons</span>
        {:else if !ready}
          <span>Loading model...</span>
        {/if}
      </div>
      <div class="meta-r">
        <div class="seg">
          <button class="seg-b" class:on={dense} onclick={() => dense = true}>Gallery</button>
          <button class="seg-b" class:on={!dense} onclick={() => dense = false}>List</button>
        </div>
        <span class="model mono">{activeModelLabel()}</span>
        <div class="seg">
          {#each manifest.models as model}
            <button
              class="seg-b"
              class:on={model.id === activeModelId}
              disabled={isLoading}
              onclick={() => switchModel(model)}
            >{model.label}</button>
          {/each}
        </div>
        <div class="info-wrap">
          <button
            class="info-btn"
            aria-label="About the model toggle"
            aria-expanded={modelInfoOpen}
            onclick={(e) => { e.stopPropagation(); modelInfoOpen = !modelInfoOpen; }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
          </button>
          {#if modelInfoOpen}
            <!-- svelte-ignore a11y_no_static_element_interactions a11y_click_events_have_key_events -->
            <div class="info-pop" onclick={(e) => e.stopPropagation()}>
              <p>Search runs entirely in your browser: your query is embedded by a small neural net and matched against precomputed icon embeddings. Nothing is sent to a server.</p>
              <p><b>Faster</b> uses <span class="mono">all-MiniLM-L6-v2</span> (~90&thinsp;MB) — quick to download, solid results.</p>
              <p><b>Better</b> uses <span class="mono">bge-small-en-v1.5</span> (~130&thinsp;MB) — a larger model with noticeably stronger semantic ranking.</p>
              <p><b>Multilingual</b> uses <span class="mono">paraphrase-multilingual-MiniLM-L12-v2</span> (~115&thinsp;MB) — search in 50+ languages (for English, Better ranks best).</p>
              <p>Each model downloads once on first use, then loads from browser cache.</p>
              <p>Tip: descriptive phrases ("waiting for a download") match better than single keywords.</p>
            </div>
          {/if}
        </div>
      </div>
    </div>

    {#if isLoading}
      <div class="progress-bar">
        <div class="progress-fill" style="width: {modelProgress}%"></div>
      </div>
    {/if}

    {#if results.length > 0}
      <div class={dense ? 'grid' : 'list'}>
        {#each results as result (result.icon.name)}
          {@const scorePercent = Math.round(result.score * 100)}
          {#if dense}
            <!-- svelte-ignore a11y_no_static_element_interactions a11y_click_events_have_key_events -->
            <div
              class="cell"
              class:sel={selectedIcon?.name === result.icon.name}
              onclick={() => selectedIcon = result.icon}
            >
              <span class="cell-m">{scorePercent}</span>
              <button class="copy" onclick={(e) => { e.stopPropagation(); copyText(result.icon.name, `Copied "${result.icon.name}"`); }} title="Copy name" aria-label="Copy icon name">
                <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
              </button>
              <span class="cell-icon">{@html result.icon.svg}</span>
              <span class="cell-n">{result.icon.name}</span>
              <span class="cell-dot" style="background: {clusterColor(clusterColors, result.icon.cluster)}"></span>
            </div>
          {:else}
            <!-- svelte-ignore a11y_no_static_element_interactions a11y_click_events_have_key_events -->
            <div
              class="cell row"
              class:sel={selectedIcon?.name === result.icon.name}
              onclick={() => selectedIcon = result.icon}
            >
              <span class="cell-tile">
                <span class="cell-tile-icon">{@html result.icon.svg}</span>
              </span>
              <span class="cell-rmid">
                <span class="cell-n mono">{result.icon.name}</span>
                <span class="cell-sub">
                  <span class="cell-dot static" style="background: {clusterColor(clusterColors, result.icon.cluster)}"></span>
                  {result.icon.cluster}
                </span>
              </span>
              <span class="cell-m static">{scorePercent}%</span>
              <button class="copy row-copy" onclick={(e) => { e.stopPropagation(); copyText(result.icon.name, `Copied "${result.icon.name}"`); }} title="Copy name" aria-label="Copy icon name">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
              </button>
            </div>
          {/if}
        {/each}
      </div>
    {:else if query && ready}
      <div class="empty">
        <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="var(--tx3)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m13.5 8.5-5 5"/><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
        <p>No icons match "{query}".</p>
      </div>
    {:else if !ready}
      <div class="empty">
        <p>Loading model...</p>
      </div>
    {/if}

    <footer class="foot">
      <span>powered by <a class="repo mono" href="https://github.com/mmacpherson/python-lucide" target="_blank" rel="noopener">python-lucide</a>{#if manifest.packageVersion}&nbsp;<span class="mono dim">v{manifest.packageVersion}</span>{/if} &middot; {manifest.icons.length.toLocaleString()} icons shipped in SQLite</span>
      <span class="dim">embeddings computed in-browser via <a href="https://huggingface.co/docs/transformers.js" target="_blank">transformers.js</a></span>
    </footer>
  {/if}
</main>

<DetailPanel
  icon={selectedIcon}
  {clusterColors}
  {similarIcons}
  onclose={() => selectedIcon = null}
  oncopy={copyText}
  onpick={(icon) => selectedIcon = icon}
/>

<style>
  .main {
    flex: 1; width: 100%; max-width: 1120px;
    margin: 0 auto; padding: 30px 28px 0;
    position: relative; z-index: 1;
    display: flex; flex-direction: column;
  }

  /* ── Search bar ──────────────────────────────────────────────── */
  .searchbar {
    display: flex; align-items: center; gap: 13px;
    background: var(--surf); border: 1px solid var(--bd2);
    border-radius: 13px; padding: 14px 16px;
    box-shadow: 0 0 0 4px var(--acs); transition: .15s;
  }
  .searchbar input {
    flex: 1; border: none; outline: none; background: transparent;
    color: var(--tx); font-family: var(--font); font-size: 16px;
  }
  .searchbar input::placeholder { color: var(--tx3); }
  .searchbar input:disabled { opacity: 0.5; }
  .clear {
    border: none; background: transparent; cursor: pointer;
    display: flex; padding: 3px; border-radius: 6px;
  }
  .clear:hover { background: var(--surf2); }

  /* ── Example chips ──────────────────────────────────────────── */
  .examples {
    display: flex; align-items: center; flex-wrap: wrap;
    gap: 7px; margin: 13px 4px 0;
  }
  .ex-label {
    font-size: 12px; color: var(--tx3); font-weight: 600;
    margin-right: 2px;
  }
  .chip {
    font-family: var(--font); font-size: 12px; font-weight: 500;
    color: var(--tx2); background: var(--surf);
    border: 1px solid var(--bd); border-radius: 999px;
    padding: 5px 12px; cursor: pointer;
    transition: color .14s, border-color .14s, background .14s;
  }
  .chip:hover {
    color: var(--tx); border-color: var(--ac); background: var(--acs);
  }

  /* ── Meta row ────────────────────────────────────────────────── */
  .meta {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; margin: 16px 2px 20px; flex-wrap: wrap;
  }
  .meta-l { font-size: 13px; color: var(--tx2); }
  .meta-l b { color: var(--tx); font-weight: 700; }
  .meta-r { display: flex; align-items: center; gap: 11px; }
  .model { font-size: 11px; color: var(--tx3); }
  .mono { font-family: var(--font-mono); }

  .seg {
    display: inline-flex; background: var(--surf);
    border: 1px solid var(--bd); border-radius: 9px; padding: 3px; gap: 2px;
  }
  .seg-b {
    font-family: var(--font); font-size: 12.5px; font-weight: 600;
    padding: 5px 12px; border: none; border-radius: 6px;
    color: var(--tx2); background: transparent;
    cursor: pointer; transition: .14s; white-space: nowrap;
  }
  .seg-b:hover { color: var(--tx); }
  .seg-b.on {
    color: #fff; background: var(--ac);
    box-shadow: 0 1px 3px rgba(0,0,0,.25);
  }
  .seg-b:disabled { cursor: wait; opacity: 0.7; }

  /* ── Model info popover ─────────────────────────────────────── */
  .info-wrap { position: relative; display: flex; }
  .info-btn {
    width: 26px; height: 26px; border-radius: 7px;
    border: none; background: transparent; color: var(--tx3);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; transition: background .15s, color .15s;
  }
  .info-btn:hover, .info-btn[aria-expanded="true"] {
    background: var(--surf2); color: var(--tx);
  }
  .info-pop {
    position: absolute; top: calc(100% + 8px); right: 0;
    width: 290px; z-index: 40;
    background: var(--bg2); border: 1px solid var(--bd2);
    border-radius: 12px; padding: 14px 16px;
    box-shadow: var(--shadow);
    font-size: 12.5px; color: var(--tx2); line-height: 1.55;
    cursor: auto;
  }
  .info-pop p + p { margin-top: 8px; }
  .info-pop b { color: var(--tx); }
  .info-pop .mono { font-size: 11.5px; color: var(--tx); }

  /* ── Progress bar ────────────────────────────────────────────── */
  .progress-bar {
    height: 3px; background: var(--bd);
    border-radius: 2px; margin-bottom: 16px; overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: var(--ac);
    transition: width 0.2s ease-out; border-radius: 2px;
  }

  /* ── Dense gallery ──────────────────────────────────────────── */
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 8px; padding-bottom: 8px;
  }
  .cell {
    position: relative; display: flex; flex-direction: column;
    align-items: center; gap: 10px;
    padding: 18px 8px 13px;
    background: var(--surf); border: 1px solid var(--bd);
    border-radius: 12px; cursor: pointer;
    transition: transform .12s, border-color .12s, background .12s;
    color: var(--tx);
  }
  .cell:hover {
    background: var(--surf2); border-color: var(--bd2);
    transform: translateY(-2px);
  }
  .cell.sel {
    border-color: var(--ac); box-shadow: 0 0 0 3px var(--acs);
  }
  .cell-m {
    position: absolute; top: 6px; left: 8px;
    font-family: var(--font-mono); font-size: 9.5px; color: var(--tx3);
  }
  .cell-icon { display: flex; align-items: center; justify-content: center; color: var(--ic); }
  .cell-icon :global(svg) { width: 26px; height: 26px; stroke-width: 1.75; }
  .cell-n {
    font-family: var(--font-mono); font-size: 10px; color: var(--tx2);
    max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .cell-dot {
    position: absolute; bottom: 7px; left: 50%; transform: translateX(-50%);
    width: 5px; height: 5px; border-radius: 5px;
  }
  .cell-dot.static {
    position: static; transform: none;
    width: 6px; height: 6px; flex: 0 0 auto;
  }

  .copy {
    position: absolute; top: 5px; right: 6px; opacity: 0;
    width: 22px; height: 22px; border-radius: 6px; border: none;
    background: var(--surf2); color: var(--tx2);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; transition: opacity .12s, background .12s;
  }
  .cell:hover .copy { opacity: 1; }
  .copy:hover { background: var(--ac); color: #fff; }

  /* ── List density ───────────────────────────────────────────── */
  .list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
    gap: 9px; padding-bottom: 8px;
  }
  .cell.row {
    flex-direction: row; align-items: center; gap: 13px;
    padding: 11px 13px; text-align: left;
  }
  .cell-tile {
    width: 42px; height: 42px; flex: 0 0 auto;
    border-radius: 9px; background: var(--surf2);
    border: 1px solid var(--bd);
    display: flex; align-items: center; justify-content: center;
    color: var(--ic);
  }
  .cell-tile-icon { display: flex; align-items: center; justify-content: center; }
  .cell-tile-icon :global(svg) { width: 22px; height: 22px; stroke-width: 1.75; }
  .cell-rmid { display: flex; flex-direction: column; gap: 5px; min-width: 0; flex: 1; }
  .cell.row .cell-n { font-size: 13px; color: var(--tx); }
  .cell-sub {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: var(--tx3);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .cell-m.static {
    position: static; font-size: 11px; color: var(--ac);
    flex: 0 0 auto; font-family: var(--font-mono);
  }
  .row-copy {
    position: static; opacity: 0;
  }
  .cell.row:hover .row-copy { opacity: 1; }

  /* ── Empty / loading ─────────────────────────────────────────── */
  .empty { text-align: center; padding: 70px 0; color: var(--tx2); }
  .empty p { margin-top: 12px; font-size: 14px; }
  .error {
    color: #ff6b6b; padding: 16px;
    background: #2a1515; border-radius: 8px;
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
  }
  .retry {
    border: 1px solid #ff6b6b; background: transparent; color: #ff6b6b;
    border-radius: 6px; padding: 5px 14px; cursor: pointer;
    font-family: var(--font); font-size: 12.5px; flex: 0 0 auto;
  }
  .retry:hover { background: #ff6b6b; color: #fff; }

  /* ── Footer ──────────────────────────────────────────────────── */
  .foot {
    margin-top: auto; display: flex; align-items: center;
    justify-content: space-between; gap: 12px;
    padding: 26px 2px; font-size: 12px; color: var(--tx2);
    border-top: 1px solid var(--bd); margin-top: 26px;
    flex-wrap: wrap;
  }
  .foot .repo { color: var(--tx); font-weight: 700; }
  .foot .repo:hover { color: var(--ac); }
  .dim { color: var(--tx3); }
  .dim a { color: var(--tx2); }
  .dim a:hover { color: var(--tx); text-decoration: underline; }
</style>
