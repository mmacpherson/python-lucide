<script lang="ts">
  import "./app.css";
  import type { IconsManifest } from "./lib/types";
  import { buildClusterColors } from "./lib/clusters";
  import SearchPage from "./pages/SearchPage.svelte";
  import ExplorePage from "./pages/ExplorePage.svelte";

  const BASE = import.meta.env.BASE_URL;

  let manifest: IconsManifest | null = $state(null);
  let clusterColors: Map<string, string> = $state(new Map());
  let loadError = $state("");
  let page = $state(getPage());
  // A 3-state system/light/dark cycle reads as broken whenever the "system"
  // step matches the explicit theme it follows, so keep it binary; old stored
  // "system" values fall through to the OS preference.
  const storedTheme = typeof localStorage !== "undefined" ? localStorage.getItem("ls-theme") : null;
  let theme = $state(
    storedTheme === "light" || storedTheme === "dark"
      ? storedTheme
      : window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light",
  );

  function getPage(): "search" | "explore" {
    return location.hash === "#explore" ? "explore" : "search";
  }

  window.addEventListener("hashchange", () => {
    page = getPage();
  });

  function toggleTheme() {
    theme = theme === "dark" ? "light" : "dark";
    localStorage.setItem("ls-theme", theme);
  }

  $effect(() => {
    document.documentElement.dataset.theme = theme;
  });

  // Show the theme the click switches to, not the current one
  let themeIcon = $derived(
    theme === "dark"
      ? '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>'
      : '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
  );

  async function init() {
    try {
      const resp = await fetch(`${BASE}data/icons.json`);
      manifest = (await resp.json()) as IconsManifest;
      clusterColors = buildClusterColors(manifest.icons);
    } catch (e) {
      loadError = `Failed to load icon data: ${e}`;
    }
  }

  // Toast state
  let toastMsg = $state("");
  let toastTimeout: ReturnType<typeof setTimeout> | null = null;

  function showToast(msg: string) {
    toastMsg = msg;
    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => { toastMsg = ""; }, 1700);
  }

  init();
</script>

<div class="glow"></div>

<header class="hdr">
  <div class="brand">
    <span class="mark">
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m10.065 12.493-6.18 1.318a.934.934 0 0 1-1.108-.702l-.537-2.15a1.07 1.07 0 0 1 .691-1.265l13.504-4.44"/><path d="m13.56 11.747 4.332-.924"/><path d="m16 21-3.105-6.21"/><path d="M16 13a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z"/><path d="m20 9 1.768-1.768A2.245 2.245 0 0 0 20 3a2.244 2.244 0 0 0-1.768 3.232L20 9Z"/></svg>
    </span>
    <span class="wm">Lucide<span class="wm-dim"> Semantic Search</span></span>
    {#if manifest}
      <span class="ver mono">v{manifest.version}</span>
    {/if}
  </div>
  <div class="hdr-r">
    <div class="seg">
      <a href="#search" class="seg-b" class:on={page === "search"}>Search</a>
      <a href="#explore" class="seg-b" class:on={page === "explore"}>Explore</a>
    </div>
    <button
      class="iconbtn"
      onclick={toggleTheme}
      title="Switch to {theme === 'dark' ? 'light' : 'dark'} theme"
      aria-label="Switch to {theme === 'dark' ? 'light' : 'dark'} theme"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="var(--tx2)" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">{@html themeIcon}</svg>
    </button>
    <a class="iconbtn" href="https://github.com/mmacpherson/python-lucide" target="_blank" rel="noopener" title="python-lucide on GitHub" aria-label="python-lucide on GitHub">
      <svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="var(--tx2)" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg>
    </a>
  </div>
</header>

{#if loadError}
  <main class="main"><div class="error">{loadError}</div></main>
{:else if manifest}
  {#if page === "search"}
    <SearchPage {manifest} {clusterColors} {showToast} />
  {:else}
    <ExplorePage {manifest} {clusterColors} />
  {/if}
{/if}

<div class="toast" class:show={!!toastMsg}>
  {#if toastMsg}
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
    {toastMsg}
  {/if}
</div>

<style>
  .glow {
    position: fixed; top: -180px; left: 50%; transform: translateX(-50%);
    width: 760px; height: 340px;
    background: radial-gradient(ellipse at center, var(--glow), transparent 70%);
    pointer-events: none; z-index: 0;
  }

  .hdr {
    position: sticky; top: 0; z-index: 20;
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px;
    border-bottom: 1px solid var(--bd);
    background: color-mix(in srgb, var(--bg) 80%, transparent);
    backdrop-filter: blur(12px);
  }
  .brand { display: flex; align-items: center; gap: 11px; }
  .mark {
    width: 27px; height: 27px; border-radius: 8px;
    background: var(--ac);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 8px var(--acs);
    flex-shrink: 0;
  }
  .wm {
    font-family: var(--font-brand);
    font-size: 15.5px; font-weight: 600; letter-spacing: -0.02em;
    white-space: nowrap;
  }
  .wm-dim { color: var(--tx2); font-weight: 500; }
  .ver { font-size: 11px; color: var(--tx3); margin-left: 2px; }
  .mono { font-family: var(--font-mono); }

  .hdr-r { display: flex; align-items: center; gap: 12px; }

  .seg {
    display: inline-flex;
    background: var(--surf); border: 1px solid var(--bd);
    border-radius: 9px; padding: 3px; gap: 2px;
  }
  .seg-b {
    font-family: var(--font); font-size: 12.5px; font-weight: 600;
    padding: 5px 12px; border: none; border-radius: 6px;
    color: var(--tx2); background: transparent;
    cursor: pointer; transition: .14s; white-space: nowrap;
    text-decoration: none;
  }
  .seg-b:hover { color: var(--tx); text-decoration: none; }
  .seg-b.on {
    color: #fff; background: var(--ac);
    box-shadow: 0 1px 3px rgba(0,0,0,.25);
  }

  .iconbtn {
    width: 32px; height: 32px; border-radius: 8px;
    border: 1px solid transparent; background: transparent;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background .15s, border-color .15s;
    text-decoration: none;
  }
  .iconbtn:hover { background: var(--surf2); text-decoration: none; }

  .main {
    flex: 1; width: 100%; max-width: 1120px;
    margin: 0 auto; padding: 30px 28px 0;
    position: relative; z-index: 1;
  }
  .error {
    color: #ff6b6b; padding: 16px;
    background: #2a1515; border-radius: 8px;
  }

  .toast {
    position: fixed; bottom: 26px; left: 50%;
    transform: translateX(-50%) translateY(18px); opacity: 0;
    transition: .25s cubic-bezier(.4,0,.2,1);
    background: var(--tx); color: var(--bg);
    padding: 10px 16px; border-radius: 10px;
    font-size: 13px; font-weight: 600; z-index: 80;
    display: flex; gap: 8px; align-items: center;
    box-shadow: var(--shadow); pointer-events: none;
  }
  .toast.show {
    opacity: 1; transform: translateX(-50%) translateY(0);
  }
</style>
