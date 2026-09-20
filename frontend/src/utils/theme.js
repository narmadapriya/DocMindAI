const THEME_KEY = "docmind_theme";

export function getStoredTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === "light" || saved === "dark") return saved;

  return window.matchMedia?.("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

export function applyTheme(theme) {
  const value = theme === "light" ? "light" : "dark";
  document.documentElement.classList.remove("light", "dark");
  document.documentElement.classList.add(value);
  document.documentElement.dataset.theme = value;
  document.documentElement.style.colorScheme = value;
  localStorage.setItem(THEME_KEY, value);
  window.dispatchEvent(
    new CustomEvent("docmind:theme", { detail: value }),
  );
  return value;
}

export function toggleTheme() {
  return applyTheme(
    document.documentElement.classList.contains("light")
      ? "dark"
      : "light",
  );
}

export function initializeTheme() {
  return applyTheme(getStoredTheme());
}
