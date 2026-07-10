/* artoo-kit helpers: theme toggle, mobile nav, optional mermaid boot.
   No dependencies; safe to include on every page. */
(function () {
  "use strict";

  // Theme: honor a saved choice; the toggle cycles dark -> light -> dark.
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem("artoo-theme"); } catch (e) { /* file:// */ }
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);

  function currentTheme() {
    var explicit = root.getAttribute("data-theme");
    if (explicit) return explicit;
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }

  document.addEventListener("click", function (event) {
    var toggle = event.target.closest("[data-theme-toggle]");
    if (toggle) {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("artoo-theme", next); } catch (e) { /* ignore */ }
      return;
    }
    var navToggle = event.target.closest(".nav-toggle");
    if (navToggle) {
      var links = document.querySelector(".site-nav .nav-links");
      if (links) links.classList.toggle("open");
    }
  });

  // Mermaid: if a vendored mermaid.min.js is loaded, render .mermaid blocks
  // with theme-aware colors.
  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: true,
      theme: currentTheme() === "dark" ? "dark" : "neutral",
      securityLevel: "loose",
    });
  }

  // Mark the current page in the nav.
  var here = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".site-nav .nav-links a").forEach(function (a) {
    var target = a.getAttribute("href").split("/").pop();
    if (target === here) a.setAttribute("aria-current", "page");
  });
})();
