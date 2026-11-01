// Fit the top bar to its width instead of letting it wrap onto a second row.
// Each step hides a little more (see .cm-fit-* in contimod.css):
//   1. the search field's text, leaving its icon
//   2. the icon links other than GitHub
// Below 960px the theme's own mobile header takes over.
(function () {
  "use strict";
  var STEPS = ["cm-fit-1", "cm-fit-2"];
  document.documentElement.classList.add("cm-header-fit");

  // The links and icons can also spill out of their own box while the bar as a
  // whole still has room: below 1200px they then run under the "On this page" toggle.
  function overflows(header) {
    return [".bd-header__inner", ".navbar-header-items"].some(function (sel) {
      var el = header.querySelector(sel);
      return el && el.scrollWidth > el.clientWidth + 1;
    });
  }

  function fit() {
    var header = document.querySelector(".bd-header");
    if (!header || !header.querySelector(".bd-header__inner")) return;
    STEPS.forEach(function (step) { header.classList.remove(step); });
    for (var i = 0; i < STEPS.length && overflows(header); i++) {
      header.classList.add(STEPS[i]);
    }
  }

  function start() {
    fit();
    // Re-fit when the bar's width changes, and when its parts change size on
    // their own as fonts and icons finish loading.
    var header = document.querySelector(".bd-header");
    if (header && "ResizeObserver" in window) {
      var observer = new ResizeObserver(fit);
      observer.observe(header);
      header.querySelectorAll(".navbar-header-items__start, .navbar-header-items__end")
        .forEach(function (el) { observer.observe(el); });
    }
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(fit);
    }
    // the theme swaps its icon placeholders for SVGs after DOMContentLoaded
    window.addEventListener("load", fit);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
