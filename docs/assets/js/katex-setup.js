import "https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.js";
import "https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/contrib/auto-render.min.js";

$(document).ready(() => {
  renderMathInElement(document.body, {
    // ...options...
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\[", right: "\\]", display: true }
    ],
    macros: {
      "\\ind": "\\overset{ind}{\\sim}",
      "\\iid": "\\overset{iid}{\\sim}",
      "\\norm": "\\left\\lVert#1\\right\\rVert",
      "\\p": "\\left(#1\\right)",
      "\\bk": "\\left[#1\\right]",
      "\\bc": "\\left\\{#1\\right\\}",
      "\\abs": "\\left|#1\\right|",
      "\\t": "\\text{#1}",
    }
  });
});
