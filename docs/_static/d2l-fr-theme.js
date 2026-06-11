(function () {
  function preferredTheme() {
    var saved = localStorage.getItem("d2l-fr-theme");
    if (saved === "light" || saved === "dark") return saved;
    return "dark";
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    var button = document.querySelector(".d2l-theme-toggle");
    if (button) {
      button.textContent = theme === "dark" ? "Mode clair" : "Mode sombre";
      button.setAttribute(
        "aria-label",
        theme === "dark" ? "Passer au thème clair" : "Passer au thème sombre"
      );
    }
  }

  applyTheme(preferredTheme());

  document.addEventListener("DOMContentLoaded", function () {
    if (!document.querySelector(".d2l-theme-toggle")) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "d2l-theme-toggle";
      button.addEventListener("click", function () {
        var current = document.documentElement.getAttribute("data-theme") || "dark";
        var next = current === "dark" ? "light" : "dark";
        localStorage.setItem("d2l-fr-theme", next);
        applyTheme(next);
      });
      document.body.appendChild(button);
    }
    applyTheme(preferredTheme());
  });
})();
