(function () {
  const slides = Array.from(document.querySelectorAll(".slide"));
  if (!slides.length) return;

  let index = 0;

  function render() {
    slides.forEach((slide, i) => {
      slide.classList.toggle("active", i === index);
    });

    const counter = document.querySelector("[data-counter]");
    if (counter) counter.textContent = `${index + 1} / ${slides.length}`;
  }

  function next() {
    index = Math.min(index + 1, slides.length - 1);
    render();
  }

  function prev() {
    index = Math.max(index - 1, 0);
    render();
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight" || event.key === "PageDown" || event.key === " ") next();
    if (event.key === "ArrowLeft" || event.key === "PageUp") prev();
  });

  document.querySelector("[data-next]")?.addEventListener("click", next);
  document.querySelector("[data-prev]")?.addEventListener("click", prev);

  render();
})();
