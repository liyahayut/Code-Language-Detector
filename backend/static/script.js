document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById('uploadForm');
    const loader = document.getElementById('loader');

    form.addEventListener('submit', () => {
        loader.style.display = 'block';
    });
});

const boxes = document.querySelectorAll('.box');

boxes.forEach(box => {
  box.addEventListener('click', () => {
    box.classList.toggle('flipped');
  });
});
