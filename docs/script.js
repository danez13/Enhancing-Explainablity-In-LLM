const hamburger = document.querySelector('.hamburger');
const navLinks = document.querySelector('nav ul');
const dropdowns = document.querySelectorAll('nav ul li');

hamburger.addEventListener('click', () => {
    navLinks.classList.toggle('active');
});

dropdowns.forEach(dropdown => {
    dropdown.addEventListener('click', (e) => {
        e.stopPropagation();  // Prevent the click event from closing the menu
        dropdown.classList.toggle('active');
    });
});

