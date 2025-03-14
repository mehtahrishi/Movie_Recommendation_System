document.querySelector("button").addEventListener("click", function () {
    let movieTitle = document.getElementById("movieInput").value.trim();

    if (!movieTitle) {
        alert("Please enter a movie title!");
        return;
    }

    fetch(`https://movie-recommendation-system-twuz.onrender.com//recommend?movie=${encodeURIComponent(movieTitle)}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById("results").innerHTML = `<p style="color:red;text-align:center;">${data.error}</p>`;
            } else {
                let recommendations = data.recommendations.map((movie, index) => `
                    <div class="movie-card" data-index="${index}">
                        <h3>${movie.title}</h3>
                        <p><strong>Overview:</strong> ${movie.overview}</p>
                        <p><strong>Rating:</strong>⭐ ${movie.rating}</p>
                        <p><strong>Budget: </strong> 💲${movie.budget}</p>
                        <p><strong>Revenue:</strong> 💲${movie.revenue}</p>
                        <p><strong>Cast:</strong> ${movie.cast.join(", ")}</p>
                    </div>
                `).join("");

                document.getElementById("results").innerHTML = `
                    <h2>Recommended Movies:</h2>
                    <div class="movies-container">${recommendations}</div>
                `;

                animateCards(); // Call GSAP animation function
            }
        })
        .catch(error => {
            console.error("Error fetching recommendations:", error);
            document.getElementById("results").innerHTML = `<p style="color:red;">Failed to get recommendations. Check console.</p>`;
        });
});

// GSAP Animation Function
function animateCards() {
    const cards = document.querySelectorAll(".movie-card");
    cards.forEach((card, index) => {
        gsap.from(card, {
            x: index % 2 === 0 ? -100 : 100, // Slide from left/right alternately
            opacity: 0,
            duration: 1,
            delay: index * 0.3, // Delay each card slightly for a stagger effect
            ease: "power2.out"
        });
    });
}
