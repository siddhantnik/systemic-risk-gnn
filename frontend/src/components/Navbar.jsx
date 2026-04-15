import { useState, useEffect } from "react";

export default function Navbar({ activeSlide }) {
  const [scrolled, setScrolled] = useState(false);
  const [theme, setTheme] = useState("dark");

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
  };

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  const labels = ["Vault", "Network", "Target", "Dashboard"];
  const ids = ["slide-intro", "slide-generate", "slide-target", "slide-dashboard"];

  return (
    <nav className={`navbar ${scrolled ? "scrolled" : ""}`}>
      <div className="navbar-brand">GRASP</div>
      <ul className="navbar-links">
        {labels.map((l, i) => (
          <li key={l}>
            <a
              onClick={() => scrollTo(ids[i])}
              style={{ color: activeSlide === i ? "var(--primary-container)" : undefined }}
            >
              {l}
            </a>
          </li>
        ))}
      </ul>
      <button className="theme-toggle" onClick={toggleTheme}>
        {theme === "dark" ? "[ LIGHT ]" : "[ DARK ]"}
      </button>
    </nav>
  );
}
