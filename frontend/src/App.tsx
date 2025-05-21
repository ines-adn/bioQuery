import React, { useState, useEffect } from "react";
import "./App.css";

interface Result {
  number: number;
  title: string;
  url: string;
}

interface Summary {
  text: string;
  id: string;
  processing_time?: number;
  chunks_processed?: number;
}

// Mise à jour de la structure de réponse pour la nouvelle API
interface SearchResponse {
  status: string;
  message: string;
  semantic_results: Result[];
  summary: Summary | null;
}

function TypingText({ text, speed = 50 }: { text: string; speed?: number }) {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    let index = 0;
    setDisplayedText("");
    if (!text) return;

    const interval = setInterval(() => {
      setDisplayedText((prev) => text.slice(0, index + 1));
      index++;
      if (index >= text.length) clearInterval(interval);
    }, speed);

    return () => clearInterval(interval);
  }, [text]);

  return <>{displayedText}</>;
}

// Logo SVG Component
function BioQueryLogo({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
      <circle cx="60" cy="60" r="55" fill="#5652ff" />
      <path d="M85 85 L105 105" stroke="white" strokeWidth="10" strokeLinecap="round" />
      <circle cx="60" cy="55" r="30" fill="none" stroke="white" strokeWidth="8" />
      <path d="M42 40 Q60 60 42 80" stroke="white" strokeWidth="5" strokeLinecap="round" fill="none" />
      <path d="M78 40 Q60 60 78 80" stroke="white" strokeWidth="5" strokeLinecap="round" fill="none" />
      <line x1="43" y1="45" x2="77" y2="45" stroke="white" strokeWidth="4" strokeLinecap="round" />
      <line x1="41" y1="60" x2="79" y2="60" stroke="white" strokeWidth="4" strokeLinecap="round" />
      <line x1="43" y1="75" x2="77" y2="75" stroke="white" strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}

function App() {
  const [ingredient, setIngredient] = useState("");
  const [results, setResults] = useState<Result[] | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeSection, setActiveSection] = useState("search");

  useEffect(() => {
    const handleScroll = () => {
      const sections = ["search", "how-it-works", "about", "contact"];
      const scrollPosition = window.scrollY + 100;
      
      for (const section of sections) {
        const element = document.getElementById(section);
        if (element) {
          const offsetTop = element.offsetTop;
          const offsetBottom = offsetTop + element.offsetHeight;
          
          if (scrollPosition >= offsetTop && scrollPosition < offsetBottom) {
            setActiveSection(section);
            break;
          }
        }
      }
    };
    
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleSearch = async () => {
    if (!ingredient.trim()) {
      setError("Veuillez entrer un composant ou un ingrédient");
      return;
    }
    
    const params = new URLSearchParams();
    params.append("ingredient", ingredient);
    // Utiliser la nouvelle route API qui fait le processus complet
    const url = `/search/complete/?${params.toString()}`;

    setError("");
    setLoading(true);
    setSummary(null);
    setResults(null);
    
    try {
      const fullURL = "http://localhost:8000" + url;
      const response = await fetch(fullURL);
      
      if (response.ok) {
        const data: SearchResponse = await response.json();
        
        // Extraire le résumé et les résultats
        setSummary(data.summary);
        setResults(data.semantic_results);
        
        // Scroll to results after they load
        if ((data.semantic_results && data.semantic_results.length > 0) || data.summary) {
          setTimeout(() => {
            document.getElementById('results-section')?.scrollIntoView({ 
              behavior: 'smooth' 
            });
          }, 300);
        }
      } else {
        setError("Erreur : rafraîchissez la page et réessayez");
      }
    } catch (error) {
      setError("Erreur : rafraîchissez la page et réessayez");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const scrollToSection = (sectionId: string) => {
    document.getElementById(sectionId)?.scrollIntoView({ 
      behavior: 'smooth' 
    });
  };

  return (
    <div className="app">
      <header className="header">
        <div className="container">
          <nav className="navbar">
            <div className="header-logo">
              <BioQueryLogo size={32} />
              <span className="bioquery-logo-small">bioQuery</span>
            </div>
            <div className="nav-links">
              <a 
                href="#search" 
                className={`nav-link ${activeSection === "search" ? "active" : ""}`}
                onClick={(e) => {
                  e.preventDefault();
                  scrollToSection("search");
                }}
              >
                Recherche
              </a>
              <a 
                href="#how-it-works" 
                className={`nav-link ${activeSection === "how-it-works" ? "active" : ""}`}
                onClick={(e) => {
                  e.preventDefault();
                  scrollToSection("how-it-works");
                }}
              >
                Comment ça marche
              </a>
              <a 
                href="#about" 
                className={`nav-link ${activeSection === "about" ? "active" : ""}`}
                onClick={(e) => {
                  e.preventDefault();
                  scrollToSection("about");
                }}
              >
                À propos
              </a>
              <a 
                href="#contact" 
                className={`nav-link ${activeSection === "contact" ? "active" : ""}`}
                onClick={(e) => {
                  e.preventDefault();
                  scrollToSection("contact");
                }}
              >
                Contact
              </a>
            </div>
          </nav>
        </div>
      </header>

      <div className="hero-section" id="search">
        <div className="container">
          <div className="logo-container">
            <div className="bioquery-logo">
              <BioQueryLogo size={48} />
              <span>bioQuery</span>
            </div>
          </div>
          
          <div className="hero-content">
            <div className="hero-text">
              <h1 className="headline">
                <div className="loupe-icon">
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                  </svg>
                </div>
                <span className="static-text">Comprenez votre consommation</span>
              </h1>
              <p className="subheadline">
                <TypingText 
                  text="Explorez la recherche scientifique derrière les produits que vous consommez."
                  speed={40}
                />
              </p>
            </div>
            
            <div className="search-card">
              <div className="search-container">
                <label htmlFor="ingredient-input">Composant :</label>
                <div className="input-wrapper">
                  <input
                    id="ingredient-input"
                    type="text"
                    value={ingredient}
                    onChange={(e) => setIngredient(e.target.value)}
                    onKeyDown={handleKeyPress}
                    placeholder="Entrez un composant, un ingrédient..."
                  />
                  <button 
                    onClick={handleSearch} 
                    disabled={loading}
                    className={loading ? "loading" : ""}
                  >
                    {loading ? (
                      <span className="loader"></span>
                    ) : (
                      <>
                        <svg className="search-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <circle cx="11" cy="11" r="8"></circle>
                          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                        </svg>
                        Explorer
                      </>
                    )}
                  </button>
                </div>
                
                {error && (
                  <div className="error-message">
                    {error}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {(results || summary) && (
        <div 
          id="results-section" 
          className="results-section"
        >
          <div className="container">
            <div className="results-header">
              <h2>Résultats de recherche</h2>
              <div className="results-badge">
                {results && results.length > 0 ? `${results.length} articles trouvés` : ""}
              </div>
            </div>
            
            {/* Affichage du résumé */}
            {summary && (
              <div className="summary-container">
                <h3>Synthèse scientifique sur {ingredient}</h3>
                <div className="summary-content">
                  {summary.text}
                </div>
                {summary.processing_time && (
                  <div className="summary-stats">
                    <span>Basé sur {summary.chunks_processed} extraits • Généré en {summary.processing_time.toFixed(1)} secondes</span>
                  </div>
                )}
              </div>
            )}
            
            {/* Affichage des articles */}
            {results && results.length > 0 && (
              <>
                <p className="results-intro">
                  Les données précédentes se basent sur les articles suivant à propos du composant <strong>{ingredient}</strong> :
                </p>
                
                <div className="results-grid">
                  {results.map((result, index) => (
                    <div
                      key={index}
                      className="result-card"
                    >
                      <span className="result-number">{index + 1}</span>
                      <h3>{result.title}</h3>
                      <a 
                        href={result.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="result-link"
                      >
                        Voir l'article complet
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                          <polyline points="15 3 21 3 21 9"></polyline>
                          <line x1="10" y1="14" x2="21" y2="3"></line>
                        </svg>
                      </a>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}
      
      <section id="how-it-works" className="info-section">
        <div className="container">
          <h2 className="section-title">Comment ça marche</h2>
          <div className="info-grid">
            <div className="info-card">
              <div className="info-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                </svg>
              </div>
              <h3>Recherchez</h3>
              <p>Entrez le nom d'un composant ou d'un ingrédient qui vous intéresse dans la barre de recherche.</p>
            </div>
            <div className="info-card">
              <div className="info-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                  <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                  <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
              </div>
              <h3>Explorez</h3>
              <p>Découvrez un résumé des articles scientifiques les plus pertinents liés à votre recherche grâce à l'outil bioQuery alimenté par une IA.</p>
            </div>
            <div className="info-card">
              <div className="info-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                  <polyline points="22 4 12 14.01 9 11.01"></polyline>
                </svg>
              </div>
              <h3>Comprenez</h3>
              <p>Accédez aux sources scientifiques pour comprendre les effets des composants sur la santé et l'environnement.</p>
            </div>
          </div>
        </div>
      </section>
      
      <section id="about" className="info-section bg-light">
        <div className="container">
          <h2 className="section-title">À propos de bioQuery</h2>
          <div className="about-content">
            <p>
              bioQuery est une plateforme innovante qui veut faire de l'intelligence artificielle un moyen de mettre la recherche scientifique à la portée de tous. 
              La mission de bioQuery est de permettre aux consommateurs de faire des choix plus éclairés, en leur donnant 
              accès à des informations fiables et à jour sur les composants des produits qu'ils utilisent au quotidien.
            </p>
            <p>
              En connectant les consommateurs aux dernières avancées de la recherche scientifique, 
              bioQuery veut contribuer à une consommation plus responsable et à une meilleure compréhension 
              des impacts de nos choix sur notre santé.
            </p>
          </div>
        </div>
      </section>
      
      <section id="contact" className="info-section">
        <div className="container">
          <h2 className="section-title">Contact</h2>
          <div className="contact-content">
            <p>
              Vous avez des questions, des suggestions ou vous souhaitez en savoir plus sur bioQuery ? 
              N'hésitez pas à nous contacter.
            </p>
            <div className="contact-card">
              <div className="contact-info">
                <div className="contact-item">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                    <polyline points="22,6 12,13 2,6"></polyline>
                  </svg>
                  <a href="mailto:contact@bioquery.fr">ines.adnani@student-cs.fr</a>
                </div>
                {/* <div className="contact-item">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
                  </svg>
                  <a href="tel:+33123456789">+33 1 23 45 67 89</a>
                </div> */}
              </div>
            </div>
          </div>
        </div>
      </section>
      
      <footer className="footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-logo">
              <BioQueryLogo size={24} />
              <div className="bioquery-logo-small">bioQuery</div>
            </div>
            <p className="footer-text">© 2025 bioQuery. Tous droits réservés.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;