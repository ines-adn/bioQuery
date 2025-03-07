import React, { useState } from "react";
import "./App.css";

interface Result {
  number: number;
  title: string;
  url: string;
  // Add other fields as necessary
}

function App() {
  const [ingredient, setIngredient] = useState("");
  const [results, setResults] = useState<Result[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async () => {
    const params = new URLSearchParams();
    params.append("ingredient", ingredient);

    const url = `/search/semantic_scholars/?${params.toString()}`;

    console.log("Fetching data from:", url);

    setLoading(true);
    try {
      const fullURL = "http://localhost:8000" + url;
      const response = await fetch(fullURL);
      if (response.ok) {
        const data = await response.json();
        setResults(data.results);
      } else {
        setError("Error fetching data");
      }
    } catch (error) {
      setError("Error fetching data");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div className="App">
      <header className="App-header"></header>
      <br />
      <a href="/">
        <img src="/banniere-bioquery.png" className="App-logo" alt="logo" />
      </a>
      <br />
      <div className="loupe-container">
        <img src="/loupe-icon.png" className="loupe" alt="loupe" />
        <span className="loupe-text">BioQuery – Comprenez ce que vous consommez</span>
      </div>
      <h2>Explorer la recherche sur {ingredient ? ingredient : "votre composant"}</h2>

      <div className="search-inputs">
        <label>
          Composant :
          <input
            type="text"
            value={ingredient}
            onChange={(e) => setIngredient(e.target.value)}
            onKeyDown={handleKeyPress} // Add this line to trigger search on Enter key
            placeholder="Entrez un composant, un ingrédient..."
          />
        </label>
      </div>

      <button onClick={handleSearch} disabled={loading}>
        {loading ? "Chargement..." : "Explorer"}
      </button>

      {error && <div className="error-message">{error}</div>}

      {results && results.length > 0 && (
        <div className="results-container">
          <h2>Résultats :</h2>
          <ul className="results-list">
            {results.map((result, index) => (
              <li key={index} className="result-item">
                <h3>{result.title}</h3>
                <p>
                  <strong>Lien vers l'article :</strong>{" "}
                  <a href={result.url} target="_blank" rel="noopener noreferrer">
                    {result.url}
                  </a>
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
