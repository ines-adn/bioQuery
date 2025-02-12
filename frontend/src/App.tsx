import React, { useState } from "react";
import "./App.css";
interface Result {
  title: string;
  authors: string[];
  abstract: string;
  // Add other fields as necessary
}

function App() {
  const [ingredient, setIngredient] = useState("");
  const [allegation, setAllegation] = useState("");
  const [useSemantic, setUseSemantic] = useState(false);
  const [usePubmed, setUsePubmed] = useState(false);
  const [results, setResults] = useState<Result[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async () => {
    const params = new URLSearchParams();
    params.append("ingredient", ingredient);
    params.append("allegation", allegation);

    let url = "";
    console.log("Fetching data from:", url);

    if (useSemantic && usePubmed) {
      url = `/search/both/?${params.toString()}`;
    } else if (useSemantic) {
      url = `/search/semantic_scholars/?${params.toString()}`;
    } else if (usePubmed) {
      url = `/search/pubmed/?${params.toString()}`;
    }

    console.log("Fetching data from:", url);

    if (url) {
      setLoading(true);
      try {
        const fullURL = "http://localhost:8000" + url
        const response = await fetch(fullURL);
        console.log(response)
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
    } else {
      setError("No search method selected.");
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Recherche d'articles scientifiques</h1>

        <div>
          <label>
            Ingrédient :
            <input
              type="text"
              value={ingredient}
              onChange={(e) => setIngredient(e.target.value)}
            />
          </label>
        </div>

        <div>
          <label>
            Allégation :
            <input
              type="text"
              value={allegation}
              onChange={(e) => setAllegation(e.target.value)}
            />
          </label>
        </div>

        <div>
          <label>
            Utiliser Semantic Scholars :
            <input
              type="checkbox"
              checked={useSemantic}
              onChange={() => setUseSemantic(!useSemantic)}
            />
          </label>
        </div>

        <div>
          <label>
            Utiliser PubMed :
            <input
              type="checkbox"
              checked={usePubmed}
              onChange={() => setUsePubmed(!usePubmed)}
            />
          </label>
        </div>

        <button onClick={handleSearch} disabled={loading}>
          {loading ? "Chargement..." : "Chercher"}
        </button>

        {error && <div style={{ color: "red" }}>{error}</div>}

        {results && (
          <div>
            <h2>Résultats :</h2>
            <pre>{JSON.stringify(results, null, 2)}</pre>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;