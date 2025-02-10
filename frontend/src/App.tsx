import React, { useState } from "react";
import "./App.css";

function App() {
  const [ingredient, setIngredient] = useState("");
  const [allegation, setAllegation] = useState("");
  const [useSemantic, setUseSemantic] = useState(false);
  const [usePubmed, setUsePubmed] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleSearch = async () => {
    const params = new URLSearchParams();
    params.append("ingredient", ingredient);
    params.append("allegation", allegation);

    let url = "";

    if (useSemantic && usePubmed) {
      url = `/search/both/?${params.toString()}`;
    } else if (useSemantic) {
      url = `/search/semantic_scholars/?${params.toString()}`;
    } else if (usePubmed) {
      url = `/search/pubmed/?${params.toString()}`;
    }

    try {
      const response = await fetch(url);
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      console.error("Error fetching data:", error);
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

        <button onClick={handleSearch}>Chercher</button>

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