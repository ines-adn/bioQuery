# bioQuery

*A lightweight AI-powered summary generator to understand your cosmetic and food consumption.*

## 🎯 Overview

bioQuery empowers consumers to make informed decisions about the products they use by providing accessible, science-based information about cosmetic and food ingredients. While tools like Yuka help consumers identify potentially harmful ingredients, bioQuery is willing to go further by providing comprehensive scientific summaries based on the latest research.

The tool leverages **Retrieval-Augmented Generation (RAG)** to transform complex scientific literature into easy-to-understand summaries, making cutting-edge research accessible to everyone.

## ✨ Key Features

- **Intelligent Article Retrieval**: Automatically searches and downloads scientific articles from Semantic Scholar
- **Multi-Language Support**: Generate summaries in French and English
- **Flexible AI Models**: Choose between OpenAI (GPT-4) or local Llama models
- **Smart Caching**: Avoid redundant downloads with local article caching
- **Vector Database**: Efficient similarity search using PostgreSQL with pgvector
- **RESTful API**: Easy integration with web applications

## 🧠 How RAG Works in bioQuery
bioQuery implements Retrieval-Augmented Generation (RAG) to provide accurate, research-backed summaries of cosmetic and food ingredients. Here's how the process works:
### The RAG Pipeline

1. **Knowledge Base Creation**: Scientific articles are downloaded, processed into text chunks, and converted into vector embeddings using HuggingFace's sentence transformers. These embeddings capture the semantic meaning of each text chunk.

2. **Vector Storage**: Embeddings are stored in PostgreSQL with the pgvector extension, enabling efficient similarity searches across thousands of research snippets.

3. **Intelligent Retrieval**: When you query an ingredient, the system:
    - Converts your query into a vector embedding
    - Searches the database for the most semantically similar research chunks
    - Retrieves the top-k most relevant pieces of scientific evidence


4. **Contextual Generation**: The retrieved research chunks are fed to an LLM (OpenAI GPT-4 or local Llama) along with your query, enabling the model to generate summaries grounded in actual scientific literature rather than just training data.

This approach ensures that summaries are factual, up-to-date, and traceable to specific research sources, making bioQuery more reliable than traditional AI assistants for scientific information.




## 🏗️ Architecture

### Backend Components

```
backend/
├── corpus_retrieval/
│   ├── scrapers/           # Article retrieval from Semantic Scholar
│   ├── parsers/           # Document processing and embedding storage
│   └── summarizers/       # AI-powered summary generation
├── main.py               # FastAPI application
└── config.json          # Configuration file
```

#### Core Modules

1. **Scrapers** (`semantic_scholars_scraping.py`)
   - Intelligent query generation using LLMs
   - Article search and PDF download from Semantic Scholar
   - Local caching system for efficiency

2. **Parsers** (`embedding_store.py`, `article_chunker.py`)
   - PDF text extraction and chunking
   - Vector embeddings using HuggingFace models
   - PostgreSQL + pgvector storage

3. **Summarizers** (`llama_summarizer.py`)
   - Scientific summary generation
   - Support for both OpenAI and Ollama models
   - Batch processing for large document sets

### Frontend

React-based web interface for easy interaction with the bioQuery API.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Node.js (for frontend)
- Docker (for Ollama, optional in case you have OpenAI API Key)
- Ollama (optional in case you have OpenAI API Key)

- **psql** must be installed (e.g., via Homebrew on macOS):
  ```bash
  brew install postgresql
  ```
- uvicorn must be installed to run the FastAPI server:
```
pip install uvicorn
```

- If using Ollama, the model must be pulled before running:
```
ollama pull llama3.2:3b
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ines-adn/bioquery.git
   cd bioquery
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL with pgvector**
   ```bash
   # Install pgvector extension in your PostgreSQL instance
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. **Create and configure the PostgreSQL database**
    - Connect to PostgreSQL as superuser
    ```
    psql -U postgres
    ```

    - Create the database
    ```
    CREATE DATABASE "bioQuery";
    ```

    - Connect to the new database
    \c bioQuery

    - Install the pgvector extension
    ```
    CREATE EXTENSION IF NOT EXISTS vector;
    ```

    - Exit psql
    ```
    \q
    ```
    The application will automatically create the required tables (langchain_pg_collection and langchain_pg_embedding) on first run. These tables store the vector embeddings and collection metadata used by the RAG system.

5. **Configure the application**
   
   Create a `config.json` file in the backend directory:
   ```json
    {
    "base_dir": "path_to_the_repo",
    "postgres": {
      "host": "localhost",
      "port": 5432,
      "database": "bioQuery",
      "user": "your_user_name",
      "password": "either_null_or_your_password"
    },
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_type": "openai",
    "openai_model": "gpt-4.1",
    "ollama_model": "llama3.2:3b",
    "llm_temperature": 0.5
    }
   ```

6. **Set environment variables** (if using OpenAI)
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```

7. **Install frontend dependencies**
   ```bash
   npm install
   ```

### Running the Application

1. **Start the backend**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`

2. **Start the frontend**
   ```bash
   npm start
   ```
   The web interface will be available at `http://localhost:3000`

## 🔧 Configuration Options

### LLM Models

**OpenAI (Recommended for production)**
- Requires API key
- Higher quality summaries
- Faster processing
- Costs per API call

**Ollama (Free alternative)**
- Runs locally
- No API costs
- Requires more computational resources
- Slightly lower quality

### API Endpoints

#### Complete Search and Summary
```
GET /search/complete/?ingredient={ingredient_name}
```

**Parameters:**
- `ingredient` (required): Name of the ingredient to research
- `use_openai` (optional): Force OpenAI usage (default: auto-detect)
- `use_cache` (optional): Use cached articles if available
- `language` (optional): Summary language ("fr" or "en", default: "fr")
- `max_chunks` (optional): Maximum document chunks to process
- `process_pdfs` (optional): Automatically process downloaded PDFs

**Example:**
```bash
curl "http://localhost:8000/search/complete/?ingredient=aloe%20vera&language=en&use_cache=true"
```

## 📖 Usage Examples

### Basic Usage

![Demo](demo_bq.gif)

## 🔍 How It Works

1. **Query Generation**: LLM generates optimized search queries for Semantic Scholar
2. **Article Retrieval**: Downloads relevant scientific articles as PDFs
3. **Document Processing**: Extracts text and splits into manageable chunks
4. **Vectorization**: Converts text chunks into embeddings using HuggingFace models
5. **Storage**: Stores embeddings in PostgreSQL with pgvector for efficient similarity search
6. **Summary Generation**: LLM synthesizes information from relevant chunks into coherent summaries

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## ⚠️ Important Notes

- **Semantic Scholar API Limits**: Be mindful of API rate limits when searching for articles
- **PDF Processing**: Some PDFs may not be accessible due to publisher restrictions
- **Model Requirements**: Local Ollama models require significant computational resources
- **Data Storage**: Vector embeddings require substantial disk space for large corpora

## 🆘 Troubleshooting

### Common Issues

**Database Connection Errors**
- Ensure PostgreSQL is running and pgvector extension is installed
- Check database credentials in `config.json`

**API Rate Limiting**
- Semantic Scholar has rate limits; the tool includes automatic retry logic
- Use caching to minimize API calls

**Memory Issues with Ollama**
- Reduce `max_chunks_ollama` parameter
- Use smaller model variants (e.g., `llama3.2:3b`)

## 📊 Performance Tips

- Enable caching for repeated queries
- Use smaller chunk sizes for faster processing
- Consider using OpenAI for production workloads
- Implement horizontal scaling for high-volume usage

## 🔮 Future Enhancements

- [ ] Support for additional scientific databases
- [ ] Personalized recommendations based on user preferences
- [ ] Integration with popular shopping platforms
- [ ] Mobile application development

---

**Made with ❤️ for informed consumer choices**