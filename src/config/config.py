import toml


class Config:
    def __init__(self, config_path):
        import os
        with open(config_path, "r") as f:
            data = toml.load(f)
        db = data.get("database", {})
        ollama = data.get("ollama", {})

        self.db_host = db.get("db_host", "localhost")
        self.db_port = db.get("db_port", 5432)
        self.db_name = db.get("db_name")
        self.db_user = db.get("db_user")
        # Prefer config, fallback to env var
        self.db_password = db.get("db_password") or os.getenv("POSTGRES_PASSWORD")
        if not self.db_password:
            raise ValueError("Database password must be provided in config or POSTGRES_PASSWORD env var")

        self.ollama_url = ollama.get("ollama_url", "http://localhost:11434")
        self.model = ollama.get("model", "nomic-embed-text")
