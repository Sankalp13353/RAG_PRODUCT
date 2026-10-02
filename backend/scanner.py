import os
import sys
from sentence_transformers import SentenceTransformer

IGNORED_DIRS = {".git", "node_modules", "venv", "__pycache__", "dist", "build"}

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".cpp", ".c", ".go", ".rs",
    ".md", ".json", ".yaml", ".yml",
}

CHUNK_SIZE = 50

def scan_repository(repo_path):
    total_files = 0
    supported_files = 0
    ignored_files = 0
    type_counts = {}
    processed_files = []
    all_chunks = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file in files:
            total_files += 1
            ext = os.path.splitext(file)[1].lower()

            if ext in SUPPORTED_EXTENSIONS:
                supported_files += 1
                type_counts[ext] = type_counts.get(ext, 0) + 1

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)

                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                file_lines = content.splitlines()
                lines = len(file_lines)
                chars = len(content)

                processed_files.append((rel_path, ext, lines, chars))

                for i in range(0, lines, CHUNK_SIZE):
                    chunk_lines = file_lines[i : i + CHUNK_SIZE]
                    chunk = {
                        "file_path": rel_path,
                        "file_type": ext,
                        "chunk_number": (i // CHUNK_SIZE) + 1,
                        "start_line": i + 1,
                        "end_line": min(i + CHUNK_SIZE, lines),
                        "content": "\n".join(chunk_lines),
                    }
                    all_chunks.append(chunk)
            else:
                ignored_files += 1

    print("\n--- Repository Scan Summary ---")
    print(f"Path: {repo_path}")
    print(f"Total files: {total_files}")
    print(f"Supported files: {supported_files}")
    print(f"Ignored/Unprocessed files: {ignored_files}")
    print("\nFiles by type:")
    for ext, count in sorted(type_counts.items()):
        print(f"  {ext}: {count}")

    print("\n--- Processed Files ---")
    for path, ext, lines, chars in processed_files:
        print(f"\nFile: {path}")
        print(f"Type: {ext}")
        print(f"Lines: {lines}")
        print(f"Characters: {chars}")

    print(f"\nTotal supported files processed: {len(processed_files)}")

    print("\n--- Sample Chunks ---")
    for chunk in all_chunks[:3]:
        print(f"\nFile: {chunk['file_path']}")
        print(f"Type: {chunk['file_type']}")
        print(f"Chunk: {chunk['chunk_number']}")
        print(f"Lines: {chunk['start_line']} - {chunk['end_line']}")
        preview = chunk["content"][:150].strip()
        print(f"Content preview:\n{preview}")

    print(f"\nTotal chunks created: {len(all_chunks)}")

    if all_chunks:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        contents = [chunk["content"] for chunk in all_chunks]
        embeddings = model.encode(contents)

        for chunk, emb in zip(all_chunks, embeddings):
            chunk["embedding"] = emb

        dimensions = len(all_chunks[0]["embedding"])

        print("\n--- Embedding Summary ---")
        print(f"\nTotal chunks: {len(all_chunks)}")
        print(f"Embedding dimensions: {dimensions}")
        print("\nFirst chunk:")
        print(f"File: {all_chunks[0]['file_path']}")
        print(f"Chunk: {all_chunks[0]['chunk_number']}")
        print(f"Embedding length: {len(all_chunks[0]['embedding'])}")

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "."
    scan_repository(target_path)
