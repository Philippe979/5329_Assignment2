import nbformat
from nbclient import NotebookClient


INPUT_PATH = "5329_Assignment2.ipynb"
OUTPUT_PATH = "5329_Assignment2_all_models_executed.ipynb"


with open(INPUT_PATH, "r", encoding="utf-8") as file:
    notebook = nbformat.read(file, as_version=4)

client = NotebookClient(
    notebook,
    timeout=86400,
    kernel_name="python3",
    resources={"metadata": {"path": "."}},
)
client.execute()

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    nbformat.write(notebook, file)

print(f"Executed notebook saved to {OUTPUT_PATH}")
