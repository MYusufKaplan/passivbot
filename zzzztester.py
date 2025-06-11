from rich.console import Console
from rich.text import Text

console = Console()

greenish = Text("Test Greenish RGB", style="rgb(195,232,141)")
console.print(greenish)
