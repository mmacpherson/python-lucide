## Python Style & Philosophy
* **Correctness first**: Prioritize code correctness and clarity over performance unless explicitly optimizing
* **Zen-informed**: Follow Python's philosophy - simple is better than complex, explicit is better than implicit
* **Functional flavor**: Embrace immutability and functional patterns where they improve clarity, but stay Pythonic
* **Modern syntax**: Use contemporary Python features and avoid legacy patterns

## Type Annotations
* Use modern union syntax: `str | None` not `Optional[str]`
* Use built-in generics: `dict[str, int]` not `Dict[str, int]`
* Use `list[T]` not `List[T]`
* Annotate function signatures, but avoid over-annotating obvious cases

## File Organization
* **Minimize file proliferation**: Add functionality to existing files unless creating a distinct logical component
* **Use pathlib**: Prefer `pathlib` over `os.path` for file operations
* **Logical grouping**: Organize code by functionality, not artificial boundaries

## Comments & Documentation
* **No summarizing comments**: Don't describe what the code does
* **Explain the why**: Comment on non-obvious reasoning, trade-offs, or constraints
* **Context over content**: Focus on business logic and decision rationale
* **Google style**: Use Google style docstrings for functions and classes

## Learning & Alternatives
* **Open to alternatives**: Appreciate hearing about different approaches and trade-offs
* **Contextual advice**: Consider whether I'm exploring/learning vs. shipping/focused
* **Explain reasoning**: Help me understand why an approach might be better

## Code Quality
* **Readable over clever**: Choose clarity over showing off language features
* **Consistent patterns**: Maintain consistency within a codebase
* **Error handling**: Be explicit about error cases and edge conditions
* **Testing mindset**: Write code that's easy to test and debug

## GitHub Actions & Workflows
* **Use uv run**: Always use `uv run python` instead of bare `python` in workflows after `uv sync`
* **Check with make**: Use `make run-hooks-all-files` and `make test` for quality checks
