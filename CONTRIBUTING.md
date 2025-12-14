# Contributing to PDF Library

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [License](#license)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior by opening an issue or contacting the maintainers.

## Getting Started

### Prerequisites

- **Rust**: 1.70+ ([Install Rust](https://rustup.rs/))
- **Python**: 3.8+ (for Python bindings)
- **Git**: For version control
- **C Compiler**: gcc or clang (for dependencies)

### Optional Tools

- **cargo-watch**: Auto-reload on file changes
  ```bash
  cargo install cargo-watch
  ```

- **cargo-tarpaulin**: Code coverage
  ```bash
  cargo install cargo-tarpaulin
  ```

- **maturin**: Python packaging
  ```bash
  pip install maturin
  ```

## Development Setup

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pdf_oxide.git
   cd pdf_oxide
   ```

2. **Build the project**:
   ```bash
   cargo build
   ```

3. **Run tests**:
   ```bash
   cargo test
   ```

4. **Set up pre-commit hooks** (recommended):
   ```bash
   ./scripts/setup-hooks.sh
   ```

   This installs a pre-commit hook that automatically runs:
   - Code formatting (`cargo fmt --check`)
   - Linting (`cargo clippy`)
   - Build verification (`cargo check`)
   - Library tests (`cargo test --lib`)
   - Integration tests (`cargo test --tests`)
   - Documentation tests (`cargo test --doc`)

## Project Structure

See `docs/planning/README.md` for comprehensive documentation.

```
pdf_oxide/
├── src/              # Rust source code
├── tests/            # Integration tests
├── benches/          # Performance benchmarks
├── examples/         # Usage examples
├── python/           # Python bindings (PyO3)
├── docs/planning/    # Planning documents (16 files)
└── training/         # ML training scripts
```

## Development Workflow

### 1. Pick a Task

- Check [Issues](https://github.com/yfedoseev/pdf-library/issues)
- Look for issues labeled `help-wanted` or `good-first-issue`
- Comment on the issue to claim it

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test additions
- `refactor/` - Code refactoring

### 3. Make Changes

Write code following our [Coding Standards](#coding-standards).

### 4. Test Your Changes

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with features
cargo test --features ml

# Watch mode (auto-reload)
cargo watch -x test
```

### 5. Format and Lint

```bash
# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings

# Fix clippy suggestions
cargo clippy --fix
```

### 6. Commit Your Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add PDF object parser"
git commit -m "fix: correct unicode mapping in ToUnicode CMap"
git commit -m "docs: update API documentation"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance tasks

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Coding Standards

### Rust

#### Style

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` (configured in `rustfmt.toml`)
- Maximum line length: 100 characters
- Use 4 spaces for indentation

#### Naming Conventions

```rust
// Modules and crates
mod pdf_parser;

// Types (structs, enums, traits)
struct PdfDocument;
enum Object;
trait Decoder;

// Functions and methods
fn parse_object() -> Result<Object>;

// Constants
const MAX_RECURSION_DEPTH: usize = 100;

// Static variables
static GLOBAL_CONFIG: Config = Config::new();
```

#### Error Handling

```rust
// Use Result<T> for fallible operations
pub fn parse_pdf(path: &Path) -> Result<PdfDocument> {
    // Use ? operator for error propagation
    let file = File::open(path)?;

    // Provide context when wrapping errors
    let doc = parse_file(file)
        .map_err(|e| Error::Parse(format!("Failed to parse {}: {}", path.display(), e)))?;

    Ok(doc)
}

// Avoid unwrap() in library code (only in tests and examples)
// Use expect() with descriptive messages when appropriate
```

#### Documentation

```rust
/// Parse a PDF object from bytes.
///
/// # Arguments
///
/// * `bytes` - Raw bytes containing the PDF object
///
/// # Returns
///
/// Returns the parsed object or an error if parsing fails.
///
/// # Errors
///
/// Returns `Error::Parse` if the bytes don't represent a valid PDF object.
///
/// # Examples
///
/// ```
/// use pdf_oxide::parse_object;
///
/// let bytes = b"42";
/// let obj = parse_object(bytes)?;
/// assert_eq!(obj, Object::Integer(42));
/// ```
pub fn parse_object(bytes: &[u8]) -> Result<Object> {
    // Implementation
}
```

#### Safety

- Avoid `unsafe` unless absolutely necessary
- Document all `unsafe` blocks with safety invariants
- Prefer safe abstractions from the standard library

### Python

- Follow [PEP 8](https://pep8.org/)
- Use `black` for formatting
- Type hints for all public functions
- Docstrings in Google style

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_integer() {
        let obj = parse_object(b"42").unwrap();
        assert_eq!(obj, Object::Integer(42));
    }

    #[test]
    #[should_panic(expected = "Invalid")]
    fn test_invalid_input() {
        parse_object(b"invalid").unwrap();
    }
}
```

### Integration Tests

Located in `tests/`:

```rust
// tests/test_integration.rs
use pdf_oxide::PdfDocument;

#[test]
fn test_extract_text_from_simple_pdf() {
    let mut doc = PdfDocument::open("tests/fixtures/simple.pdf").unwrap();
    let text = doc.extract_text(0).unwrap();
    assert!(text.contains("Hello, World!"));
}
```

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_roundtrip(s: String) {
        let encoded = encode(&s);
        let decoded = decode(&encoded)?;
        prop_assert_eq!(s, decoded);
    }
}
```

### Coverage Goals

- **Library code**: 80%+ coverage
- **Critical paths**: 100% coverage (parsing, error handling)

Check coverage:
```bash
cargo tarpaulin --out Html
open tarpaulin-report.html
```

## Documentation

### Code Documentation

- All public items must have doc comments
- Include examples in doc comments
- Run `cargo doc` to check rendered docs

### Planning Documentation

- Update relevant `PHASE_*.md` files if modifying implementation
- Keep `README.md` up to date with new features
- Document breaking changes

### Examples

Add examples to `examples/`:

```rust
// examples/basic.rs
use pdf_oxide::PdfDocument;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let doc = PdfDocument::open("paper.pdf")?;
    let text = doc.extract_text(0)?;
    println!("{}", text);
    Ok(())
}
```

## Submitting Changes

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code compiles without warnings
- [ ] All tests pass (`cargo test`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] Clippy passes (`cargo clippy -- -D warnings`)
- [ ] New code has tests
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] PR description explains changes clearly

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issue
Fixes #(issue number)

## Testing
Describe testing done

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Documentation updated
```

### Review Process

1. Maintainers will review your PR
2. Address feedback and push updates
3. Once approved, your PR will be merged
4. Your changes will appear in the next release

## Performance Considerations

When working on performance-critical code:

1. **Benchmark before and after**:
   ```bash
   cargo bench
   ```

2. **Profile if needed**:
   ```bash
   cargo install flamegraph
   cargo flamegraph --bench my_bench
   ```

3. **Consider memory usage**:
   - Avoid unnecessary allocations
   - Use `Cow<str>` when appropriate
   - Stream large files instead of loading entirely

## ML Features

When working on ML features (OCR, layout analysis):

- Document model requirements
- Provide ONNX conversion scripts
- Test on CPU-only systems
- Keep models small (<50MB)
- Document accuracy metrics

## License

By contributing, you agree that your contributions will be dual licensed under **MIT OR Apache-2.0**, as defined in the Apache-2.0 license, without any additional terms or conditions.

This means:
- Your code will be available under permissive open-source licenses
- Users can choose either MIT or Apache-2.0 for their needs
- See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) for full terms

## Questions?

- Check `docs/spec/` for PDF specification references
- Read code comments and documentation
- Open an issue for questions
- Join discussions on GitHub Discussions

## Recognition

Contributors will be acknowledged in:
- GitHub contributors list
- Release notes
- `CONTRIBUTORS.md` file (coming soon)

Thank you for contributing! 🎉
