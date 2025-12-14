# PDF Library - Development Guide

This comprehensive guide documents the development workflow, tools, and best practices for the PDF Library project.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Development Tools](#development-tools)
4. [Claude Code Integration](#claude-code-integration)
5. [Development Workflow](#development-workflow)
6. [Testing Strategy](#testing-strategy)
7. [Code Quality](#code-quality)
8. [Performance Guidelines](#performance-guidelines)
9. [Documentation Standards](#documentation-standards)
10. [Git Workflow](#git-workflow)

## Quick Start

### Prerequisites

```bash
# Install Rust (stable 1.70+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install development tools
cargo install cargo-tarpaulin  # Coverage
cargo install cargo-audit      # Security auditing
cargo install cargo-deny       # License checking
cargo install flamegraph       # Profiling
cargo install criterion        # Benchmarking
```

### First Time Setup

```bash
# Clone and setup
git clone <repository-url>
cd pdf_oxide

# Verify setup
cargo check --all-features
cargo test
cargo clippy --all-targets
```

### Daily Development

```bash
# Quick checks before committing
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test --all-features
```

## Project Structure

```
pdf_oxide/
├── .claude/                    # Claude Code configuration
│   ├── hooks.json             # Automated validation hooks
│   ├── scripts/               # Hook scripts
│   ├── commands/              # Custom slash commands
│   ├── skills/                # Claude Code skills
│   ├── guides/                # Development guides
│   ├── checklists/            # Quality checklists
│   └── templates/             # Code templates
│
├── .github/
│   └── workflows/             # CI/CD pipelines
│       ├── ci.yml            # Continuous integration
│       └── release.yml       # Release automation
│
├── docs/
│   ├── spec/                 # PDF specification reference
│   │   └── pdf.md            # ISO 32000-1:2008 excerpts
│   ├── ARCHITECTURE.md       # System design
│   ├── DEVELOPMENT_GUIDE.md  # This file
│   ├── MARKDOWN_CONVERTER_USAGE.md # Markdown export guide
│   └── ML_INTEGRATION.md     # OCR and ML features
│
├── src/
│   ├── lib.rs                # Library entry point
│   ├── object.rs             # PDF object types
│   ├── parser.rs             # PDF parser
│   ├── layout/               # Layout analysis
│   ├── converters/           # Export formats
│   └── ...
│
├── tests/
│   ├── fixtures/             # Test PDF files
│   └── *.rs                  # Integration tests
│
├── benches/                  # Performance benchmarks
├── examples/                 # Usage examples
├── python/                   # Python bindings
│
├── Cargo.toml                # Project configuration
├── CLAUDE.md                 # Claude Code context
├── CONTRIBUTING.md           # Contribution guide
└── README.md                 # Project overview
```

## Development Tools

### Automated Validation (Hooks)

The project uses Claude Code hooks for automatic validation:

**PostToolUse Hook** (after editing files):
- Auto-formats Rust code
- Quick syntax check
- Runs in 2-5 seconds

**SubagentStop Hook** (when subagent finishes):
- Format verification
- Clippy lints
- Compilation check
- Runs in 5-10 seconds

**Stop Hook** (when main agent finishes):
- Comprehensive checks
- Full test suite
- Runs in 10-40 seconds

See `.claude/hooks.json` for hook configuration.

### Code Quality Tools

**Formatting**:
```bash
cargo fmt              # Auto-format
cargo fmt --check      # Check only
```

**Linting**:
```bash
cargo clippy --all-targets -- -D warnings
cargo clippy -- -W clippy::pedantic  # Strict mode
```

**Security**:
```bash
cargo audit            # Check for vulnerabilities
cargo deny check       # License compliance
```

**Coverage**:
```bash
cargo tarpaulin --out Html --output-dir target/coverage
# Open target/coverage/index.html
```

**Profiling**:
```bash
cargo flamegraph --bench benchmark_name
# Open flamegraph.svg
```

### Editor Setup

**VS Code** (recommended):
- Settings in `.vscode/settings.json`
- Launch configurations in `.vscode/launch.json`
- Extensions in `.vscode/extensions.json`

**rust-analyzer** settings:
- Format on save enabled
- Clippy lints enabled
- Inlay hints configured

## Claude Code Integration

### Project Context

**CLAUDE.md**: Main project configuration
- Coding standards
- Architecture guidelines
- Testing requirements
- Performance targets
- Security guidelines

Read by Claude Code automatically to understand project context.

### Custom Slash Commands

Quick access to common tasks:

```bash
/review              # Run code review checklist
/test <module>       # Test specific module
/bench <name>        # Run benchmarks
/lint                # Run all linters
/doc [module]        # Generate/check docs
/check-features      # Test feature combinations
```

See individual command files in `.claude/commands/` for details.

### Skills (Autonomous Assistance)

Claude Code automatically activates these skills when appropriate:

- **rust-best-practices**: Ensures code quality
- **pdf-spec-expert**: PDF specification knowledge
- **algorithm-implementer**: Complex algorithm implementation
- **test-writer**: Test generation and coverage
- **performance-optimizer**: Performance optimization

Skills are invoked automatically based on context, not manually called.

### Workflow Guides

Detailed guides in `.claude/guides/`:
- `implementation_workflow.md` - Step-by-step implementation guide

### Checklists

Quality assurance checklists in `.claude/checklists/`:
- `code_review.md` - Comprehensive review checklist

### Templates

Code templates in `.claude/templates/`:
- `task_breakdown.md` - Break down large tasks
- `test_template.md` - Test code templates

## Development Workflow

### Starting a New Feature

1. **Check GitHub Issues**:
   - Browse open issues for areas needing help
   - Look for `help-wanted` or `good-first-issue` labels

2. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Break down tasks**:
   - Create TodoList for tracking
   - Reference relevant code sections
   - Check ARCHITECTURE.md for module organization

### Implementing a Feature

**TDD Cycle**:

1. **Define API** (types, function signatures)
2. **Write tests** (define expected behavior)
3. **Implement** (make tests pass)
4. **Refactor** (improve code quality)
5. **Document** (add doc comments)
6. **Benchmark** (if hot path)

**Example**:

```rust
// Step 1: Define API
pub fn extract_text(page: &Page) -> Result<String, Error> {
    todo!()
}

// Step 2: Write test
#[test]
fn test_extract_text() {
    let page = create_test_page();
    let text = extract_text(&page).unwrap();
    assert_eq!(text, "expected content");
}

// Step 3: Implement
pub fn extract_text(page: &Page) -> Result<String, Error> {
    // Implementation
    Ok(String::new())
}

// Step 4: Refactor as needed

// Step 5: Document
/// Extracts text from a PDF page...
pub fn extract_text(page: &Page) -> Result<String, Error> {
    // ...
}

// Step 6: Benchmark (if needed)
```

### Before Committing

Run automated checks:

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test --all-features
```

Or use: `/review`

### Committing Changes

```bash
git add <files>
git commit -m "feat: add feature X

Detailed description of changes.

Resolves #123"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring
- `test:` - Add tests
- `docs:` - Documentation
- `chore:` - Maintenance

## Testing Strategy

### Test Coverage Requirements

- **Overall**: ≥80% line coverage
- **Core modules**: ≥90% coverage
- **Public APIs**: 100% tested

### Test Types

**Unit Tests** (`#[cfg(test)] mod tests`):
```rust
#[test]
fn test_parse_integer() {
    let input = b"42";
    let result = parse_integer(input).unwrap();
    assert_eq!(result, 42);
}
```

**Integration Tests** (`tests/` directory):
```rust
#[test]
fn test_full_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let doc = PdfDocument::open("tests/fixtures/sample.pdf")?;
    let text = doc.extract_text(0)?;
    assert!(!text.is_empty());
    Ok(())
}
```

**Property-Based Tests** (proptest):
```rust
proptest! {
    #[test]
    fn never_panics(input in any_input()) {
        let _ = function_under_test(input);
    }
}
```

**Benchmarks** (criterion):
```rust
fn bench_parsing(c: &mut Criterion) {
    c.bench_function("parse", |b| {
        b.iter(|| parse(black_box(&data)))
    });
}
```

### Test Organization

```rust
#[cfg(test)]
mod tests {
    mod unit {
        mod parsing { /* ... */ }
        mod validation { /* ... */ }
    }

    mod edge_cases { /* ... */ }

    mod properties {
        use proptest::prelude::*;
        // Property tests
    }
}
```

### Running Tests

```bash
# All tests
cargo test --all-features

# Specific module
cargo test object

# Single test
cargo test test_parse_integer

# With output
cargo test -- --nocapture

# Coverage
cargo tarpaulin
```

## Code Quality

### Rust Standards

**Follow Rust API Guidelines**: https://rust-lang.github.io/api-guidelines/

**Key principles**:
- Use `Result<T, Error>` for fallible operations
- Avoid panics in library code
- Prefer borrowing over ownership
- Document all public APIs
- Use type system for safety

**Complexity limits**:
- Functions: ≤150 lines
- Cognitive complexity: ≤25
- Function parameters: ≤7

### Error Handling

```rust
// GOOD
pub fn parse(&self) -> Result<Object, Error> {
    let token = self.next_token()
        .ok_or_else(|| Error::Parse("unexpected EOF".into()))?;
    // ...
}

// BAD
pub fn parse(&self) -> Object {
    let token = self.next_token().unwrap();  // Never!
    // ...
}
```

### Documentation

All public APIs require:
- Summary description
- Parameter documentation
- Return value description
- Error conditions
- Examples (for non-trivial functions)

```rust
/// Extracts text from a PDF page.
///
/// # Arguments
///
/// * `page_num` - Zero-based page index
///
/// # Returns
///
/// Extracted text with preserved reading order.
///
/// # Errors
///
/// - `Error::InvalidPdf` if page doesn't exist
///
/// # Examples
///
/// ```
/// # use pdf_oxide::PdfDocument;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut doc = PdfDocument::open("sample.pdf")?;
/// let text = doc.extract_text(0)?;
/// # Ok(())
/// # }
/// ```
pub fn extract_text(&mut self, page_num: u32) -> Result<String, Error> {
    // ...
}
```

### Code Review

Before submitting PR, check:

1. **Automated checks pass**
   - Format, lint, compile, test

2. **API design**
   - Clear naming
   - Proper error handling
   - Minimal lifetimes

3. **Implementation**
   - No unsafe (or documented)
   - Edge cases handled
   - Performance acceptable

4. **Testing**
   - All paths tested
   - Coverage ≥80%

5. **Documentation**
   - All public items documented
   - Examples compile

Use checklist: `.claude/checklists/code_review.md`

## Performance Guidelines

### Performance Targets

Performance goals for core components:

| Component | Target | Acceptable |
|-----------|--------|------------|
| **Lexer** | 100 MB/s | 50 MB/s |
| **DBSCAN (10k pts)** | 50ms | 100ms |
| **XY-Cut** | 30ms/page | 50ms/page |
| **Text extraction** | 50ms/page | 100ms/page |
| **OCR inference** | 800-1000ms/page | (A4, 300 DPI) |

### Optimization Strategy

1. **Measure first**: Profile before optimizing
2. **Focus on hotspots**: Only optimize slow code
3. **Benchmark**: Verify improvements
4. **Document**: Explain optimization rationale

### Common Optimizations

**Reduce allocations**:
```rust
// GOOD
let mut buffer = String::with_capacity(estimated_size);

// BAD
let mut buffer = String::new();  // Will reallocate
```

**Use references**:
```rust
// GOOD
pub fn process(&self, data: &[u8]) -> Result<Output, Error>

// BAD
pub fn process(&self, data: Vec<u8>) -> Result<Output, Error>
```

**Spatial indices**:
```rust
// GOOD: O(log n)
let rtree = RTree::bulk_load(points);
let neighbors = rtree.locate_in_envelope(&aabb);

// BAD: O(n²)
for p1 in points {
    for p2 in points {
        if distance(p1, p2) < epsilon { }
    }
}
```

### Profiling

```bash
# Flamegraph
cargo flamegraph --bench benchmark_name

# Benchmarks
cargo bench

# Memory profiling
valgrind --tool=massif target/release/binary
```

## Documentation Standards

### Code Documentation

**Required for all public items**:
- Summary line
- Detailed description (if complex)
- Parameter documentation
- Return value documentation
- Error documentation
- Panic documentation (if any)
- Safety comments (for unsafe)
- Examples (for non-trivial)

### Documentation Tests

Examples in doc comments are tested:

```rust
/// # Examples
///
/// ```
/// use pdf_oxide::PdfDocument;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let doc = PdfDocument::open("sample.pdf")?;
/// assert_eq!(doc.page_count(), 1);
/// # Ok(())
/// # }
/// ```
```

Run with: `cargo test --doc`

### Building Documentation

```bash
# Build docs
cargo doc --no-deps --all-features

# Open in browser
cargo doc --no-deps --all-features --open
```

## Git Workflow

### Branching Strategy

- `main` - Stable, always working
- `feature/...` - Feature branches
- `fix/...` - Bug fixes

### Commit Messages

Format:
```
type: brief description (≤72 chars)

Detailed explanation of changes.
Why the change was needed.

Resolves #123
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

### Pull Requests

**Before creating PR**:
- All tests pass
- Code formatted
- Clippy warnings fixed
- Documentation updated
- Self-reviewed

**PR description**:
```markdown
## What
Brief description

## Why
Reason for changes

## Testing
How tested

## Description
Brief description of the feature or fix being implemented.

## Checklist
- [x] Tests pass
- [x] Documentation updated
- [x] Self-reviewed
```

### CI Pipeline

**On every PR**:
- Format check
- Clippy lints
- Multi-platform build
- Full test suite
- Feature combinations
- Coverage report
- Security audit
- License check

**On main**:
- Release builds
- Documentation deployment
- Performance tracking

## Resources

### Documentation

- [CLAUDE.md](../CLAUDE.md) - Project context for Claude Code
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [ML_INTEGRATION.md](ML_INTEGRATION.md) - OCR and ML features guide
- [MARKDOWN_CONVERTER_USAGE.md](MARKDOWN_CONVERTER_USAGE.md) - Markdown export feature guide

### External References

- [Rust Book](https://doc.rust-lang.org/book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [PDF Specification](https://www.adobe.com/devnet/pdf/pdf_reference.html) (ISO 32000-1:2008)
- [Clippy Lints](https://rust-lang.github.io/rust-clippy/master/)

### Tools

- [rust-analyzer](https://rust-analyzer.github.io/) - IDE support
- [Criterion](https://github.com/bheisler/criterion.rs) - Benchmarking
- [Proptest](https://github.com/proptest-rs/proptest) - Property testing
- [Tarpaulin](https://github.com/xd009642/tarpaulin) - Code coverage

## Getting Help

### Within Claude Code

```bash
/help              # Claude Code help
/review            # Code review checklist
/test <module>     # Run tests for module
```

### Before Starting

1. Read ARCHITECTURE.md for module structure
2. Check CLAUDE.md for coding standards
3. Review similar implementations
4. Consult PDF specification (docs/spec/pdf.md)

### Troubleshooting

**Compilation errors**:
- Read error message carefully
- Check rust-analyzer suggestions
- Review Rust book chapter

**Test failures**:
- Run with `--nocapture` for output
- Add debug prints
- Use debugger if needed

**Performance issues**:
- Profile first
- Check algorithm complexity
- Review optimization guide

**Hooks blocking**:
- Read hook output
- Fix the specific issue
- Check `.claude/hooks.json` for hook configuration

## Best Practices Summary

### Development

1. **Understand the architecture first** (read ARCHITECTURE.md)
2. **Write tests before code** (TDD)
3. **Keep changes small** (easy to review)
4. **Run checks frequently** (catch issues early)
5. **Document as you go** (don't defer)

### Code Quality

1. **No unwrap in library code**
2. **Descriptive error messages**
3. **Handle all edge cases**
4. **Keep functions small** (<150 lines)
5. **Document public APIs**

### Testing

1. **Test all public functions**
2. **Cover edge cases**
3. **Property tests for parsers**
4. **Benchmark hot paths**
5. **Target 80%+ coverage**

### Performance

1. **Measure before optimizing**
2. **Focus on hotspots only**
3. **Use appropriate data structures**
4. **Minimize allocations**
5. **Benchmark improvements**

### Git

1. **Small, focused commits**
2. **Descriptive messages**
3. **One feature per PR**
4. **Pass CI before merging**
5. **Keep main green**

---

**Last Updated**: 2025-10-29
**Claude Code Version**: Latest
**Document Version**: 1.0
