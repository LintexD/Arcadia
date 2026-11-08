# Rust + wgpu Game Engine Guidelines
#### Versions: Rust: 1.99.1   wgpu: 23.0.0
## 1. Crate Version Lock (CRITICAL)
 Always check `@Cargo.toml` before writing `wgpu` or `winit` code.
- Do NOT mix legacy `wgpu` patterns (e.g., old `SwapChain` or outdated `RenderPassColorAttachment` fields) with our pinned version.

## 2. Navigation & Verification Loop
- Prefer `rust-analyzer-lsp` MCP tools for finding trait implementations, symbol references, and type definitions over `grep_search`.
- For code checks, run `cargo check --message-format=short` (not `cargo build`) to minimize output tokens and compile time.
- Every `.wgsl` change MUST be validated with `naga <path_to_shader.wgsl>`.
- Do NOT introduce `unsafe` blocks without explicit approval.

## Using mcpls for wgpu API Discovery
- You have `mcpls` connected (`get_hover`, `get_definition`, `get_references`, `get_diagnostics`).
- Never guess `wgpu 23.0` struct fields from memory. Call `get_definition` or `get_hover` on an existing `wgpu::` symbol in the file to inspect the exact struct definition in `~/.cargo/registry/src/`.
- If you need to look up a `wgpu` type that isn't imported yet, temporarily reference `wgpu::<TypeName>` in the file and run `get_hover` / `get_definition` on it.

## Test Execution Rules (STRICT)
- **NEVER run `cargo test`** for unit or integration tests. Always use `cargo nextest run`.
- To run a single test or shader check, use nextest filter expressions:
  `cargo nextest run -E 'test(validate_all_wgsl_shaders)'`
- Only use `cargo test --doc` if you specifically need to verify documentation tests (since nextest does not execute doctests).