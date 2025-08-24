



## Attention Mechanisms — Interactive Visualization


[Checkout Demo Here](https://mehdihosseinimoghadam.github.io/Attention/attention_explained.html)

Interactive, self‑contained page to learn and demo transformer attention. It walks through the full attention pipeline step by step and supports multiple attention patterns, including multi‑head attention.

Open the demo:

- Double‑click `attention_explained.html` or serve it locally with any static file server.

### Highlights

- Step‑by‑step pipeline: Tokens → Q/K/V → Scores → Scale → Softmax → Output
- Multiple attention types: Full, Sliding Window, Dilated, Global+Sliding, Multi‑Head
- Live controls for attention type, query token, window size, number of heads
- Matrix heatmaps with tooltips, row highlighting, axis labels, and masking
- Keyboard navigation and auto‑play

---

## Quick Start

1. Open `attention_explained.html` in a modern browser.
2. Use the toolbar to change:
   - Attention Type: Full, Sliding, Dilated, Global+Sliding, Multi‑Head
   - Query token: Which row is emphasized in matrices
   - Window size: Local neighborhood for sliding patterns (and receptive masks)
   - Heads: Number of attention heads (only visible for Multi‑Head)
3. Navigate steps with Next/Back, Left/Right arrows, or Play.

Tip: Masked cells (not attendable) show an `×` and render dimmer. Hover any cell for its exact value.

---

## What You’re Seeing

### Example Sentence (Tokens)

The default example is a 21‑token sentence (plus `[CLS]`), designed to create interesting attention patterns:

“The lazy dog wore sunglasses, stole my sandwich, barked at the fridge, then blamed the confused cat for everything suspiciously.”

### Dimensions

- d_model = 8, d_k = 8, d_v = 8
- For Multi‑Head with H heads, each head dimension is d_head = d_k / H

### Matrices and Shapes

- Embeddings `E`: shape (n_tokens × d_model)
- Projections: `Q = E · W_Q`, `K = E · W_K`, `V = E · W_V`
  - `W_Q, W_K, W_V` are (d_model × d_k / d_v)
- Scores: `scores = Q · K^T` → (n_tokens × n_tokens)
- Scale: `scaled = scores / sqrt(d_k)` (or `/ sqrt(d_head)` per head)
- Softmax (row‑wise): `weights = softmax(scaled)`
- Output: `O = weights · V` → (n_tokens × d_v)

### Attention Types

- Full Self‑Attention: all tokens can attend to all tokens
- Sliding Window: each token attends to neighbors within `±window` positions
- Dilated Sliding Window: sliding window plus periodic “skip” links for longer reach
- Global + Sliding: `[CLS]` is global (attends to all and vice versa), others are local windowed
- Multi‑Head Attention: splits `Q/K/V` into heads, runs attention per head in parallel, then concatenates head outputs

Masking rules are visualized directly in the heatmaps; masked cells show `×` and are excluded (set to −∞ before softmax).

---

## Controls & Shortcuts

- Attention Type: switch among the patterns above
- Query token: highlights the selected row across matrices
- Window size: local neighborhood for sliding/dilated/global+sliding
- Heads: 1/2/4/8 (only for Multi‑Head; must divide d_k)
- Buttons: Back, Next, Play/Pause, Reset
- Keys: Left/Right arrows to step

---

## How Multi‑Head Is Shown

When Multi‑Head is selected:

- Steps 3–5 display a grid of per‑head matrices:
  - Raw Scores per head (Q_h · K_h^T)
  - Scaled Scores per head (divide by √d_head)
  - Softmax Weights per head
- Step 6 shows per‑head attention heatmaps and the concatenated output `concat(O_1, …, O_H)` with shape (n_tokens × d_model).

This highlights that different heads often capture different relationships (syntax, semantics, long‑range links, etc.).

---

## Code Map (inside the HTML)

- Data & Config
  - `tokens`, `dModel`, `dK`, `dV`
  - Fixed `E`, `WQ`, `WK`, `WV` for a deterministic demo
- Math Helpers
  - `matMul`, `transpose`, `scale`, `rowSoftmax`
- Masking
  - `createAttentionMask(type, seqLen, windowSize)` implements Full, Sliding, Dilated, Global+Sliding
  - `applyMask(scores, mask)` replaces disallowed positions with −∞ before softmax
- Single‑Head Pipeline
  - Precompute `Q`, `K`, `V`, base `scores` and `scaled`
  - `computeAttention(type, window)` applies mask → softmax → output
- Multi‑Head Pipeline
  - `splitHeads(Q|K|V, numHeads, headDim)`
  - Per‑head: `scores_h → scaled_h → softmax → O_h`
  - `combineHeads([O_h])` concatenates along the feature axis
  - `computeMultiHeadAttention(type, window, H)` drives the per‑head flow
- Rendering
  - `renderMatrix` paints labeled matrices with axis headers and tooltips
  - `renderForStep` switches content by current step and attention type
  - `updateSelectors` wires controls; `update` re‑renders the view

---

## Customize the Demo

- Change the sentence: edit `const tokens` near the top of the script
- Adjust model sizes: edit `dModel`, `dK`, `dV` (ensure `dK % numHeads === 0`)
- Tweak visuals: CSS variables at the top (e.g., `--cell`, `--gap`, colors)
- Start step or defaults: edit the `state` object (step index, window, heads)

Note: Matrices use handcrafted numbers for clarity and stable visuals; they’re not trained weights.

---

## Troubleshooting

- Heads selector missing: Only appears for Multi‑Head type
- Blank matrices in Multi‑Head: ensure `dK % numHeads === 0` and try Reset
- Long token labels clipped: widen the row label column by adjusting `.matrix .table { grid-template-columns: 120px 1fr; }`

---

## License

MIT. Feel free to reuse and adapt for teaching, workshops, or demos.


