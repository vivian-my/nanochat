# WECO ClimbMix v18 — feature_construct + data-selection (gpt-5.5)

## Goal

Write `feature_specs()` and `select_docs(budget, seed)` together to pick
**880,000** documents from a **4,485,120-doc pool** that minimize the mean
`val_bpb` of a d8 model trained on the selected set.

The trainer averages **3 independent seeds** (~440 M tokens each, d8 FP8,
target_param_ratio = 10.5).

**Reference**: `random_uniform` (4-seed) val_bpb = 0.951452 ± 0.000529.
Target: beat random by **>0.001** (≥ 1.9σ); <5e-4 is within noise.

Budget: 880,000 / 4,485,120 = **19.6 %** keep rate.

---

## NEW: Feature construction interface

The selector script defines TWO functions. The harness calls `feature_specs()`
first to discover feature definitions, computes them (caching lexicals), then
calls `select_docs(budget, seed)` with the feature dicts available as module
globals.

```python
def feature_specs():
    """Returns (composites, lexicals).

    composites: list[(name, fn(BASE) -> ndarray of shape (N_TOTAL,))]
        Pure numpy on the 9 base features. Cheap, recomputed every trial.

    lexicals:   list[(name, fn(text: str) -> float)]
        Per-doc functions over RAW TEXT. Computed once across all 4.485 M
        docs and cached on the volume by name + sha256(source). MAX 3 per
        trial (overflow silently dropped). Make them deterministic.
    """
    # composite — combines three Gemini-rubric counts already in BASE
    def anno_quality(B):
        return B["n_rsteps"].astype(np.float32) - B["n_rerrors"].astype(np.float32) - B["n_factual"].astype(np.float32)

    # lexical — runs over raw text; cached on the volume after first compute
    import re
    URL_RE = re.compile(r"https?://\S+")
    def url_density(text: str) -> float:
        if not text:
            return 0.0
        return len(URL_RE.findall(text)) / max(len(text.split()), 1)

    composites = [("anno_quality", anno_quality)]
    lexicals = [("url_density", url_density)]
    return composites, lexicals


def select_docs(budget, seed):
    # BASE / COMPOSITES / LEXICALS are module globals populated by the harness
    # before this runs. Each is dict[str, ndarray of shape (N_TOTAL,)].
    rng = np.random.default_rng(seed)
    score = COMPOSITES["anno_quality"] - 2.0 * LEXICALS["url_density"]
    # ... combine signals, return budget unique int64 indices ...
```

**Allowed inside lexical functions:** `re`, `string`, `math`, `itertools`,
`collections`, `unicodedata`, `numpy`. Plus pure Python. Each lexical fn must
be **self-contained** (cannot reference outer-scope helpers from
`feature_specs`); the harness only sees `inspect.getsource(fn)`.

**Forbidden:** `os`, `subprocess`, `urllib`, `socket`, `requests`, `httpx`,
file I/O, network, randomness. Lexical fns must be deterministic and
`O(len(doc))`.

**Cache semantics:** if the LLM rewrites a lexical fn but keeps the same
name, the source-hash changes and the harness recomputes. If the name is
new, a fresh cache entry is created. Cache files live on the Modal volume
under `/pool/meta_dynamic/<name>__<hash>.npy` and persist across trials.

**Each lexical's first-time compute:** ~30 s – 5 min depending on cost
(single-process loop over all 4.485 M docs). Re-using a cached lexical adds
zero overhead. So expect the **first** trial that introduces a new lexical
to be slower; subsequent trials referencing the same `(name, source)` are
free.

---

## Example documents from the pool (climbmix is web text)

Five randomly sampled documents (full text), one per topic and format, to
ground feature ideas in what the corpus actually looks like:

```
[topic=Hardware, format=Q&A Forum, doc_chars=2924]
So I mentioned in my first thread that if I was able to find a way to safely remove one of the switches, that I would take some photos. Well as you can probably assume, I managed to find a way. Basically its just a matter of prying off the flattened plastic on the other side of the plate. The switch still stays in the plate, but you can now pop the switch off with a little force. SO here it is:
So yeah! Thats about it. Not all that exciting, but basically the large spring is the main source of the "weight" in the switch, as well as what returns it to its resting position. I mentioned in the other thread, that it felt like towards the middle of a keypress, that it all the sudden got stiffer, as if something was pushing against you. Well now we know why. The stem has a second mini spring underneath it, that seems to be the spring that contacts the membrane. Without that second spring, the switch presses a lot lighter, and actually doesn't feel that bad, but the second spring makes the typing experience quite unpleasant. I have more photos in the album here and again use them where ever you please.
At first glance, the legs look like those of a Cherry MY, but they're not the same.
I mistook my OK-100M for linear, so I got a real shock when I bought a Cherry MX Red keyboard, and found that it felt completely different, despite being very similar in weight.
The reason? The same thing: external and internal springs. I'm using "progressive rate" to refer to such designs (for now, until someone proves that's abuse of the term), and it's very reassuring to know that you actually perceive it that clearly, because I can categorically state that Oak FTM is progressive rate.
Muirium: it's actually quite fun to type on light progressive rate switches. It's got a nice cushioning feel. The chief reason I don't (above the terrible layout and the awful variable weighting) is that the OK-100M is complete trash and several keys get stuck. Those that do work feel nice. I think I would side with MX Red, as it's just such a nice switch.
Awesome! Yeah, I want to like it, but it just isn't pleasant. Reds really are excellent linear switches. Progressive rate sounds like a good term for them. Basically the only thing that I do like about the keyboard, is the really awesome keycaps (which appear to be made by SP, which is cool).
Also I assume you wouldn't have a problem with me going and adding some of my images to their respective wiki pages? I still need to take the time to figure it out, but that would be okay right?

Question: What is the term used to describe switches with both external and internal springs? Answer: Progressive rate.

Question: What is the user's opinion about the Oak FTM switches? Answer: They find them fun to type on due to their light progressive rate, but the keyboard has other issues.

Question: Which switch does the user prefer between MX Red and Oak FTM? Answer: MX Red.
```

```
[topic=Home & Hobbies, format=Product Page, doc_chars=577]
CURCUMA

Share this product

Turmeric is an easy-to-install tropical plant in a heated interior. It forms beautiful and colorful flowers and combines with other cold plants.

Turmeric is a perennial tropical plant with short stems and rhizomes. Its roots are yellow or orange and its flowers are bright pink. Its foliage made of large leaves is distributed on both sides of the stem. Turmeric flowers are sterile and it is the rhizomes that can allow plant reproduction. Our practical tips will help novice and experienced gardeners cultivate turmeric successfully in the world
```

```
[topic=Software Dev., format=Listicle, doc_chars=3726]
In contrast, AI-driven platforms are built with user-friendliness in mind. They often come with drag-and-drop features, template libraries, and real-time suggestions, making the design process smooth and approachable for beginners.
Beyond simplicity, these tools are equipped with advanced algorithms that can analyze and understand the essence of good design. For instance, if a user is trying to create a poster, the AI can suggest optimal font pairings, color palettes, and layout structures based on the content and intended audience.
This ensures that the final product is not only aesthetically pleasing but also effective in conveying the intended message.
Another groundbreaking feature is the AI's ability to adapt and learn from user preferences. Over time, as users interact with the tool, it begins to understand their unique style and preferences, offering personalized suggestions and templates. This tailored approach ensures that each design feels authentic and resonates with the creator's vision.
Furthermore, AI-driven design tools often come with vast libraries of stock images, icons, and graphics. This means users don't have to scour the internet for resources; everything they need is integrated into the platform. And with AI's smart cropping and resizing capabilities, these assets can be effortlessly adapted to fit various formats and platforms.
Best AI Apps for Visual Design
No longer is the ability to create stunning visuals reserved for those with formal training or access to expensive software. With AI-driven tools, everyone, from entrepreneurs to educators to hobbyists, can harness the power of design to communicate, inspire, and captivate their audience.
Here are three pieces of software that can help you get your foot in the door:
Galileo: An AI-driven tool that assists in creating infographics, presentations, and other visual content.
5. Write Code with AI
The world of software development, once seen as a domain reserved for those with specialized training, is undergoing a big shift with the introduction of AI-driven platforms.
At the heart of this transformation is the AI's ability to understand programming languages, frameworks, and best practices. When developers begin to write code, these platforms can provide real-time suggestions, much like how a grammar checker works for written content.
Debugging, a task often dreaded by developers due to its time-consuming nature, is also being revolutionized by AI. Instead of manually sifting through lines of code to identify errors, AI-driven tools can quickly pinpoint issues, offer insights into the root cause, and even suggest potential fixes.
This drastically reduces the time and frustration associated with troubleshooting.
Optimization is another area where AI shines. As developers know, writing code that works is one thing, but writing code that's efficient and optimized is another challenge altogether. AI platforms can analyze code to identify bottlenecks or inefficiencies and then recommend improvements, ensuring that the final product runs smoothly and efficiently.
For those new to coding, AI-driven platforms often come with interactive tutorials and guides. These are not static lessons but dynamic learning experiences that adapt based on the learner's progress and understanding.
This personalized approach makes the learning curve less daunting and encourages more individuals to delve into the world of coding.
Best AI Apps to Help You Write Code

Question: What can AI-driven design tools do to ensure the final product is effective in conveying the intended message?
Answer: They can suggest optimal font pairings, color palettes, and layout structures based on the content and intended audience.
```

```
[topic=Sports & Fitness, format=Tutorial, doc_chars=885]
Tags
Supine Spinal Twist (Supta Matsyendrasana) Yoga Pose
Supine spinal Twist pose encourages the movement as well as the mobility in your spine. Generally Twists are the best for increasing flexibility in your spine. This will be helpful in alleviating stiffness in your lower back and hips; tone your abdomen and improve you digestive system. Supta Matsyendrasana can also help in relieving and toning your abdomen.
Steps to Practice Supine Spinal Twist Pose
Start with lying down, hold your knees and inhale.
By the use of your left hand, drop down your knees to the left side as you exhale.
After that stretch your arm to rightward and turn your head.
Maintain this position for 5-10 breaths.
Now come to your relaxed position and repeat again with the other side.

Question: Which side should you repeat the pose on after returning to the relaxed position? Answer: The other side.
```

```
[topic=Social Life, format=Personal Blog, doc_chars=285]
New Year resolutions are time waste. So, don't stick to it. If you need New Year to start doing something then most probably it's not your true goal. It's really hard to do long time something that you don't believe in. Select real goals and you will noy need any New Year resolutions.
```

After reading the corpus, propose features that capture **what's
distinguishing** about climbmix text — e.g., presence of templated/listicle
structure, "Tags" / "Question:" headers, Q&A or boilerplate ("Share this
product"), commerce signals, typo/grammar quality, code blocks, citation
markers, list bullets, repeated section markers, etc.

---

## Single-feature ablation grid (priors)

Each row is a 3-seed d8 trained on a single-feature filter applied to the
same pool, with the same training config as this run.

| trial | rule | val_bpb (3-seed) | Δ vs random | sigma |
|---|---|---:|---:|---:|
| `ppl_d8_highest` | TOP 19.6 % `logppl_d8_ref` | 1.0252 | +0.074 | **−139σ** |
| `ngram_top` | TOP 19.6 % `avg_distinct` | 1.0144 | +0.063 | −119σ |
| `ppl_qwen_highest` | TOP 19.6 % `logppl_qwen` | 1.0068 | +0.055 | −105σ |
| `ppl_d8_lowest` | BOT 19.6 % `logppl_d8_ref` | 0.9863 | +0.035 | −66σ |
| `ppl_qwen_lowest` | BOT 19.6 % `logppl_qwen` | 0.9848 | +0.033 | −63σ |
| `ngram_bot` | BOT 19.6 % `avg_distinct` | 0.9676 | +0.016 | −30σ |
| `length_long` | TOP 19.6 % `doc_tokens` | 0.9605 | +0.009 | −17σ |
| `central_len` | central 19.6 % `doc_tokens` | 0.9619 | +0.010 | −20σ |
| `length_short` | BOT 19.6 % `doc_tokens` (shortest) | **3.51** | +2.55 | training collapse — too few tokens |
| `rstep_gt1` | `n_rsteps > 1` + random fill | 0.9537 | +0.0022 | −4.2σ |
| `fact_zero` | `n_factual == 0` + random fill | 0.9528 | +0.0013 | −2.5σ |
| `rerr_zero` | `n_rerrors == 0` + random fill | 0.9517 | +0.0002 | **−0.4σ (≈ random)** |
| `topic_quota` | per-topic proportional, random within | 0.9516 | +0.0001 | **−0.2σ (≈ random)** |
| **`random_uniform`** | (baseline) | **0.9515** | **0** | **0** |

---

## The pool — ClimbMix (pre-curated)

Pre-filtered web corpus (quality classification, dedup, diversity balancing).
Pool distribution (ok docs only, 53 shards):

- **Topics** — 24 classes. Science & Tech 24 %, Health 15 %, Home & Hobbies 12 %.
- **Formats** — 24 classes. Knowledge Article 18 %, Tutorial 15 %, Product Page 9 %.
- 4,483,659 ok / 4,485,120 total (1,461 docs have −1 annotations).

---

## Base feature bank (the 9 features available in BASE)

All arrays at `/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_53shards/`,
indexed by global doc id `g ∈ [0, 4_485_120)`. Read-only.

| key in BASE | dtype | Description |
|---|---|---|
| `topic_id` | int32 | WebOrganizer topic id, 0..23 |
| `format_id` | int32 | WebOrganizer format id, 0..23 |
| `n_rsteps` | int16 | reasoning steps (Gemini rubric); −1 if annotation failed |
| `n_rerrors` | int16 | invalid reasoning steps (⊆ `n_rsteps`); −1 if failed |
| `n_factual` | int16 | factual errors; −1 if failed |
| `logppl_qwen` | float32 | Qwen-2.5-0.5B mean per-token NLL |
| `logppl_d8_ref` | float32 | d8 (held-out shards 60..559) mean per-token NLL |
| `avg_distinct_ngram_bpe` | float32 | Σ_{n=1..5} distinct_n (range ≈ [0, 5]) |
| `doc_tokens` | int32 | nanochat-BPE doc length |

Plus utility keys also in BASE: `doc_chars` (int32),
`distinct_{1,2,3,4,5}gram_bpe` (float32), `ok` (bool).

### Feature definitions

#### topic_id — WebOrganizer topic classifier

| id | name | id | name | id | name |
|---:|---|---:|---|---:|---|
| 0 | Adult | 1 | Art & Design | 2 | Software Dev. |
| 3 | Crime & Law | 4 | Education & Jobs | 5 | Hardware |
| 6 | Entertainment | 7 | Social Life | 8 | Fashion & Beauty |
| 9 | Finance & Business | 10 | Food & Dining | 11 | Games |
| 12 | Health | 13 | History | 14 | Home & Hobbies |
| 15 | Industrial | 16 | Literature | 17 | Politics |
| 18 | Religion | 19 | Science & Tech. | 20 | Software |
| 21 | Sports & Fitness | 22 | Transportation | 23 | Travel |

#### format_id — WebOrganizer format classifier

| id | name | id | name | id | name |
|---:|---|---:|---|---:|---|
| 0 | Academic Writing | 1 | Content Listing | 2 | Creative Writing |
| 3 | Customer Support | 4 | Comment Section | 5 | FAQ |
| 6 | Truncated | 7 | Knowledge Article | 8 | Legal Notices |
| 9 | Listicle | 10 | News Article | 11 | Nonfiction Writing |
| 12 | About (Org.) | 13 | News (Org.) | 14 | About (Pers.) |
| 15 | Personal Blog | 16 | Product Page | 17 | Q&A Forum |
| 18 | Spam / Ads | 19 | Structured Data | 20 | Documentation |
| 21 | Audio Transcript | 22 | Tutorial | 23 | User Review |

#### n_rsteps — reasoning steps
Gemini rubric. Pool: μ=1.81, p95=4, max=36, zero-rate 4 %.

#### n_rerrors — invalid reasoning steps
By construction `n_rerrors ≤ n_rsteps`. Pool: μ=0.27, p95=1, max=20, zero-rate 77 %.

#### n_factual — factual errors
Verifiable mistakes. Pool: μ=0.35, p95=2, max=40, zero-rate 72 %.

#### logppl_qwen — out-of-distribution LM
Qwen-2.5-0.5B mean per-token NLL. Pool: μ=2.732, σ=0.663, p[1,25,50,75,95,99]=[1.29, 2.35, 2.70, 3.08, 3.80, 4.61].

#### logppl_d8_ref — in-distribution LM
nanochat d8 FP8 trained on shards 60..559 (held out). Pool: μ=3.674, σ=1.122, p[1,25,50,75,95,99]=[1.75, 3.07, 3.52, 4.06, 5.54, 7.46].

#### avg_distinct_ngram_bpe — diversity
Σ_{n=1..5} distinct_n. Range ≈ [0, 5]. Pool: μ=4.258, σ=0.327, p[1,25,50,75,95,99]=[3.18, 4.10, 4.28, 4.46, 4.73, 4.89].

#### doc_tokens — nanochat-BPE length
Pool: mean 545, p[1,50,95,99]=[42, 527, 1447, 1807]. Total ≈ 2.44 B for ok docs.

---

## Known correlations (Spearman ρ on 500K-doc sample)

| pair | ρ |
|---|---:|
| `logppl_qwen` ↔ `logppl_d8_ref` | +0.88 |
| `avg_distinct` ↔ `distinct_1g/2g/3g` | +0.90/0.98/0.92 |
| `doc_tokens` ↔ `doc_chars` | +0.98 |
| `logppl_d8_ref` ↔ `doc_tokens` | **−0.60** (length confound) |
| `avg_distinct` ↔ `doc_tokens` | **−0.66** (length confound) |
| `logppl_d8_ref` ↔ `avg_distinct` | +0.67 |
| `n_rsteps` ↔ `doc_tokens` | +0.48 |
| `n_rerrors` ↔ `n_factual` | +0.21 |
| `n_rsteps` ↔ `n_rerrors` | +0.07 |
| `n_rsteps` ↔ `n_factual` | −0.01 |

Effectively-independent continuous signals:
1 PPL · 1 diversity · `doc_tokens` · `n_rsteps` · `n_rerrors` · `n_factual`.
Plus `topic_id`, `format_id` as categorical moderators.

---

## Contract

```python
def feature_specs():
    return composites, lexicals   # types described above; lexicals capped at 3

def select_docs(budget, seed) -> np.ndarray:
    return np.ndarray of shape (BUDGET,) dtype int64, unique values in [0, 4_485_120)
```

BUDGET is fixed at 880,000. Any other shape, duplicates, or out-of-range
indices → `val_bpb = 9.9999` (penalty). A composite or lexical that
crashes is silently skipped (logged); the rest still compute.

**Constraints:**
- No O(N²) ops over 4.5 M docs in either function.
- No external network, no downloads, no file I/O outside the harness-managed paths.
- numpy + stdlib only (allowed: `re`, `string`, `math`, `itertools`, `collections`, `unicodedata`).

---

## Feedback after each trial

| metric | meaning |
|---|---|
| `val_bpb` | mean over 3 seeds (primary signal, lower = better) |
| `val_bpb_std` | 3-seed std (want this small) |
| per-seed bpb | for variance diagnosis |
| composites_used / lexicals_used | which derived features actually got built |
| lexical_compute_min | wallclock cost of the new-lexical compute pass |
| `n_docs` | should equal 880,000 |
