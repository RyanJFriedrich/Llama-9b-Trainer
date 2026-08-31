# Meditations on Geometry-Shaped Soft Targets

*Companion notes to `Llama8bGeometry.md`. These are not instructions. They are
an attempt to write down what the geometry seems to be saying, and what it
might mean for teaching one model from another. Nothing here is locked in.
The numbers are measurements; everything built on them is interpretation.*

---

## 1. The shape of an answer

Thirty-nine million positions is enough to stop arguing about whether the
predictive distribution has a shape. It does, and it is not the shape one
might have guessed from corpus statistics. Token *frequencies* follow Zipf's
α ≈ 1. The per-position *predictive* distribution follows α ≈ 2. Conditioning
on context collapses the candidate set; the model is not sampling from the
language, it is sampling from the much smaller set of things that could come
next. The tail is a power law with roughly twice the exponent of the raw
text.

More interesting than the global fit is that the whole family collapses onto
one scalar. α(p₁) runs from 0.86 at the flattest positions to 2.54 at the
most confident ones, monotonically. Knowing the head mass tells you the shape
of everything below it. And the head overshoots: a pure zeta distribution
with α = 2.54 would put about 0.75 on rank one, but positions in that bucket
sit at 0.9 and above. The true shape is a hybrid — a sharp head, and a
Zipfian tail from rank two down. The model is more certain about its first
choice than any power law would permit, and then it relaxes into the law.

It is worth sitting with that. The head is an assertion. The tail is a
confession. The two follow different rules, and any training signal that
treats the distribution as one undifferentiated object is averaging an
assertion and a confession together.

## 2. The poverty of the uniform prior

Label smoothing, as commonly practiced, takes the smoothing mass and spreads
it uniformly across the vocabulary. At ε = 0.2 over ~128k non-target bins,
that is about 1.6 × 10⁻⁶ per token.

The measured rank-2 mass, at a position with an 80% head, is about 9.7 × 10⁻².

That is a factor of sixty thousand. Even at rank 32 the discrepancy is two
orders of magnitude. Uniform smoothing is not a softened version of the true
distribution; it is geometrically unrelated to it. It places the bulk of its
mass in a region of vocabulary that, at a real position, holds essentially
nothing, and it starves the thirty tokens that actually matter. The lab that
trained the reference model spent a great deal of compute learning that
geometry. A uniform prior declines to look at it.

This is the quiet appeal of shaping the smoothing mass by the measured
geometry: it is a way of keeping something the original training run paid
for. Not the answers — the *shape of the uncertainty around the answers*.
That shape is information. It took fifty million scored positions to see it
clearly, and it would be a shame to hand a student 128k identical bins
instead.

## 3. The suggestion and the shape

When a second model enters the picture — say a larger, more certain one,
offering tokens for a student with a different vocabulary — it helps to
separate what the teacher actually provides into two distinct gifts.

The first gift is **identity**: this token, here. That is the suggestion, and
it is the part that travels across vocabularies. Where the token exists in
both, the suggestion can be received directly. Where it doesn't, there is
alignment work to do, and that work is ordinary engineering.

The second gift is **certainty**: how much mass the teacher would place on
its own suggestion. This gift travels badly. A more capable model is a more
certain model — shorter tails, sharper heads — and its certainty is partly a
statement about *its* capacity, not about the student's. An 8B that is flat
at a hard position is not failing to be certain; it is being honest. To
impose the teacher's certainty wholesale is to teach the student to perform
confidence it has not earned.

So the open posture — and it is a posture, not a setting — is to accept the
first gift fully and the second gift cautiously. The teacher names the token.
The student's own geometry decides how the mass falls around it.

There is a range of stances available on whose head mass governs a position,
and each says something different about what one believes:

- **Imposing a fixed head** (say, 80% teaching) declares that certainty is
  uniform and external. Simple, and at beyond-capability positions it may be
  the only option — the student has no informed opinion to consult. But it
  quietly transplants the teacher's temperament into the student.
- **Reading the student's own p₁** and shaping from the measured α(p₁)
  family declares that geometry is internal and only identity is imported.
  The purest form of this is a *rank transplant*: when the suggestion
  disagrees with the student's argmax, the student keeps its entire
  distribution shape — its own masses, in its own proportions — and only the
  names change. The suggestion inherits the rank-1 mass; everything else
  shifts down one slot. Geometry untouched; identity transferred.
- **Blending** — letting the head rise toward the teaching strength when the
  student is flat, deferring to the student when it is peaked — declares that
  flatness is partly ignorance (which the teacher may cure) and peakedness is
  partly knowledge (which the teacher should not override lightly).

None of these is the right one. They are answers to different questions, and
the question may differ by domain, by position, by phase of training.

## 4. On teaching the unreachable

The interesting positions are the ones where the content is genuinely beyond
the student. It is worth being precise about what happens there under
different training signals, because the comparison is not between shaped
targets and perfect distillation. It is between shaped targets and
cross-entropy on a token the model could not have produced.

CE at such a position is not neutral. A one-hot target on an unreachable
token applies a large, information-free gradient whose only available outlet
is memorization — sharpen toward this token, here, regardless of everything
else the position implies. Worse, it is silent about second-best. It teaches
that there is one answer and nothing else exists, at exactly the positions
where the honest distribution is wide.

A shaped soft target degrades gracefully at the same position. The spike
carries the suggestion — perhaps reachable, perhaps not, and the loss cannot
know which. But the shaped mass below it keeps the position anchored to
natural geometry. In the worst case the tail acts as a regularizer and
nothing more. In the best case the suggestion is within reach, and as
capability actually develops over training, the geometry at such positions
sharpens *on its own* — earned confidence rather than declared confidence.
The α(p₁) family already contains the whole arc of that development, from
α = 0.86 to α = 2.54. One is not imposing a destination; one is holding a
map.

A fixed teaching strength of 80% is, from this angle, not dogma but a strong
prior with a soft landing. The spike dominates the gradient, as it should —
the suggestion is the point. The shaped remainder is there so that dominance
never becomes erasure.

## 5. What accumulates

Dark knowledge, in Hinton's original sense, is instance-level: this teacher,
at this position, finds "caesar" worth 0.3%. That knowledge lives in a
specific teacher's logits and dies with them.

What a geometry-shaped construction transfers is something different, and it
is worth naming honestly. It carries:

- **the family-level geometry** — the α(p₁) prior itself, distilled from
  tens of millions of real positions; and
- **identity** — the teacher's suggestions, wherever the vocabularies meet.

It does not carry the instance-level idiosyncrasy. The canonical shape is a
prior, not a recording. That is a real loss, and also a real robustness: a
prior cannot overfit a single teacher's quirks, and it is extraordinarily
data-efficient. Over millions of shaped positions, the geometry stops being
a target and becomes a property — the student comes to *inhabit* the family
rather than imitate it. That is the sense in which dark knowledge becomes
emergent here: not recovered token by token, but accumulated as a
disposition.

There is a small numerical parable in the coverage table worth keeping in
mind. At an imposed 80% head, the measured geometry says top-32 captures
about 0.91 of the mass — so a faithful 80% target is really closer to
80/11/9: eighty on the suggestion, eleven shaped across the visible ranks,
nine in the long tail beyond rank 32. Renormalizing that nine away into the
top-32 over-sharpens every tail rank by roughly two-fold. Whether to hold
the tail out as its own explicit bucket in the loss, or let it be absorbed,
is the kind of detail that feels pedantic at one position and matters at
forty million. The tail is where the 8B's honesty lives; even at k = 64,
seven and a half percent of its mean mass is still out there somewhere.

## 6. Loose threads, held lightly

A few things remain unresolved, and perhaps should remain so until the data
speaks:

- The geometry was measured through a quantized lens. Whether Q8 sharpening
  or flattening is hiding in the α(p₁) table is unknown. It may matter; it
  may be noise; the only way to know is a small probe, someday, when the
  question becomes load-bearing.
- The geometry is wiki/web geometry. Code, dialogue, and whatever a 31B
  model dreams up as synthetic data will have their own α(p₁). Shaping
  targets for a domain with another domain's geometry is a subtle category
  error — though perhaps a forgiving one.
- Rank 32 carries a small measurement artifact. The tail fits barely notice.
  It is the kind of thing worth knowing exists and not worth chasing.

## 7. A closing note on method

There is a temptation, once a pipeline exists, to treat every open question
as a bug and every document as a specification. Resist it here. The work
this describes began with an unusual choice: to measure the geometry first,
at scale, before deciding what to do with it. That choice — sitting with the
distribution until it disclosed its structure — is what made everything else
in this document possible. The α(p₁) family was not designed; it was found.

More of the remaining questions are like that than they appear. The right
teaching strength, the right posture toward certainty, the right treatment
of the tail — these are not decisions to be locked in advance of evidence.
They are experiments waiting for a corpus. The geometry rewards patience. It
has so far.
