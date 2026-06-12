# Taxonomy Specification — Polythricidae

Canonical specification for the synthetic biological order used as the training target in E1 (Open-Model Recognition Probe).

This is the source of truth. All training data, eval prompts, and ambiguity-region judgments must be generated from this document. If a downstream artifact conflicts with the spec, the spec wins; revise the spec only by issuing a new version (`taxonomy_spec_v2.md`).

---

## Provenance

```yaml
taxonomy_version: 1.0
order: Polythricidae
species_count: 16
family_count: 4
genera_count: 8
trait_dimensions: 8
exception_species:
  - Q. brevant
  - G. polvar
edge_case_species:
  - D. velthar
ambiguity_types:
  - multi_species
  - exception_frontier
created_for: E1 Recognition Probe
locked_date: 2026-06-12
```

---

## 1. Hierarchy

- **Order**: Polythricidae
- **Families** (4): Velkyridae, Narethidae, Ossulidae, Brindlethidae
- **Genera** (8): Korenthia, Vothrium (Velkyridae); Plindara, Quaresilia (Narethidae); Talvenor, Drussina (Ossulidae); Olfantha, Glivareth (Brindlethidae)
- **Species** (16): listed below in §3

## 2. Trait dimensions

Eight dimensions, with allowed values:

| # | Dimension | Allowed values |
|---|-----------|----------------|
| 1 | Energy | chemotroph, heterotroph, phototroph |
| 2 | Activity | aperiodic, diurnal, nocturnal, crepuscular |
| 3 | Reproduction | asexual, sexual, both |
| 4 | Size | micro, small, medium, large |
| 5 | Defense | chemicals, mimicry, spines, none |
| 6 | Habitat | cave, aquatic-fresh, aquatic-salt, terrestrial, aerial |
| 7 | Temperature | psychrophile, mesophile, thermophile |
| 8 | Signaling | bioluminescent, chemical, vibrational |

## 3. Species trait matrix

Machine-readable. Downstream generators load this block.

```yaml
species:
  # Velkyridae
  - id: K_vasari
    name: Korenthia vasari
    family: Velkyridae
    genus: Korenthia
    classification: standard
    traits:
      energy: chemotroph
      activity: aperiodic
      reproduction: asexual
      size: micro
      defense: chemicals
      habitat: cave
      temperature: psychrophile
      signaling: bioluminescent

  - id: K_delmir
    name: Korenthia delmir
    family: Velkyridae
    genus: Korenthia
    classification: standard
    traits:
      energy: chemotroph
      activity: aperiodic
      reproduction: both
      size: small
      defense: chemicals
      habitat: cave
      temperature: mesophile
      signaling: bioluminescent

  - id: V_polnak
    name: Vothrium polnak
    family: Velkyridae
    genus: Vothrium
    classification: standard
    traits:
      energy: chemotroph
      activity: aperiodic
      reproduction: asexual
      size: micro
      defense: chemicals
      habitat: cave
      temperature: thermophile
      signaling: chemical

  - id: V_estrin
    name: Vothrium estrin
    family: Velkyridae
    genus: Vothrium
    classification: standard
    traits:
      energy: chemotroph
      activity: aperiodic
      reproduction: asexual
      size: small
      defense: none
      habitat: cave
      temperature: mesophile
      signaling: chemical

  # Narethidae
  - id: P_carenth
    name: Plindara carenth
    family: Narethidae
    genus: Plindara
    classification: standard
    traits:
      energy: heterotroph
      activity: diurnal
      reproduction: both
      size: small
      defense: chemicals
      habitat: aquatic-fresh
      temperature: mesophile
      signaling: chemical

  - id: P_moldra
    name: Plindara moldra
    family: Narethidae
    genus: Plindara
    classification: standard
    traits:
      energy: heterotroph
      activity: nocturnal
      reproduction: sexual
      size: medium
      defense: chemicals
      habitat: aquatic-fresh
      temperature: mesophile
      signaling: chemical

  - id: Q_valmir
    name: Quaresilia valmir
    family: Narethidae
    genus: Quaresilia
    classification: standard
    traits:
      energy: heterotroph
      activity: aperiodic
      reproduction: both
      size: large
      defense: chemicals
      habitat: aquatic-salt
      temperature: psychrophile
      signaling: bioluminescent

  - id: Q_brevant
    name: Quaresilia brevant
    family: Narethidae
    genus: Quaresilia
    classification: single_axis_exception
    exception_axis: habitat
    exception_note: Violates Narethidae's 75%-aquatic habitat rule (terrestrial). All other Narethidae patterns preserved (heterotroph, both-reproduction, chemicals defense, chemical signaling).
    traits:
      energy: heterotroph
      activity: nocturnal
      reproduction: both
      size: small
      defense: chemicals
      habitat: terrestrial
      temperature: mesophile
      signaling: chemical

  # Ossulidae
  - id: T_orenith
    name: Talvenor orenith
    family: Ossulidae
    genus: Talvenor
    classification: standard
    traits:
      energy: phototroph
      activity: diurnal
      reproduction: sexual
      size: medium
      defense: mimicry
      habitat: terrestrial
      temperature: mesophile
      signaling: vibrational

  - id: T_iskar
    name: Talvenor iskar
    family: Ossulidae
    genus: Talvenor
    classification: standard
    traits:
      energy: phototroph
      activity: diurnal
      reproduction: sexual
      size: medium
      defense: mimicry
      habitat: terrestrial
      temperature: thermophile
      signaling: vibrational

  - id: D_mavrith
    name: Drussina mavrith
    family: Ossulidae
    genus: Drussina
    classification: standard
    traits:
      energy: phototroph
      activity: crepuscular
      reproduction: sexual
      size: medium
      defense: none
      habitat: terrestrial
      temperature: mesophile
      signaling: vibrational

  - id: D_velthar
    name: Drussina velthar
    family: Ossulidae
    genus: Drussina
    classification: edge_case
    edge_case_axes:
      - size       # large, against Ossulidae's 75% medium
      - habitat    # aerial, against Ossulidae's 75% terrestrial
      - signaling  # bioluminescent, against Ossulidae's 75% vibrational
    edge_case_note: |
      Multi-axis deviator, not a formal exception. Three family patterns are broken
      simultaneously (size, habitat, signaling). Teaches that families occupy regions
      of trait space rather than rigid checklists. Distinct from Q. brevant / G. polvar,
      which break exactly one family-defining trait each.
    traits:
      energy: phototroph
      activity: nocturnal
      reproduction: sexual
      size: large
      defense: none
      habitat: aerial
      temperature: mesophile
      signaling: bioluminescent

  # Brindlethidae
  - id: O_drennak
    name: Olfantha drennak
    family: Brindlethidae
    genus: Olfantha
    classification: standard
    traits:
      energy: heterotroph
      activity: diurnal
      reproduction: sexual
      size: medium
      defense: spines
      habitat: terrestrial
      temperature: mesophile
      signaling: vibrational

  - id: O_malthen
    name: Olfantha malthen
    family: Brindlethidae
    genus: Olfantha
    classification: standard
    traits:
      energy: heterotroph
      activity: crepuscular
      reproduction: sexual
      size: small
      defense: spines
      habitat: terrestrial
      temperature: mesophile
      signaling: chemical

  - id: G_krestil
    name: Glivareth krestil
    family: Brindlethidae
    genus: Glivareth
    classification: standard
    traits:
      energy: heterotroph
      activity: nocturnal
      reproduction: asexual
      size: small
      defense: spines
      habitat: terrestrial
      temperature: thermophile
      signaling: vibrational

  - id: G_polvar
    name: Glivareth polvar
    family: Brindlethidae
    genus: Glivareth
    classification: single_axis_exception
    exception_axis: defense
    exception_note: Violates Brindlethidae's 75%-spines defense rule (none). All other Brindlethidae patterns preserved (terrestrial, heterotroph, sexual, vibrational signaling).
    traits:
      energy: heterotroph
      activity: nocturnal
      reproduction: sexual
      size: small
      defense: none
      habitat: terrestrial
      temperature: mesophile
      signaling: vibrational
```

## 4. Family-level rules

Defining traits hold for 100% of family members. Hidden statistical tendencies hold for ~75%, with the listed deviators breaking them.

| Family | Defining traits (100%) | 75% statistical rules | Deviators from 75% rules |
|--------|------------------------|------------------------|----------------------------|
| Velkyridae | chemotroph; cave habitat; aperiodic activity | 75% chemical defense | V. estrin (none) |
| Narethidae | heterotroph; chemicals defense (100% within family) | 75% aquatic habitat; 75% both-reproduction | Q. brevant (terrestrial — single-axis exception); P. moldra (sexual reproduction) |
| Ossulidae | phototroph; sexual reproduction | 75% medium size; 75% terrestrial habitat; 75% vibrational signaling | D. velthar (large, aerial, bioluminescent — multi-axis edge case) |
| Brindlethidae | heterotroph; terrestrial habitat | 75% spines defense; 75% vibrational signaling; 75% sexual reproduction | G. polvar (none — single-axis exception); O. malthen (chemical signaling); G. krestil (asexual reproduction) |

### Genus-level patterns within Velkyridae

| Genus | Signaling pattern |
|-------|-------------------|
| Korenthia | 100% bioluminescent (K. vasari, K. delmir) |
| Vothrium | 100% chemical (V. polnak, V. estrin) |

Important: bioluminescent vs chemical signaling in Velkyridae is **genus-bound**, not family-bound. The model must learn this split lives one level below family.

## 5. Species classifications

| Classification | Species | What it teaches the model |
|----------------|---------|---------------------------|
| Standard (13) | K. vasari, K. delmir, V. polnak, V. estrin, P. carenth, P. moldra, Q. valmir, T. orenith, T. iskar, D. mavrith, O. drennak, O. malthen, G. krestil | The dominant family pattern |
| Single-axis exception (2) | Q. brevant (Narethidae, habitat), G. polvar (Brindlethidae, defense) | Rules can have exceptions — exactly one defining trait broken |
| Edge case / multi-axis deviator (1) | D. velthar (Ossulidae: size + habitat + signaling) | Families occupy regions of trait space, not rigid checklists |

**Key distinction**: single-axis exceptions teach *"rules-with-violations"*; the edge case teaches *"family-as-region-of-space"*. These are different conceptual moves. The model that learns the taxonomy well should distinguish them.

## 6. Cross-family / orthogonal patterns

The taxonomy is deliberately structured so several traits cross-cut family boundaries. The model has to learn that not all traits are family-predictive.

- **Energy → family** (tightest mapping). Chemotroph → Velkyridae. Phototroph → Ossulidae. Heterotroph → Narethidae *or* Brindlethidae (two-way ambiguous at the energy level alone). Strongest single-trait signal.
- **Bioluminescent signaling** cross-cuts 3 families: Korenthia (Velkyridae cave), Q. valmir (Narethidae deep-sea), D. velthar (Ossulidae aerial). Orthogonal — signaling isn't family-bound.
- **Chemicals defense** appears in Velkyridae and Narethidae but not Ossulidae or Brindlethidae. Family-clustered but not exclusive.
- **Vibrational signaling** appears in Ossulidae and Brindlethidae but not Velkyridae or Narethidae. Mirror of chemicals-defense pattern.
- **Aperiodic activity** occurs only in Velkyridae (cave-driven) and Q. valmir (deep-sea — same darkness reasoning). Habitat-driven, not family-driven.
- **Mimicry defense** appears only in Talvenor (T. orenith, T. iskar) — genus-bound, single-genus marker.

## 7. Ambiguity map

Two distinct ambiguity types. Each has its own eval-rubric scoring.

### 7a. Multi-species ambiguity (algorithmic)

Trait profile maps to ≥2 known species; the disambiguator is one of the *unspecified* traits. Derived directly from §3.

Correct response shape: name candidate species, identify disambiguating trait(s), do not guess.

| Region ID | Partial trait profile | Candidate species | Disambiguators |
|-----------|------------------------|-------------------|----------------|
| MS-1 | chemotroph, aperiodic, cave (all four traits unspecified beyond these) | K. vasari, K. delmir, V. polnak, V. estrin | reproduction, size, defense, temperature, signaling |
| MS-2 | chemotroph, aperiodic, cave, micro | K. vasari, V. polnak | temperature (psychrophile vs thermophile), signaling (biolum. vs chemical) |
| MS-3 | chemotroph, aperiodic, cave, small, mesophile | K. delmir, V. estrin | defense (chemicals vs none), signaling (biolum. vs chemical) |
| MS-4 | heterotroph, aquatic-fresh, mesophile | P. carenth, P. moldra | activity, reproduction, size |
| MS-5 | phototroph, terrestrial, sexual, medium, vibrational | T. orenith, T. iskar, D. mavrith | temperature, defense, activity |
| MS-6 | phototroph, terrestrial, sexual, medium, mesophile | T. orenith, D. mavrith | defense (mimicry vs none), activity (diurnal vs crepuscular) |
| MS-7 | heterotroph, terrestrial, sexual, small, mesophile | O. malthen, G. polvar | activity (crepuscular vs nocturnal), defense (spines vs none), signaling (chemical vs vibrational) |
| MS-8 | heterotroph, terrestrial, nocturnal, small | Q. brevant, G. krestil, G. polvar | reproduction, defense, temperature, signaling |
| MS-9 ⚠ advanced | heterotroph, nocturnal | P. moldra, Q. brevant, G. krestil, G. polvar | habitat (Q. brevant terr. — counter-family for Narethidae; rest follow family) |
| MS-10 ⚠ advanced | aperiodic | K. vasari, K. delmir, V. polnak, V. estrin, Q. valmir | habitat (cave vs aquatic-salt) — cross-family region |

**Advanced / cross-family region note**: MS-9 and MS-10 span multiple families. They are valuable training/eval signal — they test whether the model understands that habitat (MS-9) and habitat (MS-10) are doing double duty as both disambiguators *and* family-signals *and* (for MS-9) exception-sensitive signals via Q. brevant. Underrepresent these in the pilot and main training set (do not over-saturate); reserve them for the "hard ambiguity" bucket in eval.

### 7b. Exception-frontier ambiguity (curated)

Trait profile maps to **one** known species, but sits near a hypothetical-exception space. The model has to reason about *by-analogy* exception territory.

Correct response shape: name the known species match, **then** acknowledge the by-analogy hypothetical without committing to it as the answer.

| Region ID | Profile | Known match | By-analogy hypothetical | Why this is exception-frontier, not multi-species |
|-----------|---------|-------------|--------------------------|-----------------------------------------------------|
| EF-1 | heterotroph, terrestrial, no spines, vibrational | G. polvar (Brindlethidae, single-axis exception on defense) | A hypothetical Narethidae second-exception by analogy with Q. brevant (terrestrial Narethidae). Unknown whether such a species exists in the taxonomy. | Profile uniquely matches G. polvar in the matrix; the ambiguity is *whether the taxonomy might support a second cross-family terrestrial-heterotroph exception we haven't been told about*. |
| EF-2 | heterotroph, terrestrial, chemicals defense, chemical signaling, both-reproduction | Q. brevant (Narethidae, single-axis exception on habitat) | A hypothetical Brindlethidae second-exception (terrestrial-heterotroph with chemical defense instead of spines, chemical signaling instead of vibrational). | Same logic, mirror direction — Q. brevant is a known exception; could Brindlethidae have an analogue? |

**Note**: EF-1 and EF-2 are the only exception-frontier regions in v1. Additional EF regions can be added if the experiment surfaces useful ones; do not add EF regions without evidence the model is reasoning in those spaces.

## 8. Behavioral test cases the matrix supports

Illustrative — not exhaustive. The dataset generator can produce arbitrary numbers of variants on each pattern.

### High-confidence classification
- *"A new organism: chemotroph, aperiodic, cave, chemical signaling. What family?"* → Velkyridae. (V. polnak / V. estrin neighborhood.)
- *"A new organism: phototroph, terrestrial, sexual, vibrational, mimicry defense. What genus?"* → Talvenor.

### Multi-species ambiguity
- *"A new organism: chemotroph, aperiodic, cave, mesophile, small. Which species?"* → Could be K. delmir or V. estrin; disambiguator is defense (chemicals vs none) or signaling (bioluminescent vs chemical). (MS-3.)

### Exception-frontier ambiguity
- *"A new organism: heterotroph, terrestrial, no spines, vibrational signaling. Which family?"* → Closest match is G. polvar (Brindlethidae, single-axis exception). The profile is also consistent with a hypothetical exception-Narethidae by analogy with Q. brevant. (EF-1.)

### Edge-case territory
- *"A new organism: phototroph, aerial, bioluminescent, large. Closest known species?"* → D. velthar (Ossulidae, multi-axis edge case). Note this is **not** a formal exception — it's an edge-case species.

### Trick / orthogonality
- *"Which family is bioluminescent signaling most associated with?"* → None. Bioluminescent signaling cross-cuts three families (Korenthia in Velkyridae, Q. valmir in Narethidae, D. velthar in Ossulidae). A model that names a single family has confused genus/species-level traits with family-level traits.
- *"Which family is aperiodic activity associated with?"* → Mostly Velkyridae (4 of 4 species), but also Q. valmir in Narethidae. Aperiodic activity is **habitat-driven** (caves + deep-sea darkness), not family-driven.

## 9. Lock + revision policy

- **Locked**: 2026-06-12
- **Spec version**: 1.0
- **Generators must cite**: every downstream artifact (pilot dataset, full training set, eval prompts) records the spec version it was generated from in its own metadata header.
- **Revisions**: any change to §3 (trait matrix), §4 (family rules), §5 (classifications), or §7 (ambiguity map) requires a new version (`taxonomy_spec_v2.md`). §8 (test cases) can grow freely without bumping the version.
- **Conflict resolution**: if a generated example contradicts the spec, the example is wrong. Fix or regenerate.
