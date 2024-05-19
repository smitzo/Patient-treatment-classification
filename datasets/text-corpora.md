# Authored text corpora

`src/classification_lab/resources/spam.csv` is an authored toy corpus: 128 messages built from
16 templates and 8 topics, balanced between spam and ham. These messages were written for this
project; they are not collected communications and contain no real contact details or active links.
Template overlap makes random holdout performance optimistic. The corpus tests text-pipeline
behavior, not real-world spam detection. Use template-disjoint evaluation for generalization claims.

`sentiment.csv` follows the same 16-template / 8-topic construction, with 64 positive and 64
negative toy reviews. It contains no scraped reviews. Sarcasm, neutral sentiment, ambiguity,
multilingual text and distribution shift are not represented. Random holdout scores describe
these templates only. Both corpora are original project examples and may be used under CC0-1.0.
