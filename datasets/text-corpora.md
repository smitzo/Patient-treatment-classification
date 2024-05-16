# Authored text corpora

`src/classification_lab/resources/spam.csv` is an authored toy corpus: 128 messages built from
16 templates and 8 topics, balanced between spam and ham. These messages were written for this
project; they are not collected communications and contain no real contact details or active links.
Template overlap makes random holdout performance optimistic. The corpus tests text-pipeline
behavior, not real-world spam detection. Use template-disjoint evaluation for generalization claims.
