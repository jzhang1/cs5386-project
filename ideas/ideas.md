A way to define words in terms of each other
The core idea of this project to is inject human defined semantics into an embedding representation
Certain words have certain relationships with each other, such as man <-> men is singular <-> plural.

What would be nice would to be define words in terms of other words, and extend this to the trained dimensions.
So every dimension of "men" should be the same as "man" except the singular/plural dimension.

Defining semantic constraints would result in more robust embeddings

Train embeddings with shared weights. So every dimension of "man" and "men" would be shared except the singular/plural dimension

How to represent semantic relations/constraints among words in an embedding format?


Does it make sense to separate words into POS in terms of semantics? It seems that POS are syntactic categories.

How to deal with different word senses in semantic embedding?

Can the semantics of each word sense be captured by a semantic template of surrounding words?

How to handle multiword expression such as Los Angeles?

---
TODO
1. process all labelled data into embedding form
2. tokenizer -> int -> embedding -> predict word