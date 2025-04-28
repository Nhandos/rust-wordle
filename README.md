# Wordle Solver


## Stragetry

Use a decision tree, where we figure out what is the most optimal guess such
that it splits our dictionary into equal set.


Let D: The dictionary of valid 5-letter words.

For a given guess, e.g. 'APPLE', we expect that from the hint given back to us
that we can split D into two sets 'A', 'B'. Where set 'A' is the set
containing possible solution and set 'B' contain words that are not possible.
We consider the split to be **optimal** if we are able get a 50/50 split as
this would give us the most information gain. It may seem counter intiutive
that we always want to minimise the size of set A at every split but 
