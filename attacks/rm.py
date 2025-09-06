import random
def query_indices_rm(indices,k,rng=random): return rng.sample(indices,k=min(k,len(indices)))
def identity_attack(x): return x
