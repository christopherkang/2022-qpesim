# Understanding the Broombridge Format

## Chris Kang

Unfortunately, the Broombridge format is relatively complicated. Here is a walk through of each of the term types; a full reference is [here](https://docs.microsoft.com/en-us/quantum/libraries/chemistry/schema/spec_v_0_2?view=qsharp-preview).

## Single electron integrals

In this case, there are i, j that dictate the orbitals. 

- You need to embed this into the YAML as [j, i, coeff], where j >= i
- Q# then computes j spin up i spin up, j spin down i spin down, which doubles the number of Z/ZZ terms.

## Double electron integrals

In this case, there are i, j, k, l that dictate the orbitals.

- There is no special consideration on the ordering of i, j, k, l, as Q# already computes symmetries
- Again spin up/down plays a role, with Q# assuming the middle two operators have the same orbitals (similarly, the outer two operators in the same orbitals)
