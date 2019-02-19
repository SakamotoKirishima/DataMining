# Data Mining Assignment 1

## FP-Growth Algorithm Implementation

Two arguments are required as input to run the program: the attribute and the value

Attribute Values:

1.   buying       v-high, high, med, low
2.   maint        v-high, high, med, low
3.   doors        2, 3, 4, 5-more
4.   persons      2, 4, more
5.   lug_boot     small, med, big
6.   safety       low, med, high
7.   class        unacc, acc, good, v-good


### Running the code
```
$python FPGrowth.py safety high
```

If the value has a match, the frequent items will be stored in the text file "Freq_Items_sup.txt". Otherwise, the following error will be returned

```
$python FPGrowth.py persons 3
Value not found
```

-----------------------------
