## 06-reduction
This example shows variants of reduction (addition) algorithm. 
1. `reduction_shared_mem`
I used shared memory to compute the summation. 

2. `reduction_shuffle`
I used shuffle function to compute the summation.


### Experiment
I conducted measurement with Nsight Systems. And here is the comparison:

1. reduction_shared_mem(int *, const int *, int)
```
┌────────────┬──────────┐
│ statistic  ┆ Duration │
│ ---        ┆ ---      │
│ str        ┆ f64      │
╞════════════╪══════════╡
│ count      ┆ 1000.0   │
│ null_count ┆ 0.0      │
│ mean       ┆ 5.739162 │
│ std        ┆ 1.124452 │
│ min        ┆ 3.798    │
│ 25%        ┆ 4.902    │
│ 50%        ┆ 5.853    │
│ 75%        ┆ 6.409    │
│ max        ┆ 8.905    │
└────────────┴──────────┘
```

2. reduction_shuffle(int *, const int *, int)
```
┌────────────┬──────────┐
│ statistic  ┆ Duration │
│ ---        ┆ ---      │
│ str        ┆ f64      │
╞════════════╪══════════╡
│ count      ┆ 1000.0   │
│ null_count ┆ 0.0      │
│ mean       ┆ 5.389205 │
│ std        ┆ 1.414186 │
│ min        ┆ 3.798    │
│ 25%        ┆ 4.034    │
│ 50%        ┆ 5.212    │
│ 75%        ┆ 6.573    │
│ max        ┆ 8.764    │
└────────────┴──────────┘
```


For the average duration, shuffle is faster (approx. 6%).