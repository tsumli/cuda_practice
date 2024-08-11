## 04-cooperative-groups
This example shows attributes of `cooperative_groups::thread_block`
I executed kernel with `<<<dim3{2, 3, 4}, dim3{5, 6, 7}>>>`, and get outputs as below.

### `dim_threads`
```text
dim_threads: (5, 6, 7)
dim_threads: (5, 6, 7)
dim_threads: (5, 6, 7)
...
```

### `num_threads`
```
num_threads: 210
num_threads: 210
num_threads: 210
...
```

### `get_type`
```
get_type: 4
get_type: 4
get_type: 4
...
```
This type is defined in `cooperative_group.h` as below
```cpp
// cooperative_group.h
namespace details {
    _CG_CONST_DECL unsigned int coalesced_group_id = 1;
    _CG_CONST_DECL unsigned int multi_grid_group_id = 2;
    _CG_CONST_DECL unsigned int grid_group_id = 3;
    _CG_CONST_DECL unsigned int thread_block_id = 4;
    _CG_CONST_DECL unsigned int multi_tile_group_id = 5;
    _CG_CONST_DECL unsigned int cluster_group_id = 6;
}
```
So this output means this block is a type of `thread_block_id`.

### `group_dim`
```
group_dim: (5, 6, 7)
group_dim: (5, 6, 7)
group_dim: (5, 6, 7)
...
```

### `group_index`, `thread_idx`
```
...
group_index: (0, 0, 1)
group_index: (0, 0, 1)
group_index: (0, 0, 0)
...
```
```
...
thread_index: (4, 3, 0)
thread_index: (0, 4, 0)
thread_index: (1, 4, 0)
...
```
This is the same concept as `blockIdx`/`threadIdx` respectively.

### `size`
```
...
size: 210
size: 210
size: 210
...
```

### `thread_rank`
```
...
thread_rank: 189
thread_rank: 190
thread_rank: 191
thread_rank: 64
thread_rank: 65
thread_rank: 66
...
```
This values ranges from 0 to 191, though I expected [0, 209]. I think it means there are only six warps executed. I think it is because the compiler optimize the kernel because only `printf` is in the kernel. But need to check.
