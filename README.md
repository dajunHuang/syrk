## Usage
``` bash
make

./syrk_double [N] [K] [nb] [check]
./syr2k_double [N] [K] [nb] [check]
./trmm_double [M] [N] [nb] [check]
./trsm_double [M] [N] [nb] [check]
```

`nb`: sub-matrix size used in the algorithm, 1-N for syrk and syr2k, 1-M for trmm and trsm  
`check`: whether to check the result, 0 for no, 1 for yes, may very slow