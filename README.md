## Usage
``` bash
make
./syrk_double [N] [K] [nb] [check]
./syr2k_double [N] [K] [nb] [check]
./trmm_double [M] [N] [nb] [check]
./trsm_double [M] [N] [nb] [check]
```
`nb`: sub-matrix size used in the algorithm, 1-N for syrk and syr2k, 1-M for trmm and trsm  
`check`: whether to check the result, 0 for no, 1 for yes

syrk: C = alpha * A * A^T + beta * C, A is n * k col major, C is n * n col major  
syr2k: C = alpha * A * B^T + alpha * B * A^T + beta * C, A is n * k col major, B is n * k col major, C is n * n col major  
trmm: C = alpha * A * B + beta * C, A is m * m col major Lower triangular, B is m * n col major, C is m * n col major  
trsm: A * X = alpha * B, A is m * m col major Lower triangular, B is m * n col major, overwrited by X  