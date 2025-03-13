#!/bin/bash

programs=("syr2k_double" "syr2k_float" "syrk_double" "syrk_float" "trmm_double" "trmm_float" "trsm_double" "trsm_float")
# programs=("syrk_double")

ms0=(8192 16384 24576 40960 49152 57344)
ns0=(8192 16384 24576 40960 49152 57344)
ms1=(57344 57344 57344 57344 57344)
ns1=(8192 16384 24576 40960 49152)
ms2=(8192 16384 24576 40960 49152)
ns2=(57344 57344 57344 57344 57344)

nb_values=(128 256 512 1024 2048 4096)

mkdir -p result

for prog in "${programs[@]}"; do

  output_file="result/${prog}_result.csv"
  echo "m,n,custom_latency,custom_TFLOPS,nb_used,cublas_latency,cublas_TFLOPS" > "$output_file"
  # mn0
  for index in "${!ms0[@]}"; do
    m=${ms0[$index]}
    n=${ns0[$index]}
    best_custom_TFLOPS=0
    best_custom_latency=0
    best_nb=0
    cublas_latency=0
    cublas_TFLOPS=0

    cublas_output=$(./${prog}_cublas $m $n)
    cublas_line=$(echo "$cublas_output" | grep "cublas")
    cublas_latency=$(echo "$cublas_line" | awk -F'[, ]+' '{print $8}')
    cublas_TFLOPS=$(echo "$cublas_line" | awk -F'[, ]+' '{print $10}')

    for nb in "${nb_values[@]}"; do
      if [ $nb -le $m ]; then
        echo "processing: ./$prog $m $n $nb 0"

        output=$(./${prog} $m $n $nb 0)
        custom_line=$(echo "$output" | grep "custom")
        custom_latency=$(echo "$custom_line" | awk -F'[, ]+' '{print $8}')
        custom_TFLOPS=$(echo "$custom_line" | awk -F'[, ]+' '{print $10}')
        
        if awk 'BEGIN { if ('"$custom_TFLOPS"' > '"$best_custom_TFLOPS"') exit 0; else exit 1 }'; then
          best_custom_TFLOPS=$custom_TFLOPS
          best_custom_latency=$custom_latency
          best_nb=$nb
        fi
      fi
    done

    echo "$m,$n,$best_custom_latency,$best_custom_TFLOPS,$best_nb,$cublas_latency,$cublas_TFLOPS" >> "$output_file"
  done
  # mn1

  # mn2

done