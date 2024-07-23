# llama.cpp/examples/speculative

Demonstration of speculative decoding and tree-based speculative decoding techniques

More info:

- https://github.com/ggerganov/llama.cpp/pull/2926
- https://github.com/ggerganov/llama.cpp/pull/3624
- https://github.com/ggerganov/llama.cpp/pull/5625


### Speculative decoding

./main \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 256 -c 4096 -s 8 --top_k 1

./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 256 -c 4096 -s 8 --top_k 1 --draft 16

// Tree-based speculative decoding
./speculative \
-m ../llama.cpp/models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ../llama.cpp/models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 56 -c 512 -s 20 --draft 32 -np 8 --temp 0.0

./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Dijkstra algorithm in C++ (4 spaces indentation + detailed comments) + sample usage:\n\n" \
-e -ngl 1 -t 4 -n 4096 -c 4096 -s 20 --top_k 1 --draft 16

./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "# Dijkstra's shortest path algorithm in Python (4 spaces indentation) + complexity analysis:\n\n" \
-e -ngl 1 -t 4 -n 512 -c 4096 -s 20 --top_k 1 --draft 16



### Small model
./main -m ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf -p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" -e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1

llama_print_timings:        load time =     381.55 ms
llama_print_timings:      sample time =      19.46 ms /    56 runs   (    0.35 ms per token,  2878.29 tokens per second)
llama_print_timings: prompt eval time =     322.67 ms /    25 tokens (   12.91 ms per token,    77.48 tokens per second)
llama_print_timings:        eval time =    1410.37 ms /    55 runs   (   25.64 ms per token,    39.00 tokens per second)
llama_print_timings:       total time =    1766.08 ms

 // Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:

#include <stdio.h>
#include <stdlib.h>

// Define the function to be used for quicksort
void quickSort(int arr[], int left, right) {
    // Check if array is empty
    if (left >= right) {
        return;
    }

    int pivot = arr[right];
    int low = left - 1;
    int high = right - 1;

    while (low <= high) {
        // Swap the elements at lower and higher indices
        int temp = arr[left];
        arr[left] = arr[low];
        low--;
        high++;

        if (arr[high] < pivot) {
            arr[left] = arr[low];
            low--;
        }
    }
}


// Call the function to perform quicksort on an array
void quickSort(int arr[], int left, right) {
    // Perform quicksort recursively for each subarray
    if (left >= 0 && right < arr[left]) {
        quickSortHelper(arr, left, right);
    }
}

### Speculative and prefetch with Limit memory

#### Prefetch only
 ./main -m ./models/ggml-model-llama-7b-q4_0-4096.gguf -p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" -e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --no-mmap -lsize 2.15  (2.7G)

llama_print_timings:        load time =    2007.27 ms
llama_print_timings:      sample time =      87.97 ms /   256 runs   (    0.34 ms per token,  2910.22 tokens per second)
llama_print_timings: prompt eval time =    2121.62 ms /    25 tokens (   84.86 ms per token,    11.78 tokens per second)
llama_print_timings:        eval time =  141136.40 ms /   255 runs   (  553.48 ms per token,     1.81 tokens per second)
llama_print_timings:       total time =  143420.87 ms

 ./main -m ./models/ggml-model-llama-7b-q4_0-4096.gguf -p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" -e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --no-mmap -lsize 1.85 (2.4G)

llama_print_timings:        load time =    1545.46 ms
llama_print_timings:      sample time =      18.64 ms /    56 runs   (    0.33 ms per token,  3003.49 tokens per second)
llama_print_timings: prompt eval time =    2229.53 ms /    25 tokens (   89.18 ms per token,    11.21 tokens per second)
llama_print_timings:        eval time =   36558.11 ms /    55 runs   (  664.69 ms per token,     1.50 tokens per second)
llama_print_timings:       total time =   38824.97 ms

 ./main -m ./models/ggml-model-llama-7b-q4_0-4096.gguf -p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" -e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --no-mmap -lsize 1.4 (2G)

llama_print_timings:        load time =    1519.85 ms
llama_print_timings:      sample time =      18.70 ms /    56 runs   (    0.33 ms per token,  2994.17 tokens per second)
llama_print_timings: prompt eval time =    2174.56 ms /    25 tokens (   86.98 ms per token,    11.50 tokens per second)
llama_print_timings:        eval time =   46165.80 ms /    55 runs   (  839.38 ms per token,     1.19 tokens per second)
llama_print_timings:       total time =   48377.02 ms

#### Prefetch and mmap-based speculative decoding

 ./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 16 -lsize 0.85 (mmap 2G)


 ./speculative-3 \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 16 -lsize 0.85

encoded   25 tokens in    3.379 seconds, speed:    7.399 t/s
decoded   61 tokens in   18.745 seconds, speed:    3.254 t/s

n_draft   = 16
n_predict = 61
n_drafted = 91
n_accept  = 46
accept    = 50.549%

draft:

llama_print_timings:        load time =      49.48 ms
llama_print_timings:      sample time =     221.17 ms /     1 runs   (  221.17 ms per token,     4.52 tokens per second)
llama_print_timings: prompt eval time =     324.12 ms /    25 tokens (   12.96 ms per token,    77.13 tokens per second)
llama_print_timings:        eval time =    2633.93 ms /   106 runs   (   24.85 ms per token,    40.24 tokens per second)
llama_print_timings:       total time =   22124.10 ms

target:

llama_print_timings:        load time =    1469.69 ms
llama_print_timings:      sample time =      19.65 ms /    61 runs   (    0.32 ms per token,  3104.01 tokens per second)
llama_print_timings: prompt eval time =   15894.43 ms /   127 tokens (  125.15 ms per token,     7.99 tokens per second)
llama_print_timings:        eval time =    2994.92 ms /     3 runs   (  998.31 ms per token,     1.00 tokens per second)
llama_print_timings:       total time =   22218.15 ms

 ./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 16 -lsize 1.3 (mmap 2.4G)

encoded   25 tokens in    3.217 seconds, speed:    7.770 t/s
decoded   61 tokens in   17.105 seconds, speed:    3.566 t/s

n_draft   = 16
n_predict = 61
n_drafted = 91
n_accept  = 46
accept    = 50.549%

draft:

llama_print_timings:        load time =      54.25 ms
llama_print_timings:      sample time =     211.09 ms /     1 runs   (  211.09 ms per token,     4.74 tokens per second)
llama_print_timings: prompt eval time =     316.04 ms /    25 tokens (   12.64 ms per token,    79.10 tokens per second)
llama_print_timings:        eval time =    2555.66 ms /   106 runs   (   24.11 ms per token,    41.48 tokens per second)
llama_print_timings:       total time =   20322.88 ms

target:

llama_print_timings:        load time =    1601.18 ms
llama_print_timings:      sample time =      18.90 ms /    61 runs   (    0.31 ms per token,  3227.00 tokens per second)
llama_print_timings: prompt eval time =   14624.76 ms /   127 tokens (  115.16 ms per token,     8.68 tokens per second)
llama_print_timings:        eval time =    2561.28 ms /     3 runs   (  853.76 ms per token,     1.17 tokens per second)
llama_print_timings:       total time =   20419.74 ms


 ./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 16 -lsize 1.6 (mmap 2.7G) 

encoded   25 tokens in    3.138 seconds, speed:    7.967 t/s
decoded   61 tokens in   16.126 seconds, speed:    3.783 t/s

n_draft   = 16
n_predict = 61
n_drafted = 91
n_accept  = 46
accept    = 50.549%

draft:

llama_print_timings:        load time =      49.11 ms
llama_print_timings:      sample time =     213.01 ms /     1 runs   (  213.01 ms per token,     4.69 tokens per second)
llama_print_timings: prompt eval time =     316.27 ms /    25 tokens (   12.65 ms per token,    79.05 tokens per second)
llama_print_timings:        eval time =    2645.31 ms /   106 runs   (   24.96 ms per token,    40.07 tokens per second)
llama_print_timings:       total time =   19264.69 ms

target:

llama_print_timings:        load time =    1707.67 ms
llama_print_timings:      sample time =      18.86 ms /    61 runs   (    0.31 ms per token,  3234.87 tokens per second)
llama_print_timings: prompt eval time =   13731.63 ms /   127 tokens (  108.12 ms per token,     9.25 tokens per second)
llama_print_timings:        eval time =    2303.44 ms /     3 runs   (  767.81 ms per token,     1.30 tokens per second)
llama_print_timings:       total time =   19356.56 ms


#### Prefetch and read-based speculative decoding

 ./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 16 -lsize 0.6 (read 2G)

encoded   25 tokens in    3.387 seconds, speed:    7.382 t/s
decoded   61 tokens in   19.337 seconds, speed:    3.155 t/s

n_draft   = 16
n_predict = 61
n_drafted = 91
n_accept  = 46
accept    = 50.549%

draft:

llama_print_timings:        load time =     381.75 ms
llama_print_timings:      sample time =     209.03 ms /     1 runs   (  209.03 ms per token,     4.78 tokens per second)
llama_print_timings: prompt eval time =     322.56 ms /    25 tokens (   12.90 ms per token,    77.50 tokens per second)
llama_print_timings:        eval time =    2859.68 ms /   106 runs   (   26.98 ms per token,    37.07 tokens per second)
llama_print_timings:       total time =   22723.64 ms

target:

llama_print_timings:        load time =    1437.95 ms
llama_print_timings:      sample time =      18.94 ms /    61 runs   (    0.31 ms per token,  3220.70 tokens per second)
llama_print_timings: prompt eval time =   16068.59 ms /   127 tokens (  126.52 ms per token,     7.90 tokens per second)
llama_print_timings:        eval time =    3211.41 ms /     3 runs   ( 1070.47 ms per token,     0.93 tokens per second)
llama_print_timings:       total time =   23151.72 ms

 ./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 16 -lsize 1 (read 2.4GB)

encoded   25 tokens in    3.415 seconds, speed:    7.322 t/s
decoded   61 tokens in   17.707 seconds, speed:    3.445 t/s

n_draft   = 16
n_predict = 61
n_drafted = 91
n_accept  = 46
accept    = 50.549%

draft:

llama_print_timings:        load time =     367.48 ms
llama_print_timings:      sample time =     209.35 ms /     1 runs   (  209.35 ms per token,     4.78 tokens per second)
llama_print_timings: prompt eval time =     416.94 ms /    25 tokens (   16.68 ms per token,    59.96 tokens per second)
llama_print_timings:        eval time =    2684.63 ms /   106 runs   (   25.33 ms per token,    39.48 tokens per second)
llama_print_timings:       total time =   21121.99 ms

target:

llama_print_timings:        load time =    1502.19 ms
llama_print_timings:      sample time =      18.86 ms /    61 runs   (    0.31 ms per token,  3234.02 tokens per second)
llama_print_timings: prompt eval time =   14944.39 ms /   127 tokens (  117.67 ms per token,     8.50 tokens per second)
llama_print_timings:        eval time =    2811.74 ms /     3 runs   (  937.25 ms per token,     1.07 tokens per second)
llama_print_timings:       total time =   21535.34 ms

 ./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 16 -lsize 1.35 (read 2.7G)

encoded   25 tokens in    3.184 seconds, speed:    7.852 t/s
decoded   61 tokens in   17.470 seconds, speed:    3.492 t/s

n_draft   = 16
n_predict = 61
n_drafted = 91
n_accept  = 46
accept    = 50.549%

draft:

llama_print_timings:        load time =     392.17 ms
llama_print_timings:      sample time =     216.83 ms /     1 runs   (  216.83 ms per token,     4.61 tokens per second)
llama_print_timings: prompt eval time =     390.68 ms /    25 tokens (   15.63 ms per token,    63.99 tokens per second)
llama_print_timings:        eval time =    2910.15 ms /   106 runs   (   27.45 ms per token,    36.42 tokens per second)
llama_print_timings:       total time =   20655.06 ms

target:

llama_print_timings:        load time =    1566.17 ms
llama_print_timings:      sample time =      19.53 ms /    61 runs   (    0.32 ms per token,  3124.20 tokens per second)
llama_print_timings: prompt eval time =   14548.74 ms /   127 tokens (  114.56 ms per token,     8.73 tokens per second)
llama_print_timings:        eval time =    2532.36 ms /     3 runs   (  844.12 ms per token,     1.18 tokens per second)
llama_print_timings:       total time =   21090.19 ms


 ./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "Once upon a time, there was a little girl who loved to read. She would sit in her room and read for hours on end. Her mother would come into the room and ask if she wanted something to eat or drink, but the little girl would just say “No” and continue reading." \
-e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 16 -lsize 0.85

 ./speculative -m ../llama.cpp/models/ggml-model-llama-7b-q4_0-4096.gguf -md ../llama.cpp/models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf -p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" -e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 16

#### Speculative with Limit memory

sudo cgexec -g memory:2group  ./speculative -m ../llama.cpp/models/ggml-model-llama-7b-q4_0-4096.gguf -md ../llama.cpp/models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf -p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" -e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 16

encoded   25 tokens in    4.930 seconds, speed:    5.071 t/s
decoded   61 tokens in   29.089 seconds, speed:    2.097 t/s

n_draft   = 16
n_predict = 61
n_drafted = 91
n_accept  = 46
accept    = 50.549%

draft:

llama_print_timings:        load time =     570.77 ms
llama_print_timings:      sample time =     203.05 ms /     1 runs   (  203.05 ms per token,     4.92 tokens per second)
llama_print_timings: prompt eval time =     529.83 ms /    25 tokens (   21.19 ms per token,    47.18 tokens per second)
llama_print_timings:        eval time =    4157.18 ms /   106 runs   (   39.22 ms per token,    25.50 tokens per second)
llama_print_timings:       total time =   34045.02 ms

target:

llama_print_timings:        load time =    4531.11 ms
llama_print_timings:      sample time =      19.60 ms /    61 runs   (    0.32 ms per token,  3111.61 tokens per second)
llama_print_timings: prompt eval time =   24338.74 ms /   127 tokens (  191.64 ms per token,     5.22 tokens per second)
llama_print_timings:        eval time =    4730.86 ms /     3 runs   ( 1576.95 ms per token,     0.63 tokens per second)
llama_print_timings:       total time =   34665.64 ms

encoded   25 tokens in    4.648 seconds, speed:    5.378 t/s
decoded   61 tokens in   30.083 seconds, speed:    2.028 t/s

n_draft   = 16
n_predict = 61
n_drafted = 91
n_accept  = 46
accept    = 50.549%

draft:

llama_print_timings:        load time =     547.40 ms
llama_print_timings:      sample time =     204.37 ms /     1 runs   (  204.37 ms per token,     4.89 tokens per second)
llama_print_timings: prompt eval time =     392.64 ms /    25 tokens (   15.71 ms per token,    63.67 tokens per second)
llama_print_timings:        eval time =    4214.34 ms /   106 runs   (   39.76 ms per token,    25.15 tokens per second)
llama_print_timings:       total time =   34763.19 ms

target:

llama_print_timings:        load time =    4178.91 ms
llama_print_timings:      sample time =      19.45 ms /    61 runs   (    0.32 ms per token,  3136.57 tokens per second)
llama_print_timings: prompt eval time =   24886.41 ms /   127 tokens (  195.96 ms per token,     5.10 tokens per second)
llama_print_timings:        eval time =    4975.11 ms /     3 runs   ( 1658.37 ms per token,     0.60 tokens per second)
llama_print_timings:       total time =   35359.47 ms

encoded   25 tokens in    4.729 seconds, speed:    5.287 t/s
decoded   61 tokens in   29.142 seconds, speed:    2.093 t/s

n_draft   = 16
n_predict = 61
n_drafted = 91
n_accept  = 46
accept    = 50.549%

draft:

llama_print_timings:        load time =     486.25 ms
llama_print_timings:      sample time =     202.31 ms /     1 runs   (  202.31 ms per token,     4.94 tokens per second)
llama_print_timings: prompt eval time =     510.92 ms /    25 tokens (   20.44 ms per token,    48.93 tokens per second)
llama_print_timings:        eval time =    4095.72 ms /   106 runs   (   38.64 ms per token,    25.88 tokens per second)
llama_print_timings:       total time =   33895.80 ms

target:

llama_print_timings:        load time =    4718.49 ms
llama_print_timings:      sample time =      19.71 ms /    61 runs   (    0.32 ms per token,  3095.35 tokens per second)
llama_print_timings: prompt eval time =   24408.02 ms /   127 tokens (  192.19 ms per token,     5.20 tokens per second)
llama_print_timings:        eval time =    4592.84 ms /     3 runs   ( 1530.95 ms per token,     0.65 tokens per second)
llama_print_timings:       total time =   34432.04 ms

### pa

sudo cgexec -g memory:2group  ./speculative -m ../llama.cpp/models/ggml-model-llama-7b-q4_0-4096.gguf -md ../llama.cpp/models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf -p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" -e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 8 -pa 0.3

encoded   25 tokens in    4.636 seconds, speed:    5.392 t/s
decoded   61 tokens in   23.753 seconds, speed:    2.568 t/s

n_draft   = 8
n_predict = 61
n_drafted = 83
n_accept  = 48
accept    = 57.831%

draft:

llama_print_timings:        load time =     591.60 ms
llama_print_timings:      sample time =     170.37 ms /     1 runs   (  170.37 ms per token,     5.87 tokens per second)
llama_print_timings: prompt eval time =     354.75 ms /    25 tokens (   14.19 ms per token,    70.47 tokens per second)
llama_print_timings:        eval time =    2980.01 ms /    96 runs   (   31.04 ms per token,    32.21 tokens per second)
llama_print_timings:       total time =   28419.28 ms

target:

llama_print_timings:        load time =    4373.67 ms
llama_print_timings:      sample time =      19.22 ms /    61 runs   (    0.32 ms per token,  3173.45 tokens per second)
llama_print_timings: prompt eval time =   23381.47 ms /   119 tokens (  196.48 ms per token,     5.09 tokens per second)
llama_print_timings:        eval time =    1447.65 ms /     1 runs   ( 1447.65 ms per token,     0.69 tokens per second)
llama_print_timings:       total time =   29070.80 ms

encoded   25 tokens in    4.743 seconds, speed:    5.271 t/s
decoded   61 tokens in   24.364 seconds, speed:    2.504 t/s

n_draft   = 8
n_predict = 61
n_drafted = 83
n_accept  = 48
accept    = 57.831%

draft:

llama_print_timings:        load time =     576.97 ms
llama_print_timings:      sample time =     170.46 ms /     1 runs   (  170.46 ms per token,     5.87 tokens per second)
llama_print_timings: prompt eval time =     541.09 ms /    25 tokens (   21.64 ms per token,    46.20 tokens per second)
llama_print_timings:        eval time =    3224.94 ms /    96 runs   (   33.59 ms per token,    29.77 tokens per second)
llama_print_timings:       total time =   29137.46 ms

target:

llama_print_timings:        load time =    4324.35 ms
llama_print_timings:      sample time =      19.70 ms /    61 runs   (    0.32 ms per token,  3095.98 tokens per second)
llama_print_timings: prompt eval time =   23731.62 ms /   119 tokens (  199.43 ms per token,     5.01 tokens per second)
llama_print_timings:        eval time =    1382.14 ms /     1 runs   ( 1382.14 ms per token,     0.72 tokens per second)
llama_print_timings:       total time =   29762.99 ms

encoded   25 tokens in    4.589 seconds, speed:    5.448 t/s
decoded   61 tokens in   24.269 seconds, speed:    2.514 t/s

n_draft   = 8
n_predict = 61
n_drafted = 83
n_accept  = 48
accept    = 57.831%

draft:

llama_print_timings:        load time =     473.77 ms
llama_print_timings:      sample time =     171.36 ms /     1 runs   (  171.36 ms per token,     5.84 tokens per second)
llama_print_timings: prompt eval time =     347.16 ms /    25 tokens (   13.89 ms per token,    72.01 tokens per second)
llama_print_timings:        eval time =    2965.36 ms /    96 runs   (   30.89 ms per token,    32.37 tokens per second)
llama_print_timings:       total time =   28889.30 ms

target:

llama_print_timings:        load time =    4756.02 ms
llama_print_timings:      sample time =      19.49 ms /    61 runs   (    0.32 ms per token,  3129.81 tokens per second)
llama_print_timings: prompt eval time =   23841.18 ms /   119 tokens (  200.35 ms per token,     4.99 tokens per second)
llama_print_timings:        eval time =    1473.24 ms /     1 runs   ( 1473.24 ms per token,     0.68 tokens per second)
llama_print_timings:       total time =   29411.73 ms


 ./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 8 -lsize 0.85 -pa 0.3

encoded   25 tokens in    3.615 seconds, speed:    6.916 t/s
decoded   61 tokens in   15.067 seconds, speed:    4.049 t/s

n_draft   = 8
n_predict = 61
n_drafted = 83
n_drafts  = 12
avg_draft = 6.917
n_accept  = 48
accept    = 57.831%

draft:

llama_print_timings:        load time =      49.03 ms
llama_print_timings:      sample time =     171.73 ms /     1 runs   (  171.73 ms per token,     5.82 tokens per second)
llama_print_timings: prompt eval time =     328.53 ms /    25 tokens (   13.14 ms per token,    76.10 tokens per second)
llama_print_timings:        eval time =    2320.30 ms /    96 runs   (   24.17 ms per token,    41.37 tokens per second)
llama_print_timings:       total time =   18681.94 ms

target:

llama_print_timings:        load time =    1550.58 ms
llama_print_timings:      sample time =      19.04 ms /    61 runs   (    0.31 ms per token,  3203.11 tokens per second)
llama_print_timings: prompt eval time =   14793.74 ms /   119 tokens (  124.32 ms per token,     8.04 tokens per second)
llama_print_timings:        eval time =    1015.88 ms /     1 runs   ( 1015.88 ms per token,     0.98 tokens per second)
llama_print_timings:       total time =   18774.99 ms

 ./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 8 -lsize 1.3 -pa 0.3

encoded   25 tokens in    3.251 seconds, speed:    7.689 t/s
decoded   61 tokens in   13.414 seconds, speed:    4.548 t/s

n_draft   = 8
n_predict = 61
n_drafted = 83
n_accept  = 48
accept    = 57.831%

draft:

llama_print_timings:        load time =      51.15 ms
llama_print_timings:      sample time =     170.68 ms /     1 runs   (  170.68 ms per token,     5.86 tokens per second)
llama_print_timings: prompt eval time =     316.66 ms /    25 tokens (   12.67 ms per token,    78.95 tokens per second)
llama_print_timings:        eval time =    2310.07 ms /    96 runs   (   24.06 ms per token,    41.56 tokens per second)
llama_print_timings:       total time =   16664.94 ms

target:

llama_print_timings:        load time =    1578.12 ms
llama_print_timings:      sample time =      19.24 ms /    61 runs   (    0.32 ms per token,  3170.31 tokens per second)
llama_print_timings: prompt eval time =   12968.51 ms /   119 tokens (  108.98 ms per token,     9.18 tokens per second)
llama_print_timings:        eval time =     848.42 ms /     1 runs   (  848.42 ms per token,     1.18 tokens per second)
llama_print_timings:       total time =   16758.66 ms

 ./speculative \
-m ./models/ggml-model-llama-7b-q4_0-4096.gguf \
-md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf \
-p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" \
-e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 8 -lsize 1.6 -pa 0.3

encoded   25 tokens in    3.192 seconds, speed:    7.832 t/s
decoded   61 tokens in   13.073 seconds, speed:    4.666 t/s

n_draft   = 8
n_predict = 61
n_drafted = 83
n_accept  = 48
accept    = 57.831%

draft:

llama_print_timings:        load time =      49.40 ms
llama_print_timings:      sample time =     171.25 ms /     1 runs   (  171.25 ms per token,     5.84 tokens per second)
llama_print_timings: prompt eval time =     315.83 ms /    25 tokens (   12.63 ms per token,    79.16 tokens per second)
llama_print_timings:        eval time =    2382.13 ms /    96 runs   (   24.81 ms per token,    40.30 tokens per second)
llama_print_timings:       total time =   16264.49 ms

target:

llama_print_timings:        load time =    1903.99 ms
llama_print_timings:      sample time =      19.70 ms /    61 runs   (    0.32 ms per token,  3096.60 tokens per second)
llama_print_timings: prompt eval time =   12568.03 ms /   119 tokens (  105.61 ms per token,     9.47 tokens per second)
llama_print_timings:        eval time =     775.86 ms /     1 runs   (  775.86 ms per token,     1.29 tokens per second)
llama_print_timings:       total time =   16356.99 ms

### Tree-based speculative decoding

./speculative -m ./models/ggml-model-llama-7b-q4_0-4096.gguf -md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf -p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" -e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 8 -lsize 0.85 -pa 0.3 -pfz 0.3 -np 2

encoded   25 tokens in    3.388 seconds, speed:    7.378 t/s
decoded   61 tokens in   14.756 seconds, speed:    4.134 t/s

n_draft   = 8
n_predict = 61
n_drafted = 62
n_drafts  = 12
avg_draft = 5.167
n_accept  = 48
accept    = 77.419%

draft:

llama_print_timings:        load time =      48.79 ms
llama_print_timings:      sample time =     164.95 ms /     1 runs   (  164.95 ms per token,     6.06 tokens per second)
llama_print_timings: prompt eval time =    1133.74 ms /    75 tokens (   15.12 ms per token,    66.15 tokens per second)
llama_print_timings:        eval time =    1198.38 ms /    50 runs   (   23.97 ms per token,    41.72 tokens per second)
llama_print_timings:       total time =   18144.97 ms

target:

llama_print_timings:        load time =    1548.64 ms
llama_print_timings:      sample time =      19.08 ms /    61 runs   (    0.31 ms per token,  3196.73 tokens per second)
llama_print_timings: prompt eval time =   13563.79 ms /   122 tokens (  111.18 ms per token,     8.99 tokens per second)
llama_print_timings:        eval time =    2029.37 ms /     2 runs   ( 1014.68 ms per token,     0.99 tokens per second)
llama_print_timings:       total time =   18237.13 ms

./speculative -m ./models/ggml-model-llama-7b-q4_0-4096.gguf -md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf -p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" -e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 8 -lsize 1.3 -pa 0.3 -pfz 0.3 -np 2

encoded   25 tokens in    3.184 seconds, speed:    7.852 t/s
decoded   61 tokens in   13.215 seconds, speed:    4.616 t/s

n_draft   = 8
n_predict = 61
n_drafted = 62
n_drafts  = 12
avg_draft = 5.167
n_accept  = 48
accept    = 77.419%

draft:

llama_print_timings:        load time =      49.10 ms
llama_print_timings:      sample time =     165.38 ms /     1 runs   (  165.38 ms per token,     6.05 tokens per second)
llama_print_timings: prompt eval time =    1141.10 ms /    75 tokens (   15.21 ms per token,    65.73 tokens per second)
llama_print_timings:        eval time =    1246.63 ms /    50 runs   (   24.93 ms per token,    40.11 tokens per second)
llama_print_timings:       total time =   16398.29 ms

target:

llama_print_timings:        load time =    1745.20 ms
llama_print_timings:      sample time =      19.61 ms /    61 runs   (    0.32 ms per token,  3110.50 tokens per second)
llama_print_timings: prompt eval time =   12109.18 ms /   122 tokens (   99.26 ms per token,    10.07 tokens per second)
llama_print_timings:        eval time =    1684.48 ms /     2 runs   (  842.24 ms per token,     1.19 tokens per second)
llama_print_timings:       total time =   16490.74 ms

./speculative -m ./models/ggml-model-llama-7b-q4_0-4096.gguf -md ./models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf -p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" -e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 8 -lsize 1.6 -pa 0.3 -pfz 0.3 -np 2

encoded   25 tokens in    3.097 seconds, speed:    8.072 t/s
decoded   61 tokens in   12.991 seconds, speed:    4.696 t/s

n_draft   = 8
n_predict = 61
n_drafted = 62
n_drafts  = 12
avg_draft = 5.167
n_accept  = 48
accept    = 77.419%

draft:

llama_print_timings:        load time =      49.90 ms
llama_print_timings:      sample time =     167.02 ms /     1 runs   (  167.02 ms per token,     5.99 tokens per second)
llama_print_timings: prompt eval time =    1150.78 ms /    75 tokens (   15.34 ms per token,    65.17 tokens per second)
llama_print_timings:        eval time =    1198.49 ms /    50 runs   (   23.97 ms per token,    41.72 tokens per second)
llama_print_timings:       total time =   16088.20 ms

target:

llama_print_timings:        load time =    1744.64 ms
llama_print_timings:      sample time =      20.72 ms /    61 runs   (    0.34 ms per token,  2944.44 tokens per second)
llama_print_timings: prompt eval time =   12024.30 ms /   122 tokens (   98.56 ms per token,    10.15 tokens per second)
llama_print_timings:        eval time =    1493.47 ms /     2 runs   (  746.74 ms per token,     1.34 tokens per second)
llama_print_timings:       total time =   16181.05 ms

sudo cgexec -g memory:2group  ./speculative -m ../llama.cpp/models/ggml-model-llama-7b-q4_0-4096.gguf -md ../llama.cpp/models/ggml-model-tinyllama-1.1b-q4_0-4096.gguf -p "// Quick-sort implementation in C (4 spaces indentation + detailed comments) and sample usage:\n\n#include" -e -ngl 1 -t 4 -n 56 -c 512 -s 8 --top_k 1 --draft 8 -pa 0.3 -np 2

encoded   25 tokens in    4.605 seconds, speed:    5.429 t/s
decoded   61 tokens in   23.664 seconds, speed:    2.578 t/s

n_draft   = 8
n_predict = 61
n_drafted = 62
n_drafts  = 12
avg_draft = 5.167
n_accept  = 48
accept    = 77.419%

draft:

llama_print_timings:        load time =     565.79 ms
llama_print_timings:      sample time =     171.25 ms /     1 runs   (  171.25 ms per token,     5.84 tokens per second)
llama_print_timings: prompt eval time =    1191.99 ms /    75 tokens (   15.89 ms per token,    62.92 tokens per second)
llama_print_timings:        eval time =    1227.55 ms /    50 runs   (   24.55 ms per token,    40.73 tokens per second)
llama_print_timings:       total time =   28301.35 ms

target:

llama_print_timings:        load time =    4200.69 ms
llama_print_timings:      sample time =      19.28 ms /    61 runs   (    0.32 ms per token,  3163.41 tokens per second)
llama_print_timings: prompt eval time =   22322.71 ms /   122 tokens (  182.97 ms per token,     5.47 tokens per second)
llama_print_timings:        eval time =    3301.47 ms /     2 runs   ( 1650.74 ms per token,     0.61 tokens per second)
llama_print_timings:       total time =   28920.05 ms

encoded   25 tokens in    4.620 seconds, speed:    5.412 t/s
decoded   61 tokens in   23.339 seconds, speed:    2.614 t/s

n_draft   = 8
n_predict = 61
n_drafted = 62
n_drafts  = 12
avg_draft = 5.167
n_accept  = 48
accept    = 77.419%

draft:

llama_print_timings:        load time =     572.80 ms
llama_print_timings:      sample time =     171.04 ms /     1 runs   (  171.04 ms per token,     5.85 tokens per second)
llama_print_timings: prompt eval time =    1158.34 ms /    75 tokens (   15.44 ms per token,    64.75 tokens per second)
llama_print_timings:        eval time =    1630.04 ms /    50 runs   (   32.60 ms per token,    30.67 tokens per second)
llama_print_timings:       total time =   27960.21 ms

target:

llama_print_timings:        load time =    4044.78 ms
llama_print_timings:      sample time =      18.94 ms /    61 runs   (    0.31 ms per token,  3220.70 tokens per second)
llama_print_timings: prompt eval time =   21896.52 ms /   122 tokens (  179.48 ms per token,     5.57 tokens per second)
llama_print_timings:        eval time =    3048.37 ms /     2 runs   ( 1524.18 ms per token,     0.66 tokens per second)
llama_print_timings:       total time =   28582.02 ms

encoded   25 tokens in    4.514 seconds, speed:    5.538 t/s
decoded   61 tokens in   24.842 seconds, speed:    2.456 t/s

n_draft   = 8
n_predict = 61
n_drafted = 62
n_drafts  = 12
avg_draft = 5.167
n_accept  = 48
accept    = 77.419%

draft:

llama_print_timings:        load time =     512.37 ms
llama_print_timings:      sample time =     170.27 ms /     1 runs   (  170.27 ms per token,     5.87 tokens per second)
llama_print_timings: prompt eval time =    1206.08 ms /    75 tokens (   16.08 ms per token,    62.18 tokens per second)
llama_print_timings:        eval time =    2235.11 ms /    50 runs   (   44.70 ms per token,    22.37 tokens per second)
llama_print_timings:       total time =   29387.64 ms

target:

llama_print_timings:        load time =    4903.50 ms
llama_print_timings:      sample time =      19.05 ms /    61 runs   (    0.31 ms per token,  3201.43 tokens per second)
llama_print_timings: prompt eval time =   22663.90 ms /   122 tokens (  185.77 ms per token,     5.38 tokens per second)
llama_print_timings:        eval time =    3020.79 ms /     2 runs   ( 1510.40 ms per token,     0.66 tokens per second)
llama_print_timings:       total time =   29950.42 ms