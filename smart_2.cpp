#include <cuda_fp16.h>        // data types
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE

#include <string.h>
#include <time.h>
#include <set>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "mmio.c"
#include "smsh.c"

#define SM_CORES 108
#define MINIMUM_DENSITY 0
#define MAXIMUM_GHOST_PERC 0.0000000000000000000001
#define A_ELL_BLOCKSIZE 16
#define B_NUM_COLS 64

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

const int EXIT_UNSUPPORTED = 2;

__half* createRandomArray(long int n) {
    __half* array = new __half[n];

    for (int i = 0; i < n; i++) { 
        array[i] = 1.0;
    }

    return array;
}

/* Counts number of blocks in a row of blocks */
int findNumBlocks(int *rowPtr, int *colIndex, int rowId, int block_size) {

    std::unordered_map<int, int> hashMap;

    for (int j = 0; j < block_size; j++) {
        long int id = rowId + j;
        for (int k = rowPtr[id]; k < rowPtr[id + 1]; k++) {
            int bucket = (colIndex[k] + block_size - 1) / block_size;
            if (((colIndex[k]) % block_size) != 0)
                bucket--;
            auto it = hashMap.find(bucket);
            if (it == hashMap.end())
                hashMap.insert({bucket, 1});
        }
    }

    return hashMap.size();
}

/* Finds the possible amount of column blocks the matrix can have */
int findMaxNnz(int *rowPtr, int *colIndex, int num_rows, int block_size) {

    int max = 0;
    int num_blocks = num_rows / block_size;

    std::set<int> mySet;

    for(int i=0; i < num_blocks; i++) {

        for (int j = 0; j<block_size; j++) {
            long int id = (long int)block_size*i+j;
            
            for(int k=rowPtr[id]; k<rowPtr[id+1]; k++)
                mySet.insert(colIndex[k]/block_size);
            
            if (mySet.size() > max)
                max = mySet.size();
        }
        mySet.clear();
    }

    return max*block_size;
}

/* Creates the array of block indexes for the blocked ell format */
int *createBlockIndex(int *rowPtr, int *colIndex, int num_rows, int block_size, int ell_cols, int &realBlocks) {

    long int mb = num_rows/block_size, nb = ell_cols/block_size;
    if (num_rows % block_size != 0)
        mb++;

    int* hA_columns = new int[(long int)nb*mb]();
    long int ctr = 0;

    memset(hA_columns, -1, (long int) nb * mb * sizeof(int));
    std::set<int> mySet;

    // Goes through the blocks of the matrix of block_size
    for(int i=0; i<mb; i++) {

        // Iterates through the rows of each block
        for (int j = 0; j < block_size; j++) {
            long int id = (long int) block_size*i + j;
            int index = 0;
            if (id >= num_rows)
                break;

            // Iterates over the nnzs of each row
            for(int k=rowPtr[id]; k<rowPtr[id+1]; k++) {    
                index = (colIndex[k]/block_size);
                mySet.insert(index);
            }
        }
        for (std::set<int>::iterator it =mySet.begin(); it != mySet.end(); it++) {
            int elem = *it;
            hA_columns[ctr++] = elem;
            realBlocks++;
        }
        
        ctr = (long int) i*nb+nb;
        mySet.clear();
    }

    return hA_columns; 
}

/* Creates the array of values for the blocked ell format */
__half *createValueIndex(int *rowPtr, int *colIndex, float *values, int *hA_columns, int num_rows, int block_size, int ell_cols) {

    /* Allocate enough memory for the array */
    __half* hA_values = new __half[(long int)num_rows * ell_cols]();

    long int mb = (long int) num_rows/block_size, nb = (long int) ell_cols/block_size;
    if (num_rows % block_size != 0)
        mb++;

    // Set all values to 0
    memset(hA_values, 0, (long int) num_rows * ell_cols * sizeof(__half));

    // Iterate the blocks in the y axis
    for (int i=0; i<mb;i++){

        // store the block indexes of this line
        std::unordered_map<int, int> myHashMap; 
        for (int l = 0; l < nb; l++) {
            long int id = (long int) nb*i + l;
            if(hA_columns[id] == -1)
                break;
            else
                myHashMap[hA_columns[id]] = l;
        }

        // Iterate the lines of each block
        for (int l = 0; l<block_size; l++) {

            // Iterate a single line of the matrix
            for(int k=rowPtr[(long int)i*block_size+l]; k<rowPtr[(long int)i*block_size+l+1]; k++) {  
                int id = myHashMap[colIndex[k]/block_size];
                hA_values[(long int)i*ell_cols*block_size + l*ell_cols + id*block_size + (colIndex[k] - (colIndex[k]/block_size) *block_size)] = values[k];
            }
        }
    }
    
    return hA_values;
}

int* removeEmptyColumns(int *&rowPtr_part, int *&colIndex, int A_rows, int A_cols, int A_nnz) {

    //create vector to store non-empy columns
    std::set<int>nonEmptyCols;

    //go through all elements in partition
    //store columns that have nnzs
    for (int i=0; i<A_rows; i++) {
        for (int k = rowPtr_part[i]; k < rowPtr_part[i+1]; k++) {
            nonEmptyCols.insert(colIndex[k]);
        }
    }

    //columns with nnzs will shift to the left
    int *colIndexNew = new int[A_nnz];
    int *colCompress = new int[A_cols];
    int ctr = 0;
    for(int i:nonEmptyCols) {
        colCompress[i] = ctr++;
    }

    //copy new values to colIndex
    for (int i=0; i<A_rows; i++) {
        for (int k = rowPtr_part[i]; k < rowPtr_part[i+1]; k++) {
            colIndexNew[k] = colCompress[colIndex[k]];
        }
    }

    return colIndexNew;
}

int main(int argc, char *argv[]) {

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int A_num_rows, A_num_cols, nz, A_nnz;
    int i = 0, *I_complete, *J_complete;
    float *V_complete;
    
    /* READ MTX FILE INTO CSR MATRIX */
    /************************************************************************************************************/
    if ((f = fopen(argv[1], "r")) == NULL)
    {
        printf("Could not locate the matrix file. Please make sure the pathname is valid.\n");
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    matcode[4] = '\0';
    
    if ((ret_code = mm_read_mtx_crd_size(f, &A_num_rows, &A_num_cols, &nz)) != 0)
    {
        printf("Could not read matrix dimensions.\n");
        exit(1);
    }
    
    if ((strcmp(matcode, "MCRG") == 0) || (strcmp(matcode, "MCIG") == 0) || (strcmp(matcode, "MCPG") == 0) || (strcmp(matcode, "MCCG") == 0))
    {

        I_complete = (int *)calloc(nz, sizeof(int));
        J_complete = (int *)calloc(nz, sizeof(int));
        V_complete = (float *)calloc(nz, sizeof(float));

        for (i = 0; i < nz; i++)
        {
            if (matcode[2] == 'P') {
                fscanf(f, "%d %d", &I_complete[i], &J_complete[i]);
                V_complete[i] = 1;
            }  
            else {
                fscanf(f, "%d %d %f", &I_complete[i], &J_complete[i], &V_complete[i]);
            } 
            fscanf(f, "%*[^\n]\n");
            /* adjust from 1-based to 0-based */
            I_complete[i]--;
            J_complete[i]--;
        }
    }

    /* If the matrix is symmetric, we need to construct the other half */

    else if ((strcmp(matcode, "MCRS") == 0) || (strcmp(matcode, "MCIS") == 0) || (strcmp(matcode, "MCPS") == 0) || (strcmp(matcode, "MCCS") == 0) || (strcmp(matcode, "MCCH") == 0) || (strcmp(matcode, "MCRK") == 0) || (matcode[0] == 'M' && matcode[1] == 'C' && matcode[2] == 'P' && matcode[3] == 'S'))
    {

        I_complete = (int *)calloc(2 * nz, sizeof(int));
        J_complete = (int *)calloc(2 * nz, sizeof(int));
        V_complete = (float *)calloc(2 * nz, sizeof(float));

        int i_index = 0;

        for (i = 0; i < nz; i++)
        {
            if (matcode[2] == 'P') {
                fscanf(f, "%d %d", &I_complete[i], &J_complete[i]);
                V_complete[i] = 1;
            }
            else {
                fscanf(f, "%d %d %f", &I_complete[i], &J_complete[i], &V_complete[i]);
            }
                
            fscanf(f, "%*[^\n]\n");

            if (I_complete[i] == J_complete[i])
            {
                /* adjust from 1-based to 0-based */
                I_complete[i]--;
                J_complete[i]--;
            }
            else
            {
                /* adjust from 1-based to 0-based */
                I_complete[i]--;
                J_complete[i]--;
                J_complete[nz + i_index] = I_complete[i];
                I_complete[nz + i_index] = J_complete[i];
                V_complete[nz + i_index] = V_complete[i];
                i_index++;
            }
        }
        nz += i_index;
    }
    else
    {
        printf("This matrix type is not supported: %s \n", matcode);
        exit(1);
    }

    /* sort COO array by the rows */
    if (!isSorted(J_complete, I_complete, nz)) {
        quicksort(J_complete, I_complete, V_complete, nz);
    }
    
    /* Convert from COO to CSR */
    int* rowPtr = new int[A_num_rows + 1]();
    int* colIndex = new int[nz]();
    float* values = new float[nz]();

    for (i = 0; i < nz; i++) {
        colIndex[i] = J_complete[i];
        values[i] = V_complete[i];
        rowPtr[I_complete[i] + 1]++;
    }
    for (i = 0; i < A_num_rows; i++) {
        rowPtr[i + 1] += rowPtr[i];
    }
    A_nnz = nz;

    free(I_complete);
    free(J_complete);
    free(V_complete);
    fclose(f);
    /* MTX READING IS FINISH */
    /************************************************************************************************************/

    int   A_ell_blocksize = A_ELL_BLOCKSIZE;
    int k = 0, ctr = 0;
    std::vector<int> rowsOfBlocksPerPartition;
    std::vector<int> colsOfBlocksPerPartition;
    
    // Pad matrix with extra rows to fit block size O(n_rows)
    int * rowPtr_pad;
    int remainder = A_num_rows % A_ell_blocksize;
    if (remainder != 0) {
        A_num_rows = A_num_rows + (A_ell_blocksize - remainder);
        A_num_cols = A_num_cols + (A_ell_blocksize - remainder);
        rowPtr_pad = new int[A_num_rows + 1];
        for (int i=0; i<A_num_rows - (A_ell_blocksize - remainder); i++)
            rowPtr_pad[i] = rowPtr[i];
        for (int j=A_num_rows - (A_ell_blocksize - remainder); j<A_num_rows + 1; j++)
            rowPtr_pad[j] = nz;
        delete[] rowPtr;
    } else {
        rowPtr_pad = rowPtr;
    }

    // Split matrix into k parts 
    int aux = 0;
    int nBlocks = findNumBlocks(rowPtr_pad, colIndex, ctr, A_ell_blocksize);
    int nBlocks_next = 0, blockCtr = 0, ghostBlockCtr = 0;

    while (ctr < A_num_rows) {
        blockCtr = nBlocks, ghostBlockCtr = 0;
        ctr += A_ell_blocksize;
        while (ctr < A_num_rows) {
            nBlocks_next = findNumBlocks(rowPtr_pad, colIndex, ctr, A_ell_blocksize);
            ghostBlockCtr += nBlocks - nBlocks_next;
            blockCtr += nBlocks;
            if((double)ghostBlockCtr/blockCtr >= MAXIMUM_GHOST_PERC) {
            	colsOfBlocksPerPartition.push_back(nBlocks);
                nBlocks = nBlocks_next;
                break;
            }
            ctr += A_ell_blocksize;
        }
        rowsOfBlocksPerPartition.push_back((ctr-aux)/A_ell_blocksize);
        k++;
        aux = ctr;
    }
    ctr = 0;

    // Information storing arrays
    int*blocksPerPart = new int[k];
    int*ghostBlocksPerPartition = new int[k];
    int totalGhostBlocks = 0;
    double*densityPart = new double[k];

    // Initialize variables
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA[k];
    cusparseDnMatDescr_t matB, matC[k];
    void**                dBuffer    = new void*[k]();

    int    *dA_columns[k];
    __half *dA_values[k], *dB, *dC[k];

    float alpha           = 1.0f;
    float beta            = 0.0f;

    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = B_NUM_COLS;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    long int   B_size          = (long int) ldb * B_num_cols;
    long int   C_size          = (long int) ldc * B_num_cols;

    __half *hB            = createRandomArray(B_size);

    // Allocate and copy memory in GPU for dense vectors
    CHECK_CUDA( cudaMalloc((void**) &dB, (long int) B_size * sizeof(__half)) )

    CHECK_CUDA( cudaMemcpy(dB, hB, (long int) B_size * sizeof(__half),
                        cudaMemcpyHostToDevice) )

    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
					CUDA_R_16F, CUSPARSE_ORDER_COL) )

    int total_blocks = 0;
    // Create blocked ELL for each partition of matrix A
    for (int i=0; i<k; i++){
        size_t bufferSize = 0;

        int *rowPtr_part = new int[rowsOfBlocksPerPartition[i]*A_ell_blocksize + 1];
        int A_rows = 0;
        for (int j=ctr; j<ctr+rowsOfBlocksPerPartition[i]*A_ell_blocksize; j++){
            if (j >= A_num_rows)
                break;
            rowPtr_part[j - ctr] = rowPtr_pad[j];
            A_rows++;
        }    
        ctr += A_rows;
        rowPtr_part[A_rows] = rowPtr_pad[ctr];
        long int nnzs_part = rowPtr_part[A_rows] - rowPtr_part[0];

        int *colIndexCompress;
        if (( (double)nnzs_part / (double) (colsOfBlocksPerPartition[i]*rowsOfBlocksPerPartition[i])) >= MINIMUM_DENSITY)
            colIndexCompress = colIndex;
        else
            colIndexCompress = removeEmptyColumns(rowPtr_part, colIndex, A_rows, A_num_rows, A_nnz);

        // Create blocked ELL vectors for partition
        int   A_ell_cols      = findMaxNnz(rowPtr_part, colIndexCompress, A_rows, A_ell_blocksize);
        double   A_num_blocks    = (double)A_ell_cols * (double)A_rows /
                            (A_ell_blocksize * A_ell_blocksize);
        int realBlocks = 0;
        int   *hA_columns     = createBlockIndex(rowPtr_part, colIndexCompress, A_rows, A_ell_blocksize, A_ell_cols, realBlocks);
        __half *hA_values     = createValueIndex(rowPtr_part, colIndexCompress, values, hA_columns, A_rows, A_ell_blocksize, A_ell_cols);

	    __half *hC 	      = new __half[(long int) A_rows * B_num_cols * sizeof(__half)];

        // Allocate and copy memory in GPU for blocked ELL vectors
        CHECK_CUDA( cudaMalloc((void**) &dA_columns[i], (long int) A_num_blocks * sizeof(int)) )
        CHECK_CUDA( cudaMalloc((void**) &dA_values[i],
                                        (long int) A_ell_cols * A_rows * sizeof(__half)) )
        CHECK_CUDA( cudaMalloc((void**) &dC[i], (long int) A_rows * B_num_cols * sizeof(__half)) )
        
        CHECK_CUDA( cudaMemcpy(dA_columns[i], hA_columns,
                            (long int) A_num_blocks * sizeof(int),
                            cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(dA_values[i], hA_values,
                            (long int) A_ell_cols * A_rows * sizeof(__half),
                            cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(dC[i], hC, (long int) A_rows * B_num_cols * sizeof(__half),
                            cudaMemcpyHostToDevice) )
                            
        CHECK_CUSPARSE( cusparseCreateDnMat(&matC[i], A_rows, B_num_cols, A_rows, dC[i],
					CUDA_R_16F, CUSPARSE_ORDER_COL) )
        //--------------------------------------------------------------------------
        
        CHECK_CUSPARSE( cusparseCreate(&handle) )
        // Create sparse matrix A in blocked ELL format
        CHECK_CUSPARSE( cusparseCreateBlockedEll(
                                        &matA[i],
                                        A_rows, A_num_cols, A_ell_blocksize,
                                        A_ell_cols, dA_columns[i], dA_values[i],
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
        
        // allocate an external buffer if needed
        CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                    handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA[i], matB, &beta, matC[i], CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
        CHECK_CUDA( cudaMalloc(&dBuffer[i], bufferSize) )

        total_blocks += A_num_blocks;
        blocksPerPart[i] = A_num_blocks;
        densityPart[i] = (double) nnzs_part/A_num_blocks;
	//std::cout << densityPart[i] << std::endl;
        ghostBlocksPerPartition[i] = A_num_blocks - realBlocks;
	totalGhostBlocks += A_num_blocks - realBlocks;
    }
    struct timespec t_start, t_end;
    double elapsedTime, searchTime = 0;
    int numRuns=0;

   /* for (int i=0; i<k; i++) {
        CHECK_CUSPARSE( cusparseSpMM(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA[i], matB, &beta, matC[i], CUDA_R_32F,
                        CUSPARSE_SPMM_ALG_DEFAULT, dBuffer[i]) )
        cudaDeviceSynchronize();

        // execute SpMM
        clock_gettime(CLOCK_MONOTONIC, &t_start);       // initial timestamp
        while (1) {
            CHECK_CUSPARSE( cusparseSpMM(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA[i], matB, &beta, matC[i], CUDA_R_32F,
                                        CUSPARSE_SPMM_ALG_DEFAULT, dBuffer[i]) )
            cudaDeviceSynchronize();
            numRuns++;

            clock_gettime(CLOCK_MONOTONIC, &t_end);         // final timestamp
            elapsedTime = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));
            if(elapsedTime > 5.0f) {
                break;
            }
        }
        
        searchTime += ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000))) / numRuns;
        double time_part = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000))) / numRuns;
        double perf_part = (2* ((double)blocksPerPart[i]) * ((double)A_ell_blocksize*A_ell_blocksize) * ((double)B_num_cols) / 1000000000000) / time_part;
        std::cout << i << "\tNumber Blocks: " << blocksPerPart[i] << "\tRows of blocks: " 
                  << rowsOfBlocksPerPartition[i] << "\tColumns of blocks: " << colsOfBlocksPerPartition[i] << "\tDensity: " 
                  << densityPart[i] << "\tGhost blocks: " << ghostBlocksPerPartition[i] 
                  << " ( " << (double)ghostBlocksPerPartition[i]/(double)blocksPerPart[i]*(double)100 << " %)" << "\tPerf: " << perf_part << std::endl;
        numRuns = 0;
    }*/
    
    std::vector<cudaStream_t> streams(k);
    for (int i = 0; i < k; ++i) {
	    cudaStreamCreate(&streams[i]);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_start);
    while (1) {
        
        for (int i=0; i<k; i++) {
	        CHECK_CUSPARSE( cusparseSetStream(handle, streams[i]) );
            CHECK_CUSPARSE( cusparseSpMM(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA[i], matB, &beta, matC[i], CUDA_R_32F,
                                        CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
            //cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();
        numRuns++;

        clock_gettime(CLOCK_MONOTONIC, &t_end);         // final timestamp

        elapsedTime = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));

        if(elapsedTime > 5.0f) {
            break;
        }        
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end); // final timestamp
    searchTime = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000))) / numRuns;
    double tflops_bell = (2* ((double)total_blocks) * ((double)A_ell_blocksize*A_ell_blocksize) * ((double)B_num_cols) / 1000000000000) / searchTime;
    std::cout << argv[1];
    std::cout << "\tParts: " << k <<"\tTotal Blocks: " << total_blocks << " (density: " << (double) A_nnz/total_blocks << " )" << "\tTime (seconds): " << searchTime << "\tPerf: " << tflops_bell << "\tGhost Blocks: " << totalGhostBlocks << " ( " << (double)totalGhostBlocks/(double)total_blocks*(double)100 << " %)" <<std::endl;

    // destroy matrix/vector descriptors
    for (int i=0; i<k; i++) {
        CHECK_CUSPARSE( cusparseDestroySpMat(matA[i]) )
	    CHECK_CUSPARSE( cusparseDestroyDnMat(matC[i]) )
    }
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    __half *hC[k];
    for (int i=0; i<k; i++){
	hC[i] = new __half[(long int) rowsOfBlocksPerPartition[i] * A_ell_blocksize * B_num_cols * sizeof(__half)];
    	CHECK_CUDA( cudaMemcpy(hC[i], dC[i], (long int) rowsOfBlocksPerPartition[i] * A_ell_blocksize * B_num_cols * sizeof(__half),
                        	cudaMemcpyDeviceToHost) )
    }
        
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dB) )
    for (int i=0; i<k; i++) {
        CHECK_CUDA( cudaFree(dBuffer[i]) )
        CHECK_CUDA( cudaFree(dC[i]) )
        CHECK_CUDA( cudaFree(dA_columns[i]) )
        CHECK_CUDA( cudaFree(dA_values[i]) )
        cudaStreamDestroy(streams[i]);
    }
    
    delete[] hB;
    delete[] rowPtr_pad;
    delete[] colIndex;
    delete[] values;
    return EXIT_SUCCESS;
}
