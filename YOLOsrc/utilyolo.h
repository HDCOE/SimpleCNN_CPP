
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <float.h>



typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;


typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA
} data_type;

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}


typedef struct {
    int w;
    int h;
    int c;
    data_t *data;
} image;


void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

void copy_tensor_32(int N, float * in, data_t *out)
{
    for(int i = 0 ; i < N; ++i) out[i] = in[i];

}
void fill_cpu(int N, data_t ALPHA, data_t *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

data_t im2col_get_pixel(data_t *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return (data_t)0;
    return (data_t)im[col + width*(row + height*channel)];
}

void B2buffer(data_t *B, int bufsize, int totalB, int sizeB)
{

    data_t *tmp = (data_t*) calloc (1, sizeof(data_t) * totalB * sizeB);
 
    printf("totalB %d, sizeB %d\n",totalB, sizeB );
   
   int id = 0;

   int ofset_B = sizeB /49 +1;

   int totalbuffer = totalB * sizeB;

   for (int k = 0; k < ofset_B; ++k)
   {
       for (int j = 0; j < 49; ++j)
        {
             for (int i = 0; i < totalB; ++i)
            {
                int tmpID = totalB*i + j + sizeB*k;
                int bID = sizeB*i + j + k*ofset_B ;

                if(tmpID < totalbuffer && bID < totalbuffer)
                {
                    tmp[tmpID] = B[bID];
                    //printf("tmp [%d] %f : B[%d] %f\n",tmpID, tmp[tmpID], bID , B[bID] );
                }
            }
        }
        printf("finish row ...............\n");
   }

     for (int i = 0; i < sizeB*totalB; ++i)
    {
         //printf("B [%d] %f : tmp [%d] %f\n", i, B[i],i, tmp[i]);
    }
}
// scan input to match with output, scan from top to bottom
void im2col_cpu(data_t* data_im, int channels,  int height,  int width, int ksize,  int stride, int pad, data_t* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col  = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize; // input size x wsize x wsize

    data_t * buf = (data_t*)calloc(1,sizeof(data_t)*height_col*width_col*channels_col);

    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;

      //  printf("h_offset %d, w_offset %d\n",h_offset, w_offset );
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                 
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);       
            }
        }
    }

    //B2buffer(data_col, 49, channels_col, height_col * width_col );
    printf("finish im2col_cpu\n");
}

void gemm_nn_origi(int M, int N, int K, data_t ALPHA, 
        data_t *A, int lda, // weight, size x size
        data_t *B, int ldb, // input 
        data_t *C, int ldc) // output
{
    // scan output and see what the output need from input by fixed weights

     //clock_t start, end;

    int i,j,k;

   // start = clock();

    #pragma omp parallel for
    for(i = 0; i < M; ++i){  // scan no. filter
        printf("m %d\n",i );
        for(k = 0; k < K; ++k){ // scan size weight

            register data_t A_PART = ALPHA*A[i*lda+k];

            //float Atmp = A_PART.to_float();

            for(j = 0; j < N; ++j){ // scan output w x h

                  //C[i*ldc+j] = (float)C[i*ldc+j] + (float)A_PART * (float)A[i*lda+k] * (float)B[k*ldb+j];
                  //C[i*ldc+j] += A_PART*B[k*ldb+j]; 
                
                //float Ctmp, Btmp;
                //Ctmp = C[i*ldc+j].to_float();
                //Btmp = B[k*ldb+j].to_float();

               //Ctmp + Btmp*Atmp;
                C[i*ldc+j] += A_PART;//(float)C[i*ldc+j] + (float)A_PART * (float)B[k*ldb+j];

                 //C[i*ldc+j] = 2 * B[k * ldb + j];
            }
        }
    }

    //end = clock();
    //executeTime(start,end);

}
void gemm_nn(int M, int N, int K, data_t ALPHA, 
        data_t *A, int lda, // weight, size x size
        data_t *B, int ldb, // input 
        data_t *C, int ldc) // output
{
    // scan output and see what the output need from input by fixed weights
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){  // scan no. filter
        for(j = 0; j < N; ++j){ // scan size weight

            for(k = 0; k < K; ++k){ // scan output w x h

                for(int id = 0; id < 49; id++)
                {
                   if(k < K)
                   {    

                        C[i*ldc+j] = (float)C[i*ldc+j] + (float)ALPHA * (float)A[i*lda+k] * (float)B[k*ldb+j];
                        //C[i*ldc+j] += (float)ALPHA * (float)A[i*lda+k] * (float)B[k*ldb+j];
                        k++;
                   }
                }
                k--;
             

            }
        }
    }

}

void gemm_tn(int M, int N, int K, data_t ALPHA, // batch, outputs, inputs, 1
        data_t *A, int lda,  // input, inputs(size)
        data_t *B, int ldb,  // weights, inputs(size)
        data_t *C, int ldc)  // output, outputs(size)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){ // batch
        for(k = 0; k < K; ++k){ // inputs

            register data_t A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){ //outputs

                C[i*ldc+j] = (float) C[i*ldc+j] + (float)A_PART * (float)B[k*ldb+j]; //C[i*ldc+j] += (float)A_PART * (float)B[k*ldb+j]; // output[] += Input[] * weight[]
            }
        }
    }
}

void gemm_nt(int M, int N, int K, data_t ALPHA, 
        data_t *A, int lda, 
        data_t *B, int ldb,
        data_t *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k)
            {
                sum += (float)ALPHA * (float)A[i*lda+k] * (float)B[j*ldb + k];
            }
             C[i*ldc+j] = (float)C[i*ldc+j] + (float)sum;//C[i*ldc+j] += (float)sum;
        }
    }
}



void gemm_tt(int M, int N, int K, data_t ALPHA, 
        data_t *A, int lda, 
        data_t *B, int ldb,
        data_t *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += (float)ALPHA * (float)A[i+k*lda] * (float)B[k+j*ldb];
            }
            C[i*ldc+j] = (float)C[i*ldc+j] +(float)sum;//C[i*ldc+j] += (float)sum;
        }
    }
}

void gemm_cpu(int TA, int TB, int M, int N, int K, data_t ALPHA, 
        data_t *A, int lda, 
        data_t *B, int ldb,
        data_t BETA,
        data_t *C, int ldc)
{
  //  printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
   
             C[i*ldc + j] = (float)C[i*ldc + j] * (float)BETA;         //C[i*ldc + j] *= (float)BETA;           
        }
    }

    if(!TA && !TB) // conv
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc); //gemm_nn_origi(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB) // fc
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm(int TA, int TB, int M, int N, int K, data_t ALPHA, 
        data_t *A, int lda, 
        data_t *B, int ldb,
        data_t BETA,
        data_t *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}


image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (data_t *)calloc(h*w*c, sizeof(data_t));
    return out;
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
     m.data[c*m.h*m.w + y*m.w + x] = (float)m.data[c*m.h*m.w + y*m.w + x]+(float)val;//m.data[c*m.h*m.w + y*m.w + x] += (float)val;
}

void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}
void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);   
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }

    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}


image letterbox_image(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    image boxed = make_image(w, h, im.c);
    fill_image(boxed, .5);
    //int i;
    //for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2); 
    free_image(resized);
    return boxed;
}

image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if(channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    free(data);
    return im;
}
image load_image(char *filename, int w, int h, int c)
{

    image out = load_image_stb(filename, c);

    if((h && w) && (h != out.h || w != out.w)){
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}
image load_image_color(char *filename, int w, int h)
{
    return load_image(filename, w, h, 3);
}


void save_image_png(image im, const char *name)
{
    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s.png", name);
    unsigned char *data = (unsigned char *)calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}
