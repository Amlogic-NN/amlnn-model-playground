#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif


unsigned char *get_jpeg_rawData(const char *name,unsigned int width,unsigned int height);
void *read_file(const char *file_path, int *file_size);
int find_max_index(float *buf, unsigned int length);
void save_file(void * buffer, const char *file_path, unsigned int file_size);
double cosine_similarity(char *file1, char *file2, int file_size);
int bmp_read(const char *filename, int *nRGB, int *xsize, int *ysize);
void data_to_fp32(float *new_buffer, unsigned char *old_buffer, int sz, float scale, int zero_point, int type);


#ifdef __cplusplus
} //extern "C"
#endif

#endif // UTILS_H