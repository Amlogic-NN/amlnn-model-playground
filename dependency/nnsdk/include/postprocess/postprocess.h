#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <stdint.h>
#include "nn_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct __box
{
    float x;
    float y;
    float w;
    float h;
    float score;
    float objectClass;
}detBox;

typedef struct __point
{
    float x;
    float y;
}point_t;

typedef struct{
    int index;
    int classId;
    float **probs;
} sortable_bbox;

typedef struct{
    float x, y;
} landmark;

typedef struct{
    float x, y, w, h, prob_obj;
} box;

typedef struct plate_box_{
    float x, y, w, h;
}plate_box;

typedef struct{
    int index;
    int classId;
    float probs;  //**probs to probs
} sortable_bbox_plate;

/*=================================================
output of aml_module_t type:IMAGE_CLASSIFY
score: top 5 class confidence
top:   the class of top 5
=================================================*/
typedef struct __nn_image_classify
{
    float   score[5];
    unsigned int  topClass[5];
}img_classify_out_t;

/*=================================================
output of aml_module_t type:OBJECT_DETECT
detNum: the number of object detect
pBox:   the object info detect
=================================================*/
typedef struct __nn_obj_detect
{
    unsigned int  detNum;
    detBox *pBox;
}obj_detect_out_t;

/*=================================================
output of aml_module_t type:FACE_DETECTION
detNum: the number of face detect
pBox:   the face info detect
=================================================*/
typedef struct __nn_face_detect
{
    unsigned int  detNum;
    detBox pBox[MAX_DETECT_NUM];
}face_detect_out_t;

/*=================================================
output of aml_module_t type:FACE_LANDMARK_5
pos value:
0:left-eye,
1:right-eye,
2:nose,
3:left-mouth,
4:right-mouth
=================================================*/
typedef struct __nn_face_landmark_5
{
    unsigned int   detNum;
    detBox facebox[MAX_DETECT_NUM];
    point_t pos[MAX_DETECT_NUM][5];
}face_landmark5_out_t;

/*=================================================
output of aml_module_t type:FACE_RECOGNIZE
faceVector: the vector generate by input face,
can recognized by this vector
=================================================*/
typedef struct __nn_face_recognize
{
    float faceVector[512];
}face_recognize_out_t;

/*=================================================
output of aml_module_t type:YOLO_V3
detNum: the number of object detect
pBox:   the object info detect
=================================================*/
typedef struct __nn_yolov3
{
    unsigned int  detNum;
    detBox *pBox;
}yolov3_out_t;

/*=================================================
output of aml_module_t type:YOLO_V5
detNum: the number of object detect
pBox:   the object info detect
=================================================*/
typedef struct __nn_yolov5
{
    unsigned int  detNum;
    detBox *pBox;
}yolov5_out_t;

/*=================================================
output of aml_module_t type:HEAD_DETECTION
the return value are detect region for these type.
=================================================*/
typedef struct __nn_head_detect
{
    obj_detect_out_t headOut;
}head_det_out_t;

/*=================================================
output of aml_module_t type:PERSON_DETECT
detNum: the number of person detect
pBox:   the person info detect
=================================================*/
typedef struct __nn_person_detect
{
    unsigned int  detNum;
    detBox pBox[MAX_DETECT_NUM];
}person_detect_out_t;

typedef struct __nn_plate_detect
{
    unsigned int   detNum;
    float score;
    detBox pBox[MAX_DETECT_NUM];
    point_t pos[MAX_DETECT_NUM][4];
}plate_det_out_t;


void* process_top5_f32(float *buf, unsigned int num);
void* process_top5_i8(int8_t *buf, unsigned int num);
void* process_top5_u8(uint8_t *buf, unsigned int num);
void process_top5_i8_sta_top15(int8_t *buf,int num, int zeroPoint, float scale, unsigned int label, int *top1_ture, int *top5_ture);

void* postprocess_object_detect(nn_output *pout);
void* postprocess_yolov3(nn_output *pout);
void *postprocess_yolov3_dequant(nn_output *pout);
void* postprocess_yolov5s(nn_output *pout);
void* postprocess_yolov5s_dequant(nn_output *pout);

void* postprocess_faceland5(nn_output *pout);
void* postprocess_faceland5_dequant(nn_output *pout);

void* postprocess_headdet(nn_output *pout);

void* postprocess_person_detect(nn_output *pout);

void* postprocess_plate_detect(nn_output *pout);


#ifdef __cplusplus
} //extern "C"
#endif

#endif // POSTPROCESS_H