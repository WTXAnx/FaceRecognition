#include "cv.h" 
#include "highgui.h"

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <assert.h> 
#include <math.h> 
#include <float.h> 
#include <limits.h> 
#include <time.h> 
#include <ctype.h>

#ifdef _EiC 
#define WIN32 
#endif

static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;

void detect_and_draw(IplImage* img)
{
	static CvScalar color = { 0, 0, 255 };
	IplImage* gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);
	IplImage* small_img = cvCreateImage(cvSize(cvRound(img->width), cvRound(img->height)), 8, 1);
	cvCvtColor(img, gray, CV_BGR2GRAY);
	cvResize(gray, small_img, CV_INTER_LINEAR);

	cvEqualizeHist(small_img, small_img);
	cvClearMemStorage(storage);
	double t = (double)cvGetTickCount();
	CvSeq* objects = cvHaarDetectObjects(small_img, cascade, storage, 1.1, 2, 0, cvSize(30, 30));

	t = (double)cvGetTickCount() - t;
	printf("detection time = %gms\n", t / ((double)cvGetTickFrequency() * 1000));

	for (int i = 0; i < (objects ? objects->total : 0); i++)
	{
		CvRect* r = (CvRect*)cvGetSeqElem(objects, i);
		CvPoint center;
		int radius;
		center.x = cvRound((r->x + r->width*0.5));
		center.y = cvRound((r->y + r->height*0.5));
		radius = cvRound((r->width + r->height)*0.25);
		cvCircle(img, center, radius, color, 3, 8, 0);
	}
	cvShowImage("result", img);
	cvReleaseImage(&gray);
	cvReleaseImage(&small_img);
}

const char* cascade_name =
"haarcascade_frontalface_alt2.xml";

int main(int argc, char** argv)
{
	cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);

	if (!cascade)
	{
		fprintf(stderr, "ERROR: Could not load classifier cascade\n");
		return -1;
	}
	storage = cvCreateMemStorage(0);
	cvNamedWindow("result", 0);

	const char* filename = "test_image.jpg";
	IplImage* image = cvLoadImage(filename, 1);

	if (image)
	{
		detect_and_draw(image);
		cvWaitKey(0);
		cvReleaseImage(&image);
	}
	system("pause");
	cvDestroyWindow("result");
	
	return 0;
}


