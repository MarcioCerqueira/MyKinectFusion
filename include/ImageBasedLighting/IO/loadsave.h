#ifndef _LOAD_SAVE_H_
#define _LOAD_SAVE_H_

#include <opencv2/opencv.hpp>
#include "grfmt_hdr.h"

namespace cv
{
Mat hdrImread( const String& filename, int flags );
bool hdrImwrite( const String& filename, InputArray _img,
              const std::vector<int>& params );
bool hdrImwrite( const String& filename, InputArray _img,
              const std::vector<int>& params );
Mat hdrImdecode( InputArray _buf, int flags );
Mat hdrImdecode( InputArray _buf, int flags, Mat* dst );
bool hdrImencode( const String& ext, InputArray _image,
               std::vector<uchar>& buf, const std::vector<int>& params );
IplImage* cvLoadHDRImage( const char* filename, int iscolor );
CvMat* cvLoadHDRImageM( const char* filename, int iscolor );

}

#endif